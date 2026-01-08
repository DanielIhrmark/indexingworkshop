import json
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/11J5JRtap7p2P8BWl3gsPVs1I-DATbVdOi4Oj1fcSbe0/edit?usp=sharing"

# SAO in id.kb.se
IDKB_FIND_ENDPOINT = "https://id.kb.se/find"
SAO_SCHEME_URI = "https://id.kb.se/term/sao"

BASELINE_FIELDS = ["title", "author", "abstract"]
INDEX_FIELDS = ["keywords_free", "subjects_controlled", "ddc", "sab", "entities"]

# Community Cloud tuning
HTTP_TIMEOUT_SECONDS = 15
HTTP_RETRIES = 2
HTTP_BACKOFF_SECONDS = 1

FIND_LIMIT = 12                 # how many candidate concepts to show
RELATED_URI_LIMIT = 8           # cap on broader/narrower URIs fetched for labels
RELATED_LABEL_FETCH_LIMIT = 8   # labels to resolve (per category)


# -----------------------------
# Google Sheet helpers
# -----------------------------
def extract_sheet_id(sheet_url: str) -> Optional[str]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    return m.group(1) if m else None


def build_gviz_csv_url(sheet_url: str, sheet_name: str = "") -> str:
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        raise ValueError("Could not parse Google Sheet ID.")
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    if sheet_name.strip():
        base += "&sheet=" + urllib.parse.quote(sheet_name.strip())
    return base


@st.cache_data(ttl=120)
def load_sheet_as_df(sheet_url: str, sheet_name: str = "") -> pd.DataFrame:
    csv_url = build_gviz_csv_url(sheet_url, sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str)
    return df


# -----------------------------
# Indexing helpers
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-zÅÄÖåäö0-9]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def build_inverted_index(df: pd.DataFrame, fields: List[str], id_col: str) -> Dict[str, set]:
    inv: Dict[str, set] = {}
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, "")).strip()
        if not doc_id:
            continue
        blob = []
        for f in fields:
            if f in df.columns:
                blob.append(str(row.get(f, "")))
        for tok in tokenize(" ".join(blob)):
            inv.setdefault(tok, set()).add(doc_id)
    return inv


# -----------------------------
# HTTP helper with retries
# -----------------------------
def http_get_json(url: str, params: Optional[dict] = None, accept: str = "application/ld+json") -> Any:
    last_err: Optional[Exception] = None
    headers = {"Accept": accept}

    for attempt in range(0, 1 + HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            r.raise_for_status()

            # Sometimes servers send JSON-LD as text/html; attempt JSON parse anyway
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF_SECONDS * (2 ** attempt))
            else:
                break
    raise last_err  # type: ignore


# -----------------------------
# id.kb.se/find parsing
# -----------------------------
def _extract_uri(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("@id") or obj.get("id")
    return None


def _extract_sv_label(obj: Any) -> Optional[str]:
    """
    Try a variety of label fields commonly seen in JSON-LD/KB payloads.
    """
    if not isinstance(obj, dict):
        return None

    # common keys
    for key in ("prefLabel", "label", "name"):
        v = obj.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            # language map style
            if "sv" in v and isinstance(v["sv"], str):
                return v["sv"]
        if isinstance(v, list):
            # list of language-tagged values
            for item in v:
                if isinstance(item, dict):
                    if item.get("@language") == "sv" and isinstance(item.get("@value"), str):
                        return item["@value"]
                    if item.get("lang") == "sv" and isinstance(item.get("value"), str):
                        return item["value"]
    return None


def _flatten_graph(doc: Any) -> List[dict]:
    """
    Flatten JSON-LD structures to a list of node dicts.
    """
    if isinstance(doc, list):
        nodes = []
        for d in doc:
            nodes.extend(_flatten_graph(d))
        return nodes

    if not isinstance(doc, dict):
        return []

    if "@graph" in doc and isinstance(doc["@graph"], list):
        return [n for n in doc["@graph"] if isinstance(n, dict)]

    # Sometimes it's a single node document
    return [doc]


@st.cache_data(ttl=3600)
def sao_find_candidates(q: str, limit: int = FIND_LIMIT) -> List[dict]:
    """
    Query id.kb.se/find for SAO candidates.
    Uses the scheme facet parameter seen in the id.kb.se UI: and-inScheme.@id=<SAO>.
    :contentReference[oaicite:1]{index=1}
    """
    q = (q or "").strip()
    if len(q) < 2:
        return []

    params = {
        "q": q,
        "_limit": str(limit),
        # observed facet parameter name in the SAO UI URL
        "and-inScheme.@id": SAO_SCHEME_URI,
    }

    # Try JSON-LD via Accept header; if it fails, fall back to trying a format param
    try:
        doc = http_get_json(IDKB_FIND_ENDPOINT, params=params, accept="application/ld+json")
    except Exception:
        # fallback attempt (some services respect _format)
        params2 = dict(params)
        params2["_format"] = "application/ld+json"
        doc = http_get_json(IDKB_FIND_ENDPOINT, params=params2, accept="application/json")

    nodes = _flatten_graph(doc)

    candidates: List[dict] = []
    for n in nodes:
        uri = _extract_uri(n)
        lbl = _extract_sv_label(n)
        if uri and lbl:
            candidates.append({"label": lbl, "uri": uri})

    # de-dup by uri, preserve order
    seen = set()
    out = []
    for c in candidates:
        if c["uri"] not in seen:
            seen.add(c["uri"])
            out.append(c)

    # Heuristic: prefer exact label match first, then shortest label
    q_low = q.lower()
    out.sort(key=lambda x: (0 if x["label"].lower() == q_low else 1, len(x["label"])))
    return out[:limit]


@st.cache_data(ttl=3600)
def idkb_fetch_concept(uri: str) -> dict:
    """
    Fetch a concept URI as JSON-LD.
    """
    try:
        return http_get_json(uri, params=None, accept="application/ld+json")
    except Exception:
        # some resources may require explicit format param
        return http_get_json(uri, params={"_format": "application/ld+json"}, accept="application/json")


def _collect_related_uris(nodes: List[dict], predicate: str) -> List[str]:
    """
    Collect related URIs from nodes for a given predicate (e.g. broader/narrower/altLabel).
    """
    out: List[str] = []
    for n in nodes:
        v = n.get(predicate)
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                u = _extract_uri(item)
                if u:
                    out.append(u)
        else:
            u = _extract_uri(v)
            if u:
                out.append(u)
    # de-dup, preserve order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _collect_alt_labels(nodes: List[dict]) -> List[str]:
    """
    altLabel is usually literal strings (or language-tagged values) rather than URIs.
    """
    vals: List[str] = []
    for n in nodes:
        v = n.get("altLabel") or n.get("skos:altLabel")
        if v is None:
            continue

        def add_item(item: Any):
            if isinstance(item, str):
                vals.append(item)
            elif isinstance(item, dict):
                if item.get("@language") == "sv" and isinstance(item.get("@value"), str):
                    vals.append(item["@value"])
                elif "sv" in item and isinstance(item["sv"], str):
                    vals.append(item["sv"])

        if isinstance(v, list):
            for item in v:
                add_item(item)
        else:
            add_item(v)

    # de-dup
    seen = set()
    out = []
    for x in vals:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


@st.cache_data(ttl=3600)
def resolve_labels_for_uris(uris: List[str], limit: int = RELATED_LABEL_FETCH_LIMIT) -> List[str]:
    """
    Dereference a small number of URIs to get Swedish labels.
    (We limit per category for performance; this does not limit SAO coverage, only how many
     related labels we resolve for display.)
    """
    out: List[str] = []
    for u in uris[:limit]:
        try:
            doc = idkb_fetch_concept(u)
            nodes = _flatten_graph(doc)
            lbl = None
            for n in nodes:
                lbl = _extract_sv_label(n)
                if lbl:
                    break
            out.append(lbl or u)
        except Exception:
            out.append(u)
    return out


def compute_sao_expansion_for_token(token: str, include_hierarchy: bool) -> dict:
    """
    Main expansion pipeline:
      - find candidates via id.kb.se/find
      - choose best (exact label else first)
      - fetch concept JSON-LD and extract altLabel/broader/narrower
    """
    payload = {
        "token": token,
        "source": "id.kb.se/find",
        "candidates": [],
        "chosen": None,  # {"label","uri"}
        "altLabel": [],
        "broader_uris": [],
        "narrower_uris": [],
        "broader_labels": [],
        "narrower_labels": [],
        "expansion_tokens": [],
        "error": None,
    }

    cands = sao_find_candidates(token, limit=FIND_LIMIT)
    payload["candidates"] = cands

    if not cands:
        payload["source"] = "id.kb.se/find (no hits)"
        return payload

    # choose best candidate
    token_low = token.lower()
    chosen = None
    for c in cands:
        if c["label"].lower() == token_low:
            chosen = c
            break
    chosen = chosen or cands[0]
    payload["chosen"] = chosen

    # fetch concept data
    doc = idkb_fetch_concept(chosen["uri"])
    nodes = _flatten_graph(doc)

    payload["altLabel"] = _collect_alt_labels(nodes)

    broader_uris = _collect_related_uris(nodes, "broader")[:RELATED_URI_LIMIT] if include_hierarchy else []
    narrower_uris = _collect_related_uris(nodes, "narrower")[:RELATED_URI_LIMIT] if include_hierarchy else []

    payload["broader_uris"] = broader_uris
    payload["narrower_uris"] = narrower_uris

    # resolve a limited number of related labels for display
    if include_hierarchy:
        payload["broader_labels"] = resolve_labels_for_uris(broader_uris, limit=RELATED_LABEL_FETCH_LIMIT)
        payload["narrower_labels"] = resolve_labels_for_uris(narrower_uris, limit=RELATED_LABEL_FETCH_LIMIT)

    # Build expansion tokens for retrieval (tokenize labels/altLabels/related labels)
    phrases = []
    phrases.extend(payload["altLabel"])
    if include_hierarchy:
        phrases.extend(payload["broader_labels"])
        phrases.extend(payload["narrower_labels"])

    seen = set([token_low])
    expanded = []
    for ph in phrases:
        for t in tokenize(ph):
            if t not in seen:
                seen.add(t)
                expanded.append(t)
    payload["expansion_tokens"] = expanded
    return payload


# -----------------------------
# Search orchestration with session cache and explicit button
# -----------------------------
def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}  # token -> payload
    if "run_sao" not in st.session_state:
        st.session_state["run_sao"] = False


def search_with_expansion(inv: Dict[str, set], query: str, expand_enabled: bool, include_hierarchy: bool, run_now: bool):
    tokens = tokenize(query)
    if not tokens:
        return set(), [], [], []

    ensure_state()

    groups: List[List[str]] = []
    errors: List[str] = []
    debug: List[dict] = []

    for tok in tokens:
        group = [tok]
        dbg = {"token": tok, "source": "none", "chosen": None, "altLabel": [], "broader_labels": [], "narrower_labels": []}

        if expand_enabled:
            # compute only when explicitly triggered OR use cached result
            if run_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_sao_expansion_for_token(
                            token=tok, include_hierarchy=include_hierarchy
                        )
                    dbg = st.session_state["sao_cache"][tok]
                    group += dbg.get("expansion_tokens", [])
                except Exception as e:
                    errors.append(f"SAO expansion failed for '{tok}': {e}")

        # de-dup
        deduped = []
        seen = set()
        for t in group:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        groups.append(deduped)
        debug.append(dbg)

    # OR within each group, AND across groups
    sets = []
    for g in groups:
        s = set()
        for t in g:
            s |= inv.get(t, set())
        sets.append(s)
    ids = set.intersection(*sets) if sets else set()
    return ids, groups, errors, debug


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Indexing Lab", layout="wide")
st.title("Indexing Lab")

# define variables top-level to avoid NameError on reruns
sheet_url = DEFAULT_SHEET_URL
sheet_name = ""

with st.sidebar:
    sheet_url = st.text_input("Google Sheet URL", value=sheet_url)
    sheet_name = st.text_input("Worksheet name (optional)", value=sheet_name)
    if st.button("Refresh data"):
        st.cache_data.clear()

df = load_sheet_as_df(sheet_url, sheet_name)
if df.empty:
    st.warning("The sheet loaded, but it appears to be empty.")
    st.stop()

id_col = st.selectbox("ID column", options=list(df.columns))

available_baseline = [f for f in BASELINE_FIELDS if f in df.columns]
available_index = [f for f in INDEX_FIELDS if f in df.columns]

query = st.text_input("Query")

expand_query = st.checkbox("Enable SAO expansion (via id.kb.se/find)", value=False)
include_hierarchy = st.checkbox("Include broader/narrower terms (dereference URIs)", value=True)

run_now = False
if expand_query:
    run_now = st.button("Run SAO expansion now")

mode = st.radio("Search mode", ["Baseline", "Enriched"], horizontal=True)

if mode == "Baseline":
    default_fields = available_baseline if available_baseline else list(df.columns)[:3]
    fields = st.multiselect("Fields searched", options=list(df.columns), default=default_fields)
else:
    default_fields = list(dict.fromkeys(available_baseline + available_index))
    if not default_fields:
        default_fields = list(df.columns)[:6]
    fields = st.multiselect("Fields searched", options=list(df.columns), default=default_fields)

inv = build_inverted_index(df, fields, id_col)

ids, groups, errors, debug = search_with_expansion(inv, query, expand_query, include_hierarchy, run_now)

if query.strip():
    if expand_query:
        with st.expander("Query expansion (debug)", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            st.markdown("#### SAO lookup and hierarchy (what students should learn)")
            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "—"))

                chosen = item.get("chosen")
                if chosen:
                    st.write("Chosen concept:", chosen.get("label", "—"))
                    st.code(chosen.get("uri", "—"))
                else:
                    st.write("Chosen concept: —")

                # show candidates (disambiguation)
                cands = item.get("candidates", [])
                if cands:
                    st.write("Top candidates (disambiguation):")
                    for c in cands[:8]:
                        st.write(f"- {c['label']}")
                else:
                    st.write("Top candidates: —")

                st.write("altLabel:", ", ".join(item.get("altLabel", [])) if item.get("altLabel") else "—")
                st.write("broader:", ", ".join(item.get("broader_labels", [])) if item.get("broader_labels") else "—")
                st.write("narrower:", ", ".join(item.get("narrower_labels", [])) if item.get("narrower_labels") else "—")
                st.divider()

            if errors:
                st.warning("\n".join(errors))
            else:
                st.caption("Tip: On Community Cloud, click 'Run SAO expansion now' after you finish typing.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
