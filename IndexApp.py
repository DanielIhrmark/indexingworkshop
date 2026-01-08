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

# id.kb.se
IDKB_FIND_ENDPOINT = "https://id.kb.se/find"
SAO_SCHEME_URI = "https://id.kb.se/term/sao"  # IMPORTANT: no trailing slash :contentReference[oaicite:1]{index=1}

BASELINE_FIELDS = ["title", "author", "abstract"]
INDEX_FIELDS = ["keywords_free", "subjects_controlled", "ddc", "sab", "entities"]

# Streamlit Community Cloud tuning
HTTP_TIMEOUT_SECONDS = 15
HTTP_RETRIES = 2
HTTP_BACKOFF_SECONDS = 1

FIND_LIMIT = 15
RELATED_URI_LIMIT = 10
RELATED_LABEL_RESOLVE_LIMIT = 10


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
# HTTP + JSON-LD extraction from HTML
# -----------------------------
JSONLD_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def http_get_text(url: str, params: Optional[dict] = None) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(0, 1 + HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF_SECONDS * (2 ** attempt))
            else:
                break
    raise last_err  # type: ignore


def extract_jsonld_objects_from_html(html: str) -> List[Any]:
    """
    id.kb.se pages commonly embed JSON-LD in HTML.
    We extract all <script type="application/ld+json"> blocks and parse them.
    """
    objs: List[Any] = []
    for m in JSONLD_SCRIPT_RE.finditer(html or ""):
        raw = m.group(1).strip()
        if not raw:
            continue
        try:
            objs.append(json.loads(raw))
        except Exception:
            # Some pages may contain multiple JSON objects separated oddly; ignore failures
            continue
    return objs


def flatten_jsonld_graph(doc: Any) -> List[dict]:
    """
    Turn a JSON-LD document (or list of docs) into a list of node dicts.
    """
    if isinstance(doc, list):
        out: List[dict] = []
        for d in doc:
            out.extend(flatten_jsonld_graph(d))
        return out

    if not isinstance(doc, dict):
        return []

    if "@graph" in doc and isinstance(doc["@graph"], list):
        return [n for n in doc["@graph"] if isinstance(n, dict)]

    # sometimes the doc itself is the node
    return [doc]


def fetch_jsonld_from_page(url: str, params: Optional[dict] = None) -> List[dict]:
    """
    Fetch HTML and extract JSON-LD nodes.
    """
    html = http_get_text(url, params=params)
    jsonld_docs = extract_jsonld_objects_from_html(html)
    nodes: List[dict] = []
    for d in jsonld_docs:
        nodes.extend(flatten_jsonld_graph(d))
    return nodes


def extract_uri(node: Any) -> Optional[str]:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return node.get("@id") or node.get("id")
    return None


def extract_sv_label(node: dict) -> Optional[str]:
    """
    Try common label patterns.
    """
    for key in ("prefLabel", "label", "name"):
        v = node.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and isinstance(v.get("sv"), str):
            return v["sv"]
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and item.get("@language") == "sv" and isinstance(item.get("@value"), str):
                    return item["@value"]
    return None


def collect_literal_sv_list(v: Any) -> List[str]:
    out: List[str] = []

    def add(item: Any):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            if item.get("@language") == "sv" and isinstance(item.get("@value"), str):
                out.append(item["@value"])
            elif isinstance(item.get("sv"), str):
                out.append(item["sv"])

    if isinstance(v, list):
        for item in v:
            add(item)
    else:
        add(v)

    # de-dup case-insensitive
    seen = set()
    ded = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            ded.append(s)
    return ded


def collect_related_uris(nodes: List[dict], predicate: str) -> List[str]:
    out: List[str] = []
    for n in nodes:
        v = n.get(predicate)
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                u = extract_uri(item)
                if u:
                    out.append(u)
        else:
            u = extract_uri(v)
            if u:
                out.append(u)

    seen = set()
    ded = []
    for u in out:
        if u not in seen:
            seen.add(u)
            ded.append(u)
    return ded


# -----------------------------
# SAO lookup via id.kb.se/find (HTML + JSON-LD extraction)
# -----------------------------
@st.cache_data(ttl=3600)
def sao_find_candidates(q: str, limit: int = FIND_LIMIT) -> List[dict]:
    q = (q or "").strip()
    if len(q) < 2:
        return []

    params = {
        "q": q,
        "_limit": str(limit),
        # This filter works in the UI and yields SAO hits for klimat :contentReference[oaicite:2]{index=2}
        "and-inScheme.@id": SAO_SCHEME_URI,
    }

    nodes = fetch_jsonld_from_page(IDKB_FIND_ENDPOINT, params=params)

    # Extract candidate nodes with both uri + label
    cands: List[dict] = []
    for n in nodes:
        uri = extract_uri(n)
        lbl = extract_sv_label(n)
        if uri and lbl:
            cands.append({"label": lbl, "uri": uri})

    # de-dup by URI
    seen = set()
    out = []
    for c in cands:
        if c["uri"] not in seen:
            seen.add(c["uri"])
            out.append(c)

    # prefer exact label match, then shorter label
    q_low = q.lower()
    out.sort(key=lambda x: (0 if x["label"].lower() == q_low else 1, len(x["label"])))
    return out[:limit]


@st.cache_data(ttl=3600)
def fetch_concept_nodes(uri: str) -> List[dict]:
    """
    Concept pages also embed JSON-LD; extract nodes from HTML.
    """
    return fetch_jsonld_from_page(uri, params=None)


@st.cache_data(ttl=3600)
def resolve_labels_for_uris(uris: List[str], limit: int = RELATED_LABEL_RESOLVE_LIMIT) -> List[str]:
    labels: List[str] = []
    for u in uris[:limit]:
        try:
            nodes = fetch_concept_nodes(u)
            lbl = None
            for n in nodes:
                lbl = extract_sv_label(n)
                if lbl:
                    break
            labels.append(lbl or u)
        except Exception:
            labels.append(u)
    return labels


def compute_sao_expansion_for_token(token: str, include_hierarchy: bool) -> dict:
    payload = {
        "token": token,
        "source": "id.kb.se/find (HTML+JSON-LD)",
        "candidates": [],
        "chosen": None,
        "altLabel": [],
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

    token_low = token.lower()
    chosen = next((c for c in cands if c["label"].lower() == token_low), cands[0])
    payload["chosen"] = chosen

    # Fetch chosen concept JSON-LD
    nodes = fetch_concept_nodes(chosen["uri"])

    # altLabel: try skos:altLabel + altLabel
    alt = []
    for n in nodes:
        if "altLabel" in n:
            alt.extend(collect_literal_sv_list(n["altLabel"]))
        if "skos:altLabel" in n:
            alt.extend(collect_literal_sv_list(n["skos:altLabel"]))
    # de-dup
    seen = set()
    alt_ded = []
    for a in alt:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            alt_ded.append(a)
    payload["altLabel"] = alt_ded

    if include_hierarchy:
        broader_uris = collect_related_uris(nodes, "broader")[:RELATED_URI_LIMIT]
        narrower_uris = collect_related_uris(nodes, "narrower")[:RELATED_URI_LIMIT]

        payload["broader_labels"] = resolve_labels_for_uris(broader_uris, limit=RELATED_LABEL_RESOLVE_LIMIT)
        payload["narrower_labels"] = resolve_labels_for_uris(narrower_uris, limit=RELATED_LABEL_RESOLVE_LIMIT)

    # Build expansion tokens (for retrieval), based on alt/broader/narrower labels
    phrases = []
    phrases.extend(payload["altLabel"])
    if include_hierarchy:
        phrases.extend(payload["broader_labels"])
        phrases.extend(payload["narrower_labels"])

    seen_tok = set([token_low])
    expanded = []
    for ph in phrases:
        for t in tokenize(ph):
            if t not in seen_tok:
                seen_tok.add(t)
                expanded.append(t)

    payload["expansion_tokens"] = expanded
    return payload


# -----------------------------
# Search orchestration (explicit run + session cache)
# -----------------------------
def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}


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
        dbg = {"token": tok, "source": "none", "chosen": None, "candidates": [], "altLabel": [], "broader_labels": [], "narrower_labels": []}

        if expand_enabled:
            if run_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_sao_expansion_for_token(tok, include_hierarchy)
                    dbg = st.session_state["sao_cache"][tok]
                    group += dbg.get("expansion_tokens", [])
                except Exception as e:
                    errors.append(f"SAO expansion failed for '{tok}': {e}")

        # de-dup group
        deduped = []
        seen = set()
        for t in group:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        groups.append(deduped)
        debug.append(dbg)

    # OR within group, AND across groups
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

# Define variables top-level to avoid rerun NameErrors
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

query = st.text_input("Query")

expand_query = st.checkbox("Enable SAO expansion (id.kb.se/find)", value=False)
include_hierarchy = st.checkbox("Include broader/narrower terms", value=True)

run_now = False
if expand_query:
    run_now = st.button("Run SAO expansion now")

mode = st.radio("Search mode", ["Baseline", "Enriched"], horizontal=True)

available_baseline = [f for f in BASELINE_FIELDS if f in df.columns]
available_index = [f for f in INDEX_FIELDS if f in df.columns]

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

            st.markdown("#### SAO lookup and hierarchy")
            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "—"))

                chosen = item.get("chosen")
                if chosen:
                    st.write("Chosen concept:", chosen.get("label", "—"))
                    st.code(chosen.get("uri", "—"))
                else:
                    st.write("Chosen concept: —")

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
                st.caption("Tip: On Streamlit Community Cloud, click 'Run SAO expansion now' after typing your query.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
