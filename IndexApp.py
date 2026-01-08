import re
import time
import urllib.parse
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/11J5JRtap7p2P8BWl3gsPVs1I-DATbVdOi4Oj1fcSbe0/edit?usp=sharing"
DEFAULT_SPARQL_ENDPOINT = "https://libris.kb.se/sparql"

# IMPORTANT: use trailing-slash scheme URI (matches Libris SPARQL prefix listing) :contentReference[oaicite:1]{index=1}
SAO_SCHEME_URIS = [
    "https://id.kb.se/term/sao/",   # preferred
    "https://id.kb.se/term/sao",    # tolerate non-slash if present
]

KBV = "https://id.kb.se/vocab/"

BASELINE_FIELDS = ["title", "author", "abstract"]
INDEX_FIELDS = ["keywords_free", "subjects_controlled", "ddc", "sab", "entities"]

# Community Cloud tuning
SPARQL_TIMEOUT_SECONDS = 20
SPARQL_RETRIES = 2
SPARQL_BACKOFF_SECONDS = 1

# Keep endpoint load modest
CANDIDATE_LIMIT = 15
ALTLABEL_LIMIT = 50


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
# SPARQL helper (fast-fail + retries)
# -----------------------------
def sparql_select_json(endpoint: str, sparql: str) -> dict:
    params = {
        "query": sparql,
        "format": "application/sparql-results+json",
        "output": "application/sparql-results+json",
    }
    headers = {"Accept": "application/sparql-results+json"}

    last_err: Optional[Exception] = None
    for attempt in range(0, 1 + SPARQL_RETRIES):
        try:
            r = requests.get(endpoint, params=params, headers=headers, timeout=SPARQL_TIMEOUT_SECONDS)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < SPARQL_RETRIES:
                time.sleep(SPARQL_BACKOFF_SECONDS * (2 ** attempt))
            else:
                break
    raise last_err  # type: ignore


# -----------------------------
# SAO lookup (SPARQL) + altLabels only
# -----------------------------
@st.cache_data(ttl=3600)
def sao_candidates(endpoint: str, token: str, limit: int = CANDIDATE_LIMIT) -> List[Tuple[str, str]]:
    """
    Return SAO candidate concepts as (uri, prefLabel).
    Fixes:
      - accept BOTH SAO scheme URIs (with/without trailing slash)
      - accept kbv:inScheme OR skos:inScheme
      - accept kbv:prefLabel OR skos:prefLabel OR rdfs:label
    """
    token = (token or "").strip()
    if len(token) < 2:
        return []

    safe = token.replace('"', '\\"')
    scheme_values = " ".join(f"<{u}>" for u in SAO_SCHEME_URIS)

    sparql = f"""
    PREFIX kbv: <{KBV}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?term ?label WHERE {{
      VALUES ?scheme {{ {scheme_values} }}

      ?term (kbv:inScheme|skos:inScheme) ?scheme .
      ?term (kbv:prefLabel|skos:prefLabel|rdfs:label) ?label .
      FILTER(lang(?label) = "sv")

      # Starts-with first (fast), but keep a regex fallback for robustness
      FILTER(
        STRSTARTS(LCASE(STR(?label)), LCASE("{safe}"))
        || regex(str(?label), "{re.escape(token)}", "i")
      )
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    out: List[Tuple[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        out.append((b["term"]["value"], b["label"]["value"]))

    # de-dup by URI
    seen = set()
    ded = []
    for uri, lbl in out:
        if uri not in seen:
            seen.add(uri)
            ded.append((uri, lbl))

    # prefer exact label match, then shortest
    tok_low = token.lower()
    ded.sort(key=lambda x: (0 if x[1].lower() == tok_low else 1, len(x[1])))
    return ded


@st.cache_data(ttl=3600)
def sao_altlabels(endpoint: str, term_uri: str, limit: int = ALTLABEL_LIMIT) -> List[str]:
    """
    Fetch Swedish altLabels for a specific SAO concept URI.
    """
    sparql = f"""
    PREFIX kbv: <{KBV}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?alt WHERE {{
      VALUES ?term {{ <{term_uri}> }}
      {{
        ?term kbv:altLabel ?alt .
        FILTER(lang(?alt) = "sv")
      }}
      UNION
      {{
        ?term skos:altLabel ?alt .
        FILTER(lang(?alt) = "sv")
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    alts = [b["alt"]["value"] for b in data.get("results", {}).get("bindings", [])]

    # de-dup case-insensitive
    seen = set()
    out = []
    for a in alts:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}
    if "enable_sao" not in st.session_state:
        st.session_state["enable_sao"] = False
    if "run_sao_now" not in st.session_state:
        st.session_state["run_sao_now"] = False


def compute_altlabel_expansion_for_token(endpoint: str, token: str) -> dict:
    payload = {
        "token": token,
        "source": "SPARQL (libris.kb.se)",
        "candidates": [],
        "chosen": None,
        "altLabel": [],
        "expansion_tokens": [],
        "error": None,
    }

    cands = sao_candidates(endpoint, token, limit=CANDIDATE_LIMIT)
    payload["candidates"] = [{"uri": u, "label": l} for (u, l) in cands]

    if not cands:
        payload["source"] = "SPARQL (no SAO candidates)"
        return payload

    chosen_uri, chosen_lbl = cands[0]
    payload["chosen"] = {"uri": chosen_uri, "label": chosen_lbl}

    alts = sao_altlabels(endpoint, chosen_uri, limit=ALTLABEL_LIMIT)
    payload["altLabel"] = alts

    # expansion tokens derived from altLabels
    seen = set([token.lower()])
    exp = []
    for phrase in alts:
        for t in tokenize(phrase):
            if t not in seen:
                seen.add(t)
                exp.append(t)
    payload["expansion_tokens"] = exp
    return payload


def search_with_expansion(inv: Dict[str, set], query: str, endpoint: str, expand_enabled: bool, run_now: bool):
    tokens = tokenize(query)
    if not tokens:
        return set(), [], [], []

    ensure_state()

    groups: List[List[str]] = []
    errors: List[str] = []
    debug: List[dict] = []

    for tok in tokens:
        group = [tok]
        dbg = {
            "token": tok,
            "source": "Not run (click 'Run SAO altLabel expansion')",
            "candidates": [],
            "chosen": None,
            "altLabel": [],
            "expansion_tokens": [],
        }

        if expand_enabled:
            if run_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_altlabel_expansion_for_token(endpoint, tok)
                    dbg = st.session_state["sao_cache"][tok]
                    group += dbg.get("expansion_tokens", [])
                except Exception as e:
                    errors.append(f"SAO altLabel expansion failed for '{tok}': {e}")

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
    group_sets = []
    for g in groups:
        s = set()
        for t in g:
            s |= inv.get(t, set())
        group_sets.append(s)

    ids = set.intersection(*group_sets) if group_sets else set()
    return ids, groups, errors, debug


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Indexing Lab", layout="wide")
st.title("Indexing Lab")

ensure_state()

sheet_url = DEFAULT_SHEET_URL
sheet_name = ""
sparql_endpoint = DEFAULT_SPARQL_ENDPOINT

with st.sidebar:
    sheet_url = st.text_input("Google Sheet URL", value=sheet_url)
    sheet_name = st.text_input("Worksheet name (optional)", value=sheet_name)
    sparql_endpoint = st.text_input("SPARQL endpoint", value=sparql_endpoint)

    if st.button("Refresh data"):
        st.cache_data.clear()

    if st.button("Test SPARQL endpoint"):
        try:
            test = sparql_select_json(sparql_endpoint, "SELECT (1 as ?ok) WHERE {} LIMIT 1")
            st.success(f"Endpoint OK: {test['results']['bindings']}")
        except Exception as e:
            st.error(f"Endpoint test failed: {e}")

df = load_sheet_as_df(sheet_url, sheet_name)
if df.empty:
    st.warning("The sheet loaded, but it appears to be empty.")
    st.stop()

id_col = st.selectbox("ID column", options=list(df.columns))
query = st.text_input("Query")

expand_query = st.checkbox(
    "Enable SAO altLabel expansion (SPARQL)",
    value=st.session_state["enable_sao"],
    key="enable_sao",
)

if expand_query:
    if st.button("Run SAO altLabel expansion", key="run_sao_btn"):
        st.session_state["run_sao_now"] = True
else:
    st.session_state["run_sao_now"] = False

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

ids, groups, errors, debug = search_with_expansion(
    inv=inv,
    query=query,
    endpoint=sparql_endpoint,
    expand_enabled=expand_query,
    run_now=st.session_state["run_sao_now"],
)

# reset one-shot flag after use
st.session_state["run_sao_now"] = False

if query.strip():
    if expand_query:
        with st.expander("SAO matches and altLabels", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "—"))

                chosen = item.get("chosen")
                if chosen:
                    st.write("Matched SAO prefLabel:", chosen.get("label", "—"))
                    st.code(chosen.get("uri", "—"))
                else:
                    st.write("Matched SAO prefLabel: —")
                    st.write("Matched SAO URI: —")

                cands = item.get("candidates", [])
                if cands:
                    st.write("Top candidates:")
                    for c in cands[:8]:
                        st.write(f"- {c['label']}")
                else:
                    st.write("Top candidates: —")

                alts = item.get("altLabel", [])
                st.write("altLabel:", ", ".join(alts) if alts else "—")
                st.divider()

            if errors:
                st.warning("\n".join(errors))
            else:
                st.caption("Tip: Click 'Run SAO altLabel expansion' after typing. Results are cached per session.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
