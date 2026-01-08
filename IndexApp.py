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

BASELINE_FIELDS = ["title", "author", "abstract"]
INDEX_FIELDS = ["keywords_free", "subjects_controlled", "ddc", "sab", "entities"]

SAO_SCHEME_URI = "https://id.kb.se/term/sao"

# Community Cloud tuning
SPARQL_TIMEOUT_SECONDS = 60
SPARQL_RETRIES = 2
SPARQL_BACKOFF_SECONDS = 1

LABEL_CANDIDATE_LIMIT = 15
SAO_FILTER_LIMIT = 15

# Offline fallback (extend as needed for workshop stability)
FALLBACK_EXPANSIONS = {
    "klimat": ["klimatförändring", "global uppvärmning", "växthuseffekt", "miljö"],
    "politik": ["politiska partier", "ideologi", "demokrati", "förvaltning"],
    "energi": ["energipolitik", "energiförsörjning", "förnybar energi"],
    "migration": ["invandring", "asylpolitik", "integration"],
}


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
# SAO: minimal SPARQL (kept small)
# -----------------------------
@st.cache_data(ttl=3600)
def label_search_candidates(endpoint: str, token: str, mode: str, limit: int = LABEL_CANDIDATE_LIMIT) -> List[Tuple[str, str]]:
    token = (token or "").strip()
    if len(token) < 2:
        return []

    safe = token.replace('"', '\\"')

    if mode == "exact":
        label_filter = f'FILTER(LCASE(STR(?label)) = LCASE("{safe}"))'
    elif mode == "starts":
        label_filter = f'FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{safe}")))'
    else:
        label_filter = f'FILTER regex(str(?label), "{re.escape(token)}", "i")'

    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?s ?label WHERE {{
      {{
        ?s kbv:prefLabel ?label .
        FILTER(lang(?label) = "sv")
        {label_filter}
      }}
      UNION
      {{
        ?s skos:prefLabel ?label .
        FILTER(lang(?label) = "sv")
        {label_filter}
      }}
      UNION
      {{
        ?s rdfs:label ?label .
        FILTER(lang(?label) = "sv")
        {label_filter}
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    return [(b["s"]["value"], b["label"]["value"]) for b in data.get("results", {}).get("bindings", [])]


@st.cache_data(ttl=3600)
def filter_to_sao(endpoint: str, uris: List[str], limit: int = SAO_FILTER_LIMIT) -> List[str]:
    if not uris:
        return []
    uris = uris[:50]
    values = " ".join(f"<{u}>" for u in uris)

    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?s WHERE {{
      VALUES ?s {{ {values} }}
      {{
        ?s kbv:inScheme <{SAO_SCHEME_URI}> .
      }}
      UNION
      {{
        ?s skos:inScheme <{SAO_SCHEME_URI}> .
      }}
      UNION
      {{
        ?s kbv:inVocabulary <{SAO_SCHEME_URI}> .
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    return [b["s"]["value"] for b in data.get("results", {}).get("bindings", [])]


@st.cache_data(ttl=3600)
def sao_context_raw(endpoint: str, term_uri: str) -> Dict[str, List[str]]:
    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?p ?oLabel WHERE {{
      VALUES ?term {{ <{term_uri}> }}

      {{
        ?term kbv:altLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("altLabel" AS ?p)
      }}
      UNION
      {{
        ?term skos:altLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("altLabel" AS ?p)
      }}
      UNION
      {{
        ?term kbv:broader ?o .
        ?o kbv:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("broader" AS ?p)
      }}
      UNION
      {{
        ?term skos:broader ?o .
        ?o skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("broader" AS ?p)
      }}
      UNION
      {{
        ?n kbv:broader ?term .
        ?n kbv:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("narrower" AS ?p)
      }}
      UNION
      {{
        ?n skos:broader ?term .
        ?n skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("narrower" AS ?p)
      }}
    }}
    """
    data = sparql_select_json(endpoint, sparql)
    ctx = {"broader": [], "narrower": [], "altLabel": []}
    for b in data.get("results", {}).get("bindings", []):
        p = b["p"]["value"]
        lbl = b["oLabel"]["value"]
        if p in ctx:
            ctx[p].append(lbl)
    for k in ctx:
        ctx[k] = sorted(list(dict.fromkeys(ctx[k])), key=str.lower)
    return ctx


def best_sao_uri_for_token(endpoint: str, token: str) -> Tuple[Optional[str], List[Tuple[str, str]], List[str]]:
    cands = label_search_candidates(endpoint, token, mode="exact")
    if not cands:
        cands = label_search_candidates(endpoint, token, mode="starts")
    if not cands:
        cands = label_search_candidates(endpoint, token, mode="contains")

    sao_uris = filter_to_sao(endpoint, [u for (u, _) in cands])
    if sao_uris:
        sao_set = set(sao_uris)
        sao_cands = [(u, lbl) for (u, lbl) in cands if u in sao_set]
        if sao_cands:
            best = sorted(sao_cands, key=lambda x: len(x[1]))[0]
            return best[0], cands, sao_uris

    return None, cands, sao_uris


# -----------------------------
# Expansion: explicit run + session cache + fallback
# -----------------------------
def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}  # token -> payload


def compute_expansion_for_token(endpoint: str, token: str, include_hierarchy: bool) -> Dict:
    payload = {
        "token": token,
        "source": "none",
        "best_sao_uri": None,
        "candidate_count": 0,
        "candidate_preview": [],
        "sao_uri_count": 0,
        "sao_uri_preview": [],
        "raw_altLabel": [],
        "raw_broader": [],
        "raw_narrower": [],
        "expansion_tokens": [],
        "error": None,
    }

    # Try live SAO
    try:
        best_uri, candidates, sao_uris = best_sao_uri_for_token(endpoint, token)
        payload["best_sao_uri"] = best_uri
        payload["candidate_count"] = len(candidates)
        payload["candidate_preview"] = [(lbl, uri) for (uri, lbl) in candidates[:5]]
        payload["sao_uri_count"] = len(sao_uris)
        payload["sao_uri_preview"] = sao_uris[:5]

        if best_uri:
            ctx = sao_context_raw(endpoint, best_uri)
            payload["raw_altLabel"] = ctx.get("altLabel", [])
            payload["raw_broader"] = ctx.get("broader", []) if include_hierarchy else []
            payload["raw_narrower"] = ctx.get("narrower", []) if include_hierarchy else []

            phrases = list(payload["raw_altLabel"])
            if include_hierarchy:
                phrases += payload["raw_broader"] + payload["raw_narrower"]

            toks = []
            seen = set([token])
            for ph in phrases:
                for t in tokenize(ph):
                    if t not in seen:
                        seen.add(t)
                        toks.append(t)

            payload["expansion_tokens"] = toks
            payload["source"] = "SAO (live)"
            return payload

        # If we got here, SAO did not resolve to a term; fall back below

    except Exception as e:
        payload["error"] = str(e)

    # Fallback expansions (offline)
    if token in FALLBACK_EXPANSIONS:
        payload["source"] = "Fallback (offline)"
        payload["raw_altLabel"] = FALLBACK_EXPANSIONS[token]
        # Use phrases as tokens too
        toks = []
        seen = set([token])
        for ph in FALLBACK_EXPANSIONS[token]:
            for t in tokenize(ph):
                if t not in seen:
                    seen.add(t)
                    toks.append(t)
        payload["expansion_tokens"] = toks

    else:
        payload["source"] = "None"

    return payload


def search_with_expansion(inv, query, endpoint, expand_enabled, include_hierarchy, run_expansion_now):
    base_tokens = tokenize(query)
    if not base_tokens:
        return set(), [], [], []

    ensure_state()

    groups = []
    errors = []
    debug = []

    for tok in base_tokens:
        g = [tok]
        dbg = {"token": tok, "source": "None", "expansion_tokens": []}

        if expand_enabled:
            # Only compute when button pressed; otherwise use cache if present
            if run_expansion_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_expansion_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_expansion_for_token(
                            endpoint=endpoint,
                            token=tok,
                            include_hierarchy=include_hierarchy,
                        )
                    dbg = st.session_state["sao_cache"][tok]
                    g += dbg.get("expansion_tokens", [])
                    if dbg.get("error"):
                        errors.append(f"{tok}: {dbg['error']}")
                except Exception as e:
                    errors.append(f"{tok}: {e}")

        # de-dup group
        deduped = []
        seen = set()
        for t in g:
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

# define variables at top level (avoid NameError on reruns)
sheet_url = DEFAULT_SHEET_URL
sheet_name = ""
sparql_endpoint = DEFAULT_SPARQL_ENDPOINT

with st.sidebar:
    sheet_url = st.text_input("Google Sheet URL", value=sheet_url)
    sheet_name = st.text_input("Worksheet name (optional)", value=sheet_name)
    sparql_endpoint = st.text_input("SPARQL endpoint", value=sparql_endpoint)

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

expand_query = st.checkbox("Enable SAO expansion", value=False)
include_hierarchy = st.checkbox("Include broader/narrower terms", value=True)

run_expansion_now = False
if expand_query:
    run_expansion_now = st.button("Run SAO expansion now")

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

ids, groups, errors, debug = search_with_expansion(inv, query, sparql_endpoint, expand_query, include_hierarchy, run_expansion_now)

if query.strip():
    if expand_query:
        with st.expander("Query expansion (debug)", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            st.markdown("#### Expansion status per token")
            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "None"))
                if item.get("best_sao_uri"):
                    st.code(item["best_sao_uri"])
                st.write("altLabel:", ", ".join(item.get("raw_altLabel", [])) if item.get("raw_altLabel") else "—")
                st.write("broader:", ", ".join(item.get("raw_broader", [])) if item.get("raw_broader") else "—")
                st.write("narrower:", ", ".join(item.get("raw_narrower", [])) if item.get("raw_narrower") else "—")
                st.divider()

            if errors:
                st.warning(
                    "Live SAO lookups timed out or failed. The app may show fallback expansions instead.\n\n"
                    + "\n".join(errors)
                )
            else:
                st.caption("Tip: On Community Cloud, click 'Run SAO expansion now' only after you finish typing your query.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
