import re
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
KBV_PREFIX = "https://id.kb.se/vocab/"


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
# SPARQL helper (robust JSON)
# -----------------------------
def sparql_select_json(endpoint: str, sparql: str) -> dict:
    params = {
        "query": sparql,
        "format": "application/sparql-results+json",
        "output": "application/sparql-results+json",
    }
    headers = {"Accept": "application/sparql-results+json"}
    r = requests.get(endpoint, params=params, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


# -----------------------------
# SAO matching + expansion (KBV-first, SKOS fallback)
# -----------------------------
@st.cache_data(ttl=3600)
def sao_lookup_best_term(endpoint: str, token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (term_uri, label) for a 'best' SAO match for token.
    Uses KBV :inScheme/:prefLabel (as per KB examples), with SKOS fallback.
    Strategy: exact -> starts-with -> contains; pick shortest label as heuristic.
    """
    token = (token or "").strip()
    if len(token) < 2:
        return None, None

    safe = token.replace('"', '\\"')

    # 1) Exact match
    sparql_exact = f"""
    PREFIX : <{KBV_PREFIX}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?term ?label WHERE {{
      {{
        ?term :inScheme <{SAO_SCHEME_URI}> ;
              :prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER(LCASE(STR(?label)) = LCASE("{safe}"))
      }}
      UNION
      {{
        ?term skos:inScheme <{SAO_SCHEME_URI}> ;
              skos:prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER(LCASE(STR(?label)) = LCASE("{safe}"))
      }}
    }}
    LIMIT 20
    """
    data = sparql_select_json(endpoint, sparql_exact)
    bindings = data.get("results", {}).get("bindings", [])
    if bindings:
        b = bindings[0]
        return b["term"]["value"], b["label"]["value"]

    # 2) Starts-with match
    sparql_starts = f"""
    PREFIX : <{KBV_PREFIX}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?term ?label WHERE {{
      {{
        ?term :inScheme <{SAO_SCHEME_URI}> ;
              :prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{safe}")))
      }}
      UNION
      {{
        ?term skos:inScheme <{SAO_SCHEME_URI}> ;
              skos:prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{safe}")))
      }}
    }}
    LIMIT 100
    """
    data = sparql_select_json(endpoint, sparql_starts)
    bindings = data.get("results", {}).get("bindings", [])
    if bindings:
        best = sorted(bindings, key=lambda x: len(x["label"]["value"]))[0]
        return best["term"]["value"], best["label"]["value"]

    # 3) Contains match
    sparql_contains = f"""
    PREFIX : <{KBV_PREFIX}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?term ?label WHERE {{
      {{
        ?term :inScheme <{SAO_SCHEME_URI}> ;
              :prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER regex(str(?label), "{re.escape(token)}", "i")
      }}
      UNION
      {{
        ?term skos:inScheme <{SAO_SCHEME_URI}> ;
              skos:prefLabel ?label .
        FILTER(lang(?label) = "sv")
        FILTER regex(str(?label), "{re.escape(token)}", "i")
      }}
    }}
    LIMIT 200
    """
    data = sparql_select_json(endpoint, sparql_contains)
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None, None

    best = sorted(bindings, key=lambda x: len(x["label"]["value"]))[0]
    return best["term"]["value"], best["label"]["value"]


@st.cache_data(ttl=3600)
def sao_term_context(endpoint: str, term_uri: str) -> Dict[str, List[str]]:
    """
    Fetch raw related labels as phrases.
    KBV uses :broader/:prefLabel/:altLabel in examples; include SKOS fallback too.
    Narrower is derived via inverse broader (something broader term_uri).
    """
    sparql = f"""
    PREFIX : <{KBV_PREFIX}>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?p ?oLabel WHERE {{
      VALUES ?term {{ <{term_uri}> }}

      # broader
      {{
        ?term :broader ?o .
        ?o :prefLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("broader" AS ?p)
      }}
      UNION
      {{
        ?term skos:broader ?o .
        ?o skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("broader" AS ?p)
      }}

      # narrower (inverse broader)
      UNION
      {{
        ?n :broader ?term .
        ?n :prefLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("narrower" AS ?p)
      }}
      UNION
      {{
        ?n skos:broader ?term .
        ?n skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("narrower" AS ?p)
      }}

      # altLabel / non-preferred labels
      UNION
      {{
        ?term :altLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("altLabel" AS ?p)
      }}
      UNION
      {{
        ?term skos:altLabel ?oLabel .
        FILTER(lang(?oLabel) = "sv")
        BIND("altLabel" AS ?p)
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

    # de-dup, stable-ish
    for k in ctx:
        ctx[k] = sorted(list(dict.fromkeys(ctx[k])), key=str.lower)

    return ctx


def expand_token_tokens(
    endpoint: str,
    token: str,
    include_hierarchy: bool,
) -> Tuple[List[str], Dict[str, List[str]], Optional[str], Optional[str]]:
    """
    Returns:
      - expansion tokens (single words) used for local retrieval
      - raw context labels (phrases) for debugging/inspection
      - matched SAO URI
      - matched SAO prefLabel
    """
    term_uri, term_label = sao_lookup_best_term(endpoint, token)
    if not term_uri:
        return [], {"altLabel": [], "broader": [], "narrower": []}, None, None

    ctx = sao_term_context(endpoint, term_uri)

    raw_labels = {
        "altLabel": list(ctx.get("altLabel", [])),
        "broader": list(ctx.get("broader", [])) if include_hierarchy else [],
        "narrower": list(ctx.get("narrower", [])) if include_hierarchy else [],
    }

    # Tokenize phrases into tokens for inverted-index lookup
    labels_for_tokens = list(raw_labels["altLabel"])
    if include_hierarchy:
        labels_for_tokens += raw_labels["broader"] + raw_labels["narrower"]

    out_tokens: List[str] = []
    seen = set()
    for lbl in labels_for_tokens:
        for t in tokenize(lbl):
            if t == token:
                continue
            if t in seen:
                continue
            seen.add(t)
            out_tokens.append(t)

    return out_tokens, raw_labels, term_uri, term_label


def search_with_expansion(
    inv: Dict[str, set],
    query: str,
    endpoint: str,
    expand: bool,
    include_hierarchy: bool,
) -> Tuple[set, List[List[str]], List[str], List[Dict]]:
    base_tokens = tokenize(query)
    if not base_tokens:
        return set(), [], [], []

    groups: List[List[str]] = []
    errors: List[str] = []
    debug: List[Dict] = []

    for tok in base_tokens:
        g = [tok]
        dbg_item = {
            "token": tok,
            "matched_uri": None,
            "matched_prefLabel": None,
            "raw_altLabel": [],
            "raw_broader": [],
            "raw_narrower": [],
        }

        if expand:
            try:
                exp_tokens, raw_labels, uri, pref = expand_token_tokens(endpoint, tok, include_hierarchy)
                g += exp_tokens

                dbg_item["matched_uri"] = uri
                dbg_item["matched_prefLabel"] = pref
                dbg_item["raw_altLabel"] = raw_labels.get("altLabel", [])
                dbg_item["raw_broader"] = raw_labels.get("broader", [])
                dbg_item["raw_narrower"] = raw_labels.get("narrower", [])
            except Exception as e:
                errors.append(f"SAO expansion failed for '{tok}': {e}")

        # de-dup group
        deduped = []
        seen = set()
        for t in g:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        groups.append(deduped)
        debug.append(dbg_item)

    # OR within each group, AND across groups
    group_sets: List[set] = []
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

with st.sidebar:
    sheet_url = st.text_input("Google Sheet URL", DEFAULT_SHEET_URL)
    sheet_name = st.text_input("Worksheet name (optional)", "")
    sparql_endpoint = st.text_input("SPARQL endpoint", DEFAULT_SPARQL_ENDPOINT)
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

expand_query = st.checkbox("Expand query using Svenska Ämnesord (SAO)", value=False)
include_hierarchy = st.checkbox("Include broader/narrower terms", value=True)

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

ids, groups, errors, debug = search_with_expansion(
    inv=inv,
    query=query,
    endpoint=sparql_endpoint,
    expand=expand_query,
    include_hierarchy=include_hierarchy,
)

if query.strip():
    if expand_query:
        with st.expander("Query expansion (debug)", expanded=True):
            # Tokens used for retrieval
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            st.markdown("#### SAO matches and raw related terms (phrases)")
            for item in debug:
                st.markdown(f"**Token:** `{item['token']}`")

                if item["matched_uri"]:
                    st.write("Matched SAO prefLabel:", item["matched_prefLabel"])
                    st.code(item["matched_uri"])
                else:
                    st.write("Matched SAO prefLabel: —")
                    st.write("Matched SAO URI: —")

                st.write("altLabel:", ", ".join(item["raw_altLabel"]) if item["raw_altLabel"] else "—")
                st.write("broader:", ", ".join(item["raw_broader"]) if item["raw_broader"] else "—")
                st.write("narrower:", ", ".join(item["raw_narrower"]) if item["raw_narrower"] else "—")
                st.divider()

            if errors:
                st.error("\n".join(errors))

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
