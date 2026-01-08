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


def simple_search(inv: Dict[str, set], query: str) -> Tuple[set, List[str]]:
    toks = tokenize(query)
    if not toks:
        return set(), []
    sets = [inv.get(t, set()) for t in toks]
    return (set.intersection(*sets) if sets else set()), toks


# -----------------------------
# SAO SPARQL helpers (for expansion)
# -----------------------------
@st.cache_data(ttl=3600)
def sao_term_context(endpoint: str, term_uri: str) -> Dict[str, List[str]]:
    sparql = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?p ?oLabel WHERE {{
      VALUES ?term {{ <{term_uri}> }}
      {{
        ?term skos:broader ?o .
        ?o skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("broader" AS ?p)
      }}
      UNION
      {{
        ?term skos:narrower ?o .
        ?o skos:prefLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("narrower" AS ?p)
      }}
      UNION
      {{
        ?term skos:altLabel ?oLabel .
        FILTER(lang(?oLabel)="sv")
        BIND("altLabel" AS ?p)
      }}
    }}
    """

    r = requests.get(
        endpoint,
        params={"query": sparql, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    ctx = {"broader": [], "narrower": [], "altLabel": []}
    for b in data.get("results", {}).get("bindings", []):
        p = b["p"]["value"]
        lbl = b["oLabel"]["value"]
        if p in ctx:
            ctx[p].append(lbl)

    for k in ctx:
        ctx[k] = sorted(set(ctx[k]), key=str.lower)
    return ctx


@st.cache_data(ttl=3600)
def sao_lookup_best_term_uri(endpoint: str, token: str) -> Optional[str]:
    """
    Find a 'best' SAO concept URI for a token.
    Strategy:
      1) exact prefLabel match
      2) prefLabel starts-with match
      3) prefLabel contains match
    Picks the shortest matching label as a simple heuristic.
    """
    token = (token or "").strip()
    if len(token) < 2:
        return None

    safe = token.replace('"', '\\"')
    headers = {"Accept": "application/sparql-results+json"}

    # 1) Exact match
    sparql_exact = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?term ?label WHERE {{
      ?term skos:inScheme <https://id.kb.se/term/sao> ;
            skos:prefLabel ?label .
      FILTER(lang(?label)="sv")
      FILTER(LCASE(STR(?label)) = LCASE("{safe}"))
    }}
    LIMIT 5
    """
    r = requests.get(endpoint, params={"query": sparql_exact, "format": "json"}, headers=headers, timeout=20)
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0]["term"]["value"]

    # 2) Starts-with match
    sparql_starts = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?term ?label WHERE {{
      ?term skos:inScheme <https://id.kb.se/term/sao> ;
            skos:prefLabel ?label .
      FILTER(lang(?label)="sv")
      FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{safe}")))
    }}
    LIMIT 20
    """
    r = requests.get(endpoint, params={"query": sparql_starts, "format": "json"}, headers=headers, timeout=20)
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    if bindings:
        # pick shortest label (often the most general/closest match)
        best = sorted(bindings, key=lambda b: len(b["label"]["value"]))[0]
        return best["term"]["value"]

    # 3) Contains match (regex)
    sparql_contains = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?term ?label WHERE {{
      ?term skos:inScheme <https://id.kb.se/term/sao> ;
            skos:prefLabel ?label .
      FILTER(lang(?label)="sv")
      FILTER regex(str(?label), "{re.escape(token)}", "i")
    }}
    LIMIT 50
    """
    r = requests.get(endpoint, params={"query": sparql_contains, "format": "json"}, headers=headers, timeout=20)
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    if not bindings:
        return None

    best = sorted(bindings, key=lambda b: len(b["label"]["value"]))[0]
    return best["term"]["value"]



def expand_token(endpoint: str, token: str, include_hierarchy: bool) -> List[str]:
    uri = sao_lookup_best_term_uri(endpoint, token)
    if not uri:
        return []

    ctx = sao_term_context(endpoint, uri)

    labels = list(ctx.get("altLabel", []))
    if include_hierarchy:
        labels += ctx.get("broader", []) + ctx.get("narrower", [])

    out: List[str] = []
    seen = set()

    # Tokenize all returned labels so they match your inverted-index tokens
    for lbl in labels:
        for t in tokenize(lbl):
            if t == token:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)

    return out



def search_with_expansion(
    inv: Dict[str, set],
    query: str,
    endpoint: str,
    expand: bool,
    include_hierarchy: bool,
) -> Tuple[set, List[List[str]]]:
    base_tokens = tokenize(query)
    if not base_tokens:
        return set(), []

    groups: List[List[str]] = []
    for tok in base_tokens:
        g = [tok]
        if expand:
            try:
                g += expand_token(endpoint, tok, include_hierarchy)
            except Exception:
                pass
        # de-dup within group
        deduped = []
        seen = set()
        for t in g:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        groups.append(deduped)

    sets: List[set] = []
    for g in groups:
        s = set()
        for t in g:
            s |= inv.get(t, set())
        sets.append(s)

    return (set.intersection(*sets) if sets else set()), groups


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

ids, groups = search_with_expansion(inv, query, sparql_endpoint, expand_query, include_hierarchy)

if query.strip():
    # Always show expansion if enabled, even when there are no local hits
    if expand_query:
        with st.expander("Query expansion", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")

