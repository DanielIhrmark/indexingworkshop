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
# SAO SPARQL helpers
# -----------------------------
@st.cache_data(ttl=3600)
def sao_autocomplete(endpoint: str, q: str, limit: int = 20) -> List[Tuple[str, str]]:
    q = (q or "").strip()
    if len(q) < 2:
        return []

    sparql = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?term ?label WHERE {{
      ?term skos:inScheme <https://id.kb.se/term/sao> ;
            skos:prefLabel ?label .
      FILTER(lang(?label) = "sv")
      FILTER regex(str(?label), "{re.escape(q)}", "i")
    }}
    ORDER BY LCASE(STR(?label))
    LIMIT {limit}
    """

    r = requests.get(
        endpoint,
        params={"query": sparql, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    return [(b["label"]["value"], b["term"]["value"])
            for b in data["results"]["bindings"]]


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
    for b in data["results"]["bindings"]:
        ctx[b["p"]["value"]].append(b["oLabel"]["value"])

    for k in ctx:
        ctx[k] = sorted(set(ctx[k]), key=str.lower)
    return ctx


# -----------------------------
# SAO query expansion
# -----------------------------
@st.cache_data(ttl=3600)
def sao_lookup_by_pref_label(endpoint: str, label: str) -> Optional[str]:
    label = label.replace('"', '\\"')
    sparql = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?term WHERE {{
      ?term skos:inScheme <https://id.kb.se/term/sao> ;
            skos:prefLabel ?label .
      FILTER(lang(?label)="sv")
      FILTER(LCASE(STR(?label)) = LCASE("{label}"))
    }}
    LIMIT 1
    """
    r = requests.get(
        endpoint,
        params={"query": sparql, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=20,
    )
    r.raise_for_status()
    bindings = r.json()["results"]["bindings"]
    return bindings[0]["term"]["value"] if bindings else None


def expand_token(endpoint: str, token: str, include_hierarchy: bool) -> List[str]:
    uri = sao_lookup_by_pref_label(endpoint, token)
    if not uri:
        return []

    ctx = sao_term_context(endpoint, uri)
    labels = ctx["altLabel"]
    if include_hierarchy:
        labels += ctx["broader"] + ctx["narrower"]

    out = []
    seen = set()
    for lbl in labels:
        for t in tokenize(lbl):
            if t != token and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def search_with_expansion(inv, query, endpoint, expand, include_hierarchy):
    base_tokens = tokenize(query)
    if not base_tokens:
        return set(), []

    groups = []
    for tok in base_tokens:
        g = [tok]
        if expand:
            g += expand_token(endpoint, tok, include_hierarchy)
        groups.append(g)

    sets = []
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

id_col = st.selectbox("ID column", options=df.columns)

available_baseline = [f for f in BASELINE_FIELDS if f in df.columns]
available_index = [f for f in INDEX_FIELDS if f in df.columns]

query = st.text_input("Query")

expand_query = st.checkbox("Expand query using Svenska Ämnesord (SAO)")
include_hierarchy = st.checkbox("Include broader/narrower terms", value=True)

mode = st.radio(
    "Search mode",
    ["Baseline", "Enriched"],
    horizontal=True,
)

fields = available_baseline if mode == "Baseline" else list(set(available_baseline + available_index))

inv = build_inverted_index(df, fields, id_col)

ids, groups = search_with_expansion(
    inv,
    query,
    sparql_endpoint,
    expand_query,
    include_hierarchy,
)

if query:
    if ids:
        res = df[df[id_col].astype(str).isin(ids)]
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True)
        if expand_query:
            with st.expander("Query expansion"):
                for i, g in enumerate(groups, 1):
                    st.write(f"Concept {i}: " + " OR ".join(g))
    else:
        st.warning("No results")

st.divider()
st.subheader("SAO lookup")

sao_q = st.text_input("Search SAO")
if sao_q:
    for lbl, uri in sao_autocomplete(sparql_endpoint, sao_q):
        st.write(lbl)
        st.code(uri)
