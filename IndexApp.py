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
    r = requests.get(endpoint, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# -----------------------------
# SAO matching + expansion (GRAPH ?g)
# -----------------------------
@st.cache_data(ttl=3600)
def sao_lookup_candidates(endpoint: str, token: str, mode: str, limit: int) -> List[Tuple[str, str]]:
    """
    Return list of (term_uri, label) candidates for token by:
      mode = "exact" | "starts" | "contains"
    Searches within GRAPH ?g to ensure we see named-graph data.
    """
    token = (token or "").strip()
    if len(token) < 2:
        return []

    safe = token.replace('"', '\\"')
    if mode == "exact":
        label_filter = f'FILTER(LCASE(STR(?label)) = LCASE("{safe}"))'
    elif mode == "starts":
        label_filter = f'FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{safe}")))'
    else:  # contains
        label_filter = f'FILTER regex(str(?label), "{re.escape(token)}", "i")'

    sparql = f"""
    PREFIX : <https://id.kb.se/vocab/>

    SELECT DISTINCT ?term ?label WHERE {{
      GRAPH ?g {{
        ?term :inScheme <{SAO_SCHEME_URI}> ;
              :prefLabel ?label .
        FILTER(lang(?label) = "sv")
        {label_filter}
      }}
    }}
    LIMIT {int(limit)}
    """

    data = sparql_select_json(endpoint, sparql)
    out = []
    for b in data.get("results", {}).get("bindings", []):
        out.append((b["term"]["value"], b["label"]["value"]))
    return out


@st.cache_data(ttl=3600)
def sao_lookup_best_term(endpoint: str, token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (term_uri, prefLabel) for a best SAO match.
    Strategy: exact -> starts-with -> contains; pick shortest label.
    """
    # 1) exact
    cands = sao_lookup_candidates(endpoint, token, mode="exact", limit=20)
    if cands:
        return cands[0][0], cands[0][1]

    # 2) starts-with
    cands = sao_lookup_candidates(endpoint, token, mode="starts", limit=100)
    if cands:
        best = sorted(cands, key=lambda x: len(x[1]))[0]
        return best[0], best[1]

    # 3) contains
    cands = sao_lookup_candidates(endpoint, token, mode="contains", limit=200)
    if cands:
        best = sorted(cands, key=lambda x: len(x[1]))[0]
        return best[0], best[1]

    return None, None


@st.cache_data(ttl=3600)
def sao_term_context(endpoint: str, term_uri: str) -> Dict[str, List[str]]:
    """
    Fetch raw related labels as phrases (altLabel, broader, narrower).
    Narrower is inferred via inverse :broader in GRAPH ?g.
    """
    sparql = f"""
    PREFIX : <https://id.kb.se/vocab/>

    SELECT DISTINCT ?p ?oLabel WHERE {{
      VALUES ?term {{ <{term_uri}> }}

      GRAPH ?g {{

        # broader
        {{
          ?term :broader ?o .
          ?o :prefLabel ?oLabel .
          FILTER(lang(?oLabel) = "sv")
          BIND("broader" AS ?p)
        }}

        UNION

        # narrower (inverse broader)
        {{
          ?n :broader ?term .
          ?n :prefLabel ?oLabel .
          FILTER(lang(?oLabel) = "sv")
          BIND("narrower" AS ?p)
        }}

        UNION

        # altLabel
        {{
          ?term :altLabel ?oLabel .
          FILTER(lang(?oLabel) = "sv")
          BIND("altLabel" AS ?p)
        }}
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


def expand_token_tokens(
    endpoint: str,
    token: str,
    include_hierarchy: bool,
) -> Tuple[List[str], Dict[str, List[str]], Optional[str], Optional[str]]:
    """
    Returns:
      - expansion tokens (single words) for local retrieval
      - raw context labels (phrases) for debugging
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
    """
    Returns:
      - ids (set)
      - groups (OR-groups used for retrieval)
      - errors (list)
      - debug (raw SAO labels + matched URI/prefLabel per token)
    """
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

        # de-dup within group
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
