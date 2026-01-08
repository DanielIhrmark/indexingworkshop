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

# IMPORTANT: tolerate BOTH SAO scheme URI variants (slash/no-slash)
SAO_SCHEME_URIS = [
    "https://id.kb.se/term/sao/",
    "https://id.kb.se/term/sao",
]

# Community Cloud tuning
SPARQL_TIMEOUT_SECONDS = 60
SPARQL_RETRIES = 2
SPARQL_BACKOFF_SECONDS = 1

LABEL_CANDIDATE_LIMIT = 20
SAO_FILTER_LIMIT = 50
ALTLABEL_LIMIT = 50

# Optional offline fallback (extend as needed for workshop stability)
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
# SAO: minimal SPARQL (altLabels only)
# -----------------------------
@st.cache_data(ttl=3600)
def label_search_candidates(
    endpoint: str,
    token: str,
    mode: str,
    limit: int = LABEL_CANDIDATE_LIMIT
) -> List[Tuple[str, str]]:
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

    # NOTE: allow lang("sv") OR untagged (lang="") to reduce false negatives
    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?s ?label WHERE {{
      {{
        ?s kbv:prefLabel ?label .
        FILTER(lang(?label) = "sv" || lang(?label) = "")
        {label_filter}
      }}
      UNION
      {{
        ?s skos:prefLabel ?label .
        FILTER(lang(?label) = "sv" || lang(?label) = "")
        {label_filter}
      }}
      UNION
      {{
        ?s rdfs:label ?label .
        FILTER(lang(?label) = "sv" || lang(?label) = "")
        {label_filter}
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    return [(b["s"]["value"], b["label"]["value"]) for b in data.get("results", {}).get("bindings", [])]


@st.cache_data(ttl=3600)
def filter_to_sao(endpoint: str, uris: List[str], limit: int = SAO_FILTER_LIMIT) -> List[str]:
    """
    Filters a list of URIs to those that are in SAO, tolerating both scheme variants.
    """
    if not uris:
        return []
    uris = uris[:100]
    values = " ".join(f"<{u}>" for u in uris)
    schemes = " ".join(f"<{u}>" for u in SAO_SCHEME_URIS)

    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?s WHERE {{
      VALUES ?s {{ {values} }}
      VALUES ?scheme {{ {schemes} }}

      {{
        ?s kbv:inScheme ?scheme .
      }}
      UNION
      {{
        ?s skos:inScheme ?scheme .
      }}
      UNION
      {{
        ?s kbv:inVocabulary ?scheme .
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    return [b["s"]["value"] for b in data.get("results", {}).get("bindings", [])]


@st.cache_data(ttl=3600)
def try_direct_sao_uri(endpoint: str, token: str) -> Optional[str]:
    """
    Direct-URI fast path:
    If the concept exists at https://id.kb.se/term/sao/<Token> (case sensitive),
    confirm it is in SAO and return the URI.
    """
    token = (token or "").strip()
    if not token:
        return None

    # Try as-is and TitleCase first char
    candidates = []
    candidates.append(f"https://id.kb.se/term/sao/{urllib.parse.quote(token)}")
    if token[0].isalpha():
        candidates.append(f"https://id.kb.se/term/sao/{urllib.parse.quote(token[0].upper() + token[1:])}")

    schemes = " ".join(f"<{u}>" for u in SAO_SCHEME_URIS)

    for uri in candidates:
        sparql = f"""
        PREFIX kbv: <https://id.kb.se/vocab/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT (COUNT(*) AS ?n) WHERE {{
          VALUES ?scheme {{ {schemes} }}
          VALUES ?term {{ <{uri}> }}
          ?term (kbv:inScheme|skos:inScheme|kbv:inVocabulary) ?scheme .
        }}
        """
        data = sparql_select_json(endpoint, sparql)
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            n = int(bindings[0]["n"]["value"])
            if n > 0:
                return uri

    return None


@st.cache_data(ttl=3600)
def sao_altlabels(endpoint: str, term_uri: str, limit: int = ALTLABEL_LIMIT) -> List[str]:
    """
    Fetch Swedish altLabels (kbv:altLabel and skos:altLabel). Allow untagged too.
    """
    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?alt WHERE {{
      VALUES ?term {{ <{term_uri}> }}
      {{
        ?term kbv:altLabel ?alt .
        FILTER(lang(?alt)="sv" || lang(?alt)="")
      }}
      UNION
      {{
        ?term skos:altLabel ?alt .
        FILTER(lang(?alt)="sv" || lang(?alt)="")
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


def best_sao_uri_for_token(endpoint: str, token: str) -> Tuple[Optional[str], List[Tuple[str, str]], List[str], str]:
    """
    Returns (best_uri, label_candidates, sao_uris, lookup_token_used).
    Includes a direct URI fast-path to catch known SAO URIs like .../Fyrhjulingar.
    """
    # 0) Fast path: if token maps directly to a known SAO URI
    direct = try_direct_sao_uri(endpoint, token)
    if direct:
        return direct, [], [direct], token

    # 1) Fall back to label lookup
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
            return best[0], cands, sao_uris, token

    return None, cands, sao_uris, token


# -----------------------------
# Expansion: altLabels only + explicit run + session cache + fallback
# -----------------------------
def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}  # token -> payload
    if "enable_sao" not in st.session_state:
        st.session_state["enable_sao"] = False
    if "run_sao_now" not in st.session_state:
        st.session_state["run_sao_now"] = False


def compute_altlabel_expansion(endpoint: str, token: str) -> Dict:
    payload = {
        "token": token,
        "lookup_token_used": token,
        "source": "None",
        "best_sao_uri": None,
        "candidate_count": 0,
        "candidate_preview": [],
        "sao_uri_count": 0,
        "sao_uri_preview": [],
        "altLabel": [],
        "expansion_tokens": [],
        "error": None,
    }

    try:
        best_uri, candidates, sao_uris, tok_used = best_sao_uri_for_token(endpoint, token)
        payload["lookup_token_used"] = tok_used

        payload["best_sao_uri"] = best_uri
        payload["candidate_count"] = len(candidates)
        payload["candidate_preview"] = [(lbl, uri) for (uri, lbl) in candidates[:8]]
        payload["sao_uri_count"] = len(sao_uris)
        payload["sao_uri_preview"] = sao_uris[:8]

        if best_uri:
            alts = sao_altlabels(endpoint, best_uri, limit=ALTLABEL_LIMIT)
            payload["altLabel"] = alts

            # expansion tokens derived from altLabels
            toks = []
            seen = set([token.lower()])
            for ph in alts:
                for t in tokenize(ph):
                    if t not in seen:
                        seen.add(t)
                        toks.append(t)

            payload["expansion_tokens"] = toks
            payload["source"] = "SAO (live, altLabels)"
            return payload

    except Exception as e:
        payload["error"] = str(e)

    # Optional fallback if live fails
    tkey = token.lower()
    if tkey in FALLBACK_EXPANSIONS:
        payload["source"] = "Fallback (offline)"
        payload["altLabel"] = FALLBACK_EXPANSIONS[tkey]
        toks = []
        seen = set([tkey])
        for ph in payload["altLabel"]:
            for t in tokenize(ph):
                if t not in seen:
                    seen.add(t)
                    toks.append(t)
        payload["expansion_tokens"] = toks

    return payload


def search_with_expansion(inv, query, endpoint, expand_enabled, run_expansion_now):
    base_tokens = tokenize(query)
    if not base_tokens:
        return set(), [], [], []

    ensure_state()

    groups, errors, debug = [], [], []
    for tok in base_tokens:
        g = [tok]
        dbg = {
            "token": tok,
            "lookup_token_used": tok,
            "source": "Not run (click 'Run SAO altLabel expansion now')",
            "best_sao_uri": None,
            "candidate_preview": [],
            "sao_uri_preview": [],
            "altLabel": [],
            "expansion_tokens": [],
            "error": None,
        }

        if expand_enabled:
            # Only compute when button pressed; otherwise use cache if present
            if run_expansion_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_expansion_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_altlabel_expansion(endpoint, tok)
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

ensure_state()

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

available_baseline = [f for f in BASELINE_FIELDS if f in df.columns]
available_index = [f for f in INDEX_FIELDS if f in df.columns]

query = st.text_input("Query")

expand_query = st.checkbox(
    "Enable SAO altLabel expansion (SPARQL)",
    value=st.session_state["enable_sao"],
    key="enable_sao",
)

if expand_query:
    if st.button("Run SAO altLabel expansion now", key="run_sao_btn"):
        st.session_state["run_sao_now"] = True
else:
    st.session_state["run_sao_now"] = False

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
    expand_enabled=expand_query,
    run_expansion_now=st.session_state["run_sao_now"],
)

# reset one-shot flag after use
st.session_state["run_sao_now"] = False

if query.strip():
    if expand_query:
        with st.expander("SAO matches and altLabels (debug)", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            st.markdown("#### Expansion status per token (altLabels only)")
            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "—"))
                st.write("Lookup token used:", item.get("lookup_token_used", "—"))

                if item.get("best_sao_uri"):
                    st.write("Matched SAO URI:")
                    st.code(item["best_sao_uri"])
                else:
                    st.write("Matched SAO URI: —")

                if item.get("candidate_preview"):
                    st.write("Label candidates (preview):")
                    for lbl, uri in item["candidate_preview"]:
                        st.write(f"- {lbl}")
                else:
                    st.write("Label candidates (preview): —")

                alts = item.get("altLabel", [])
                st.write("altLabel:", ", ".join(alts) if alts else "—")
                st.divider()

            if errors:
                st.warning(
                    "Some live SAO lookups timed out or failed. Fallback expansions may be shown where available.\n\n"
                    + "\n".join(errors)
                )
            else:
                st.caption("Tip: On Community Cloud, click 'Run SAO altLabel expansion now' only after you finish typing your query.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
