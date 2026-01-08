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

# Accept BOTH SAO scheme URI variants (slash/no-slash)
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
VARIANT_LIMIT = 50

# Heuristics for “variant-like” strings
MAX_VARIANT_LEN = 60  # keep variants short
MIN_VARIANT_LEN = 2


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
# SAO: label candidate search + filtering to SAO
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
    token = (token or "").strip()
    if not token:
        return None

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
        if bindings and int(bindings[0]["n"]["value"]) > 0:
            return uri

    return None


def best_sao_uri_for_token(endpoint: str, token: str) -> Tuple[Optional[str], List[Tuple[str, str]], List[str], str]:
    direct = try_direct_sao_uri(endpoint, token)
    if direct:
        return direct, [], [direct], token

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
# SAO prefLabel + variants via depth-2 literal harvesting
# -----------------------------
@st.cache_data(ttl=3600)
def sao_preflabel(endpoint: str, term_uri: str) -> Optional[str]:
    sparql = f"""
    PREFIX kbv: <https://id.kb.se/vocab/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?lbl WHERE {{
      VALUES ?term {{ <{term_uri}> }}
      {{
        ?term kbv:prefLabel ?lbl .
      }}
      UNION
      {{
        ?term skos:prefLabel ?lbl .
      }}
      FILTER(lang(?lbl)="sv" || lang(?lbl)="")
    }}
    LIMIT 1
    """
    data = sparql_select_json(endpoint, sparql)
    bindings = data.get("results", {}).get("bindings", [])
    return bindings[0]["lbl"]["value"] if bindings else None


@st.cache_data(ttl=3600)
def sao_variant_literals_depth2(endpoint: str, term_uri: str, limit: int = 300) -> List[Tuple[str, str]]:
    """
    Harvest literal strings at depth 1 and depth 2:
      - <term> ?p ?lit
      - <term> ?p ?node . ?node ?p2 ?lit
    Return (predicate_path, literal_value) where predicate_path is either ?p or ?p/?p2.
    """
    sparql = f"""
    SELECT DISTINCT ?p ?p2 ?v WHERE {{
      VALUES ?term {{ <{term_uri}> }}

      {{
        ?term ?p ?v .
        FILTER(isLiteral(?v))
        FILTER(lang(?v)="sv" || lang(?v)="")
        BIND("" AS ?p2)
      }}
      UNION
      {{
        ?term ?p ?node .
        FILTER(isIRI(?node) || isBlank(?node))
        ?node ?p2 ?v .
        FILTER(isLiteral(?v))
        FILTER(lang(?v)="sv" || lang(?v)="")
      }}
    }}
    LIMIT {int(limit)}
    """
    data = sparql_select_json(endpoint, sparql)
    out: List[Tuple[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        p = b["p"]["value"]
        p2 = b.get("p2", {}).get("value", "")
        v = b["v"]["value"]
        path = p if not p2 else f"{p} / {p2}"
        out.append((path, v))
    return out


def _looks_like_noise(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if len(s) < MIN_VARIANT_LEN or len(s) > MAX_VARIANT_LEN:
        return True
    # drop very sentence-like strings (notes)
    if s.count(" ") >= 6:
        return True
    return False


@st.cache_data(ttl=3600)
def sao_variants(endpoint: str, term_uri: str, limit: int = VARIANT_LIMIT) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Return variants (strings) + evidence (predicate-path, value).
    This works even when “variant” is modeled via an intermediate node.
    """
    pref = (sao_preflabel(endpoint, term_uri) or "").strip()
    pairs = sao_variant_literals_depth2(endpoint, term_uri)

    # Keep candidate strings that are short and not the prefLabel itself
    candidates: List[Tuple[str, str]] = []
    for path, val in pairs:
        val_s = (val or "").strip()
        if not val_s:
            continue
        if val_s == pref:
            continue
        if _looks_like_noise(val_s):
            continue
        candidates.append((path, val_s))

    # Prefer any paths that include "variant" in the predicate name (if present)
    def score(path_val: Tuple[str, str]) -> Tuple[int, int]:
        path, val = path_val
        path_l = path.lower()
        return (0 if "variant" in path_l else 1, len(val))

    candidates.sort(key=score)

    # De-dup values case-insensitive and cap
    seen = set()
    variants: List[str] = []
    evidence: List[Tuple[str, str]] = []
    for path, val in candidates:
        k = val.lower()
        if k in seen:
            continue
        seen.add(k)
        variants.append(val)
        evidence.append((path, val))
        if len(variants) >= limit:
            break

    return variants, evidence


# -----------------------------
# Expansion: variants-only + explicit run + session cache
# -----------------------------
def ensure_state():
    if "sao_cache" not in st.session_state:
        st.session_state["sao_cache"] = {}
    if "enable_sao" not in st.session_state:
        st.session_state["enable_sao"] = False
    if "run_sao_now" not in st.session_state:
        st.session_state["run_sao_now"] = False


def compute_variant_expansion(endpoint: str, token: str) -> Dict:
    payload = {
        "token": token,
        "lookup_token_used": token,
        "source": "None",
        "best_sao_uri": None,
        "prefLabel": None,
        "candidate_count": 0,
        "candidate_preview": [],
        "sao_uri_count": 0,
        "sao_uri_preview": [],
        "variants": [],
        "variant_evidence": [],
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
            payload["prefLabel"] = sao_preflabel(endpoint, best_uri)
            vars_, evidence = sao_variants(endpoint, best_uri, limit=VARIANT_LIMIT)
            payload["variants"] = vars_
            payload["variant_evidence"] = evidence

            toks = []
            seen = set([token.lower()])
            for ph in vars_:
                for t in tokenize(ph):
                    if t not in seen:
                        seen.add(t)
                        toks.append(t)

            payload["expansion_tokens"] = toks
            payload["source"] = "SAO (live, variants depth-2)"
            return payload

    except Exception as e:
        payload["error"] = str(e)

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
            "source": "Not run (click 'Run SAO variant expansion now')",
            "best_sao_uri": None,
            "prefLabel": None,
            "variants": [],
            "variant_evidence": [],
            "expansion_tokens": [],
            "error": None,
        }

        if expand_enabled:
            if run_expansion_now or tok in st.session_state["sao_cache"]:
                try:
                    if run_expansion_now or tok not in st.session_state["sao_cache"]:
                        st.session_state["sao_cache"][tok] = compute_variant_expansion(endpoint, tok)
                    dbg = st.session_state["sao_cache"][tok]
                    g += dbg.get("expansion_tokens", [])
                    if dbg.get("error"):
                        errors.append(f"{tok}: {dbg['error']}")
                except Exception as e:
                    errors.append(f"{tok}: {e}")

        deduped = []
        seen = set()
        for t in g:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        groups.append(deduped)
        debug.append(dbg)

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
st.set_page_config(page_title="1BO416 Indexeringslabb", layout="wide")
st.title("1BO416 Indexeringslabb")

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

available_baseline = [f for f in BASELINE_FIELDS if f in df.columns]
available_index = [f for f in INDEX_FIELDS if f in df.columns]

query = st.text_input("Query")

expand_query = st.checkbox(
    "Enable SAO variant expansion (SPARQL)",
    value=st.session_state["enable_sao"],
    key="enable_sao",
)

if expand_query:
    if st.button("Run SAO expansion now", key="run_sao_btn"):
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

st.session_state["run_sao_now"] = False

if query.strip():
    if expand_query:
        with st.expander("Query expansion (debug)", expanded=True):
            for i, g in enumerate(groups, 1):
                st.write(f"Concept {i}: " + " OR ".join(g))

            st.markdown("#### SAO resolution + variants (depth-2 harvest)")
            for item in debug:
                st.markdown(f"**Token:** `{item.get('token')}`")
                st.write("Source:", item.get("source", "—"))
                st.write("Lookup token used:", item.get("lookup_token_used", "—"))

                if item.get("best_sao_uri"):
                    st.write("Matched SAO URI:")
                    st.code(item["best_sao_uri"])
                else:
                    st.write("Matched SAO URI: —")

                st.write("prefLabel:", item.get("prefLabel", "—"))

                vars_ = item.get("variants", [])
                st.write("variants:", ", ".join(vars_) if vars_ else "—")

                evidence = item.get("variant_evidence", [])
                if evidence:
                    st.caption("Variant evidence (predicate path → value):")
                    for p, v in evidence[:10]:
                        st.write(f"- {p} → {v}")

                st.divider()

            if errors:
                st.warning("\n".join(errors))
            else:
                st.caption("Tip: Click 'Run SAO expansion now' after typing. Results are cached per session.")

    if ids:
        res = df[df[id_col].astype(str).isin(ids)].copy()
        st.success(f"{len(res)} results")
        st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.warning("No results")
else:
    st.info("Enter a query to search.")
