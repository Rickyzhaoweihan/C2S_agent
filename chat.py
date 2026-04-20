"""
chat.py
-------
Chat agent for the Gut Cell Database.

Grounded in:
    Vandana et al. "Integrative Single-Cell Spatial Multi-Omics Analysis
    of Human Fetal Intestine."

Public API
----------
    CatalogueMatcher   — parse free-text queries into CellQuery objects
    C2SChatbot         — interactive chat wrapper around CellDatabaseAgent

    C2SChatbot.parse_query(text)  → CellQuery
    C2SChatbot.chat(text)         → str
    C2SChatbot.reset()            — clear conversation history
    C2SChatbot.run_interactive()  — start an interactive loop

CLI
---
    python chat.py \\
        --db_path   /path/to/guts.db \\
        --arrow_dir /path/to/arrow_ds \\
        --nl_model  /path/to/C2S-Scale-Pythia-1b-pt

    python chat.py ... --query "tell me about goblet cells in the colon"
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import replace as _dc_replace
from typing import Optional

import numpy as np

# ── Import project modules ─────────────────────────────────────────────────────
try:
    import database as db
    from cell_db_agent import CellDatabaseAgent, CellQuery
    from biology_context import (
        CELL_TYPE_ALIASES,
        REGION_ALIASES,
        MARKER_TO_CELL_TYPE,
        CELL_TYPE_MARKERS,
        EEC_SUBTYPES,
    )
    # NOTE: SYSTEM_PREAMBLE is defined in biology_context for use with a
    # general-purpose LLM backbone.  The current pipeline uses the C2S model
    # directly (via vLLM), so SYSTEM_PREAMBLE is not passed here.  Import it
    # if/when a general-LLM synthesis step is added.
except ImportError as e:
    print(f"ERROR: Could not import project module — {e}")
    print("Make sure database.py, cell_db_agent.py, and biology_context.py are in the same directory.")
    sys.exit(1)


# ── Query parsing ──────────────────────────────────────────────────────────────

class CatalogueMatcher:
    """
    Three-tier resolution:
      1. Marker gene lookup  (MECOM → stem cell)
      2. Biology alias       (GLP-1 → L cell → search catalogue)
      3. Fuzzy n-gram match  (difflib against live catalogue values)

    Also extracts numeric patterns for clusters and QC filters.
    """

    DIM_MAP = {
        "cell_type":       "cell_types",
        "tissue":          "tissues",
        "batch_condition": "batches",
        "sample_type":     "sample_types",
        "region":          "regions",
        "orig_ident":      "orig_idents",
        "age":             "ages",
        "organism":        "organisms",
        "assay":           "assays",
        "sex":             "sexes",
    }

    def __init__(self, catalogue: dict):
        self.catalogue = catalogue
        self._lower = {
            k: [str(v).lower() for v in vals]
            for k, vals in catalogue.items()
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fuzzy(self, text: str, cat_key: str, cutoff: float = 0.72) -> Optional[str]:
        t         = text.strip().lower()
        cands     = self._lower.get(cat_key, [])
        originals = self.catalogue.get(cat_key, [])
        if not cands:
            return None
        for i, c in enumerate(cands):
            # Substring match only when the query covers ≥50% of the catalogue
            # entry length.  Without this guard, a single generic word like
            # "cells" would match "Goblet Cells" (it's a substring), causing
            # "stem cells duodenum" to be mis-classified as Goblet Cells.
            query_covers_half = len(t) >= len(c) * 0.50
            if (t in c and query_covers_half) or c in t:
                return originals[i]
        hits = difflib.get_close_matches(t, cands, n=1, cutoff=cutoff)
        if hits:
            return originals[cands.index(hits[0])]
        return None

    def _resolve_cell_type(self, text: str) -> Optional[str]:
        t = text.strip()
        t_lower = t.lower()
        # Tier 1 — marker gene (exact case-insensitive match)
        for marker, cell_type in MARKER_TO_CELL_TYPE.items():
            if marker.lower() == t_lower:
                r = self._fuzzy(cell_type, "cell_types", cutoff=0.75)
                if r:
                    return r
        # Tier 2 — biology alias
        # Guard: only match if the query token covers the alias (alias in query),
        # OR the query token is ≥4 chars and appears inside the alias.
        # This prevents short stopwords like "in", "at", "of" from accidentally
        # substring-matching alias entries such as "absorptive".
        # "EEC"/"eec" are appended as fallback hints so that EEC subtype names
        # (L cell, K cell, …) can still resolve to "EECs" if the catalogue uses
        # that as the umbrella term rather than "enteroendocrine cell".
        for alias, hints in CELL_TYPE_ALIASES.items():
            alias_lower = alias.lower()
            if alias_lower in t_lower or (len(t_lower) >= 4 and t_lower in alias_lower):
                # Only append EEC fallback hints for aliases whose hint list
                # already signals an enteroendocrine/EEC relationship.
                _is_eec_alias = any(
                    "enteroendocrine" in h.lower() or "eec" in h.lower()
                    for h in hints
                )
                _extra = ["EEC", "eec"] if _is_eec_alias else []
                for hint in list(hints) + _extra:
                    r = self._fuzzy(hint, "cell_types", cutoff=0.75)
                    if r:
                        return r
        # Tier 2.5 — EEC subtype names (for queries like "L cell", "K cell", …)
        # If the query matches any EEC subtype name, try EEC umbrella terms.
        # The ≥4-char guard prevents stopwords ("in", "at", …) from triggering
        # this tier just because they appear inside "endocrine progenitor" etc.
        from biology_context import EEC_SUBTYPES
        for eec_name in EEC_SUBTYPES:
            eec_lower = eec_name.lower()
            if eec_lower in t_lower or (len(t_lower) >= 4 and t_lower in eec_lower):
                for eec_hint in ["EEC", "eec", "enteroendocrine"]:
                    r = self._fuzzy(eec_hint, "cell_types", cutoff=0.60)
                    if r:
                        return r
        # Tier 3 — direct fuzzy.
        # Raised cutoff to 0.82 to prevent suffix-only false positives
        # (e.g. "l cells" matching "goblet cells" because both end in " cells").
        return self._fuzzy(t, "cell_types", cutoff=0.82)

    def _resolve_tissue(self, text: str) -> Optional[str]:
        t = text.strip().lower()
        for alias, canonical in REGION_ALIASES.items():
            if alias.lower() in t or t in alias.lower():
                r = self._fuzzy(canonical, "tissues", cutoff=0.6)
                if r:
                    return r
        return self._fuzzy(t, "tissues")

    # ── Public ─────────────────────────────────────────────────────────────────

    def parse(self, user_text: str) -> dict:
        """Return a dict of filter kwargs for db.fetch_cells_by_filters()."""
        text = user_text.strip()

        # Numeric / cluster patterns
        seurat_cluster     = _ri(text, r"seurat\s+cluster\s+(\d+)")
        cluster_resolution = _rf(text, r"resolution\s+([\d.]+)")
        cluster_id         = _ri(text, r"resolution\s+[\d.]+\s+cluster\s+(\d+)")
        if seurat_cluster is None and cluster_resolution is None:
            seurat_cluster = _ri(text, r"\bcluster\s+(\d+)\b")
        percent_mt_max  = _rf(text, r"(?:percent|pct)[\s_]?mt\s*[<≤]\s*([\d.]+)")
        n_count_rna_min = _rf(text, r"(?:count[\s_]?rna|n_count)\s*[>≥]\s*([\d.]+)")
        n_count_rna_max = _rf(text, r"(?:count[\s_]?rna|n_count)\s*[<≤]\s*([\d.]+)")

        # N-gram categorical matching
        words  = re.sub(r"[^\w\s.\-]", " ", text).split()
        ngrams = [
            " ".join(words[i: i + n])
            for n in range(4, 0, -1)
            for i in range(len(words) - n + 1)
        ]

        matched: dict[str, Optional[str]] = {f: None for f in self.DIM_MAP}
        for ngram in ngrams:
            for field in self.DIM_MAP:
                if matched[field] is not None:
                    continue
                if field == "cell_type":
                    matched[field] = self._resolve_cell_type(ngram)
                elif field == "tissue":
                    matched[field] = self._resolve_tissue(ngram)
                else:
                    matched[field] = self._fuzzy(ngram, self.DIM_MAP[field])

        params = {k: v for k, v in matched.items() if v is not None}
        if seurat_cluster     is not None: params["seurat_cluster"]     = seurat_cluster
        if cluster_resolution is not None: params["cluster_resolution"] = cluster_resolution
        if cluster_id         is not None: params["cluster_id"]         = cluster_id
        if percent_mt_max     is not None: params["percent_mt_max"]     = percent_mt_max
        if n_count_rna_min    is not None: params["n_count_rna_min"]    = n_count_rna_min
        if n_count_rna_max    is not None: params["n_count_rna_max"]    = n_count_rna_max

        return params

    def parse_query(self, user_text: str) -> CellQuery:
        """Parse free-text into a CellQuery (used by C2SChatbot and the test notebook)."""
        params = self.parse(user_text)
        return CellQuery(**params)


# ── Gene detection ─────────────────────────────────────────────────────────────

# Common biology / English tokens that look like gene names but aren't
_NON_GENE_TOKENS = frozenset({
    # Biology acronyms
    "EEC", "DNA", "RNA", "PCR", "QC", "GI", "GIT", "IBD", "IBS",
    "GABA", "AMP", "ATP", "ADP", "NAD", "FAD",
    # Hormone/ligand abbreviations (not gene symbols)
    "GLP", "GIP", "SST",
    # Database / data identifiers
    "PMID", "IDS", "SQL",
    # Common English abbreviations
    "NOT", "OR", "AND", "FISH", "FACS", "ELISA", "GWAS",
    # Cell-type short forms (handled separately by CatalogueMatcher)
    "EEC", "TA", "ISC",
})


def _detect_gene_in_query(text: str) -> Optional[str]:
    """
    Scan *text* for a gene-like token.

    Priority:
      1. Token that is a known marker in MARKER_TO_CELL_TYPE (most reliable)
      2. Token that appears in any biology_context marker list
      3. First uppercase alphanumeric token of ≥3 chars not in _NON_GENE_TOKENS

    Returns the gene name (original casing preserved), or None.
    """
    # Match uppercase tokens: letter + letters/digits, 2–12 chars
    tokens = re.findall(r'\b[A-Z][A-Z0-9]{1,11}\b', text)
    tokens = [
        t for t in tokens
        if t not in _NON_GENE_TOKENS
        and not t.startswith("PMID")
        # Gene names have at most ~4 consecutive digits
        and not re.search(r'\d{5,}', t)
    ]

    # Tier 1: known marker
    for t in tokens:
        if t in MARKER_TO_CELL_TYPE:
            return t

    # Tier 2: any gene in biology_context marker lists
    from biology_context import CELL_TYPE_MARKERS, EEC_SUBTYPES
    known_genes: set[str] = set()
    for info in CELL_TYPE_MARKERS.values():
        known_genes.update(info.get("canonical_markers", []))
        known_genes.update(info.get("novel_markers", []))
    for info in EEC_SUBTYPES.values():
        known_genes.update(info.get("markers", []))
    for t in tokens:
        if t in known_genes:
            return t

    # Tier 3: first gene-like token (≥3 chars)
    for t in tokens:
        if len(t) >= 3:
            return t

    return None


def _ri(text, pattern):
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _rf(text, pattern):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


# ── Comparison query helpers ───────────────────────────────────────────────────

# Patterns that signal the user wants two cell types compared
_COMPARISON_RE = re.compile(
    r"(?:compare\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)"
    r"|(.+?)\s+vs\.?\s+(.+)"
    r"|difference[s]?\s+between\s+(.+?)\s+and\s+(.+)"
    r"|(.+?)\s+versus\s+(.+))",
    re.IGNORECASE,
)


def _extract_comparison_entities(text: str) -> Optional[tuple[str, str]]:
    """Return (entity_a, entity_b) if *text* is a comparison query, else None."""
    m = _COMPARISON_RE.search(text)
    if not m:
        return None
    # Pick the first non-None pair of groups
    groups = [g for g in m.groups() if g is not None]
    if len(groups) >= 2:
        return groups[0].strip(), groups[1].strip()
    return None


def _lookup_biology(name: str) -> Optional[tuple[str, dict]]:
    """
    Return (canonical_name, info_dict) for *name* from the curated biology
    knowledge base, or None if not found.
    """
    name_lower = name.lower()
    combined = {**CELL_TYPE_MARKERS, **EEC_SUBTYPES}
    # Exact / substring match first
    for ct, info in combined.items():
        if ct.lower() in name_lower or name_lower in ct.lower():
            return ct, info
    return None


def _build_comparison_response(entity_a: str, entity_b: str) -> Optional[str]:
    """
    Build a side-by-side natural-language comparison of two cell types using
    curated domain knowledge (no cosine search needed).
    Returns None if neither entity is found.
    """
    result_a = _lookup_biology(entity_a)
    result_b = _lookup_biology(entity_b)

    if not result_a and not result_b:
        return None

    def _fmt_entry(name: str, result: Optional[tuple]) -> str:
        if result is None:
            return f"**{name}** — not found in the curated biology knowledge base."
        ct_name, info = result
        lines = [f"**{ct_name}**"]
        markers = info.get("canonical_markers") or info.get("markers", [])
        novel   = info.get("novel_markers", [])
        hormone = info.get("hormone", "")
        loc     = info.get("location", "")
        pathway = info.get("pathway", "")
        desc    = info.get("description", "")
        if markers:
            lines.append(f"  Canonical markers : {', '.join(markers)}")
        if novel:
            lines.append(f"  Novel markers     : {', '.join(novel)}")
        if hormone and hormone.lower() != "none (uncommitted)":
            lines.append(f"  Hormone/product   : {hormone}")
        if loc:
            lines.append(f"  Location          : {loc}")
        if pathway:
            lines.append(f"  Key pathway       : {pathway}")
        if desc:
            lines.append(f"  {desc.strip()}")
        return "\n".join(lines)

    label_a = result_a[0] if result_a else entity_a
    label_b = result_b[0] if result_b else entity_b
    header  = f"Comparison: **{label_a}** vs **{label_b}**\n"
    return header + "\n\n".join([_fmt_entry(entity_a, result_a),
                                  _fmt_entry(entity_b, result_b)])


# ── Chat agent ─────────────────────────────────────────────────────────────────

class C2SChatbot:
    """
    Conversational wrapper around CellDatabaseAgent.

    Parameters
    ----------
    agent : CellDatabaseAgent
        Pre-initialised agent (loaded models + database).
    max_new_tokens : int
        Passed to the generation call.
    temperature, top_k, top_p : float / int
        Sampling parameters.

    Usage
    -----
        from chat import C2SChatbot
        chatbot = C2SChatbot(agent=agent, max_new_tokens=256)
        print(chatbot.chat("tell me about goblet cells"))
    """

    def __init__(
        self,
        agent: CellDatabaseAgent,
        max_new_tokens: int = 512,
        temperature: float  = 0.7,
        top_k: int          = 30,
        top_p: float        = 0.9,
    ):
        self.agent          = agent
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_k          = top_k
        self.top_p          = top_p
        self.history: list[tuple[str, str]] = []
        self.matcher        = CatalogueMatcher(agent.get_catalogue())

    # ── Public ─────────────────────────────────────────────────────────────────

    def parse_query(self, user_text: str) -> CellQuery:
        """Parse free-text → CellQuery."""
        return self.matcher.parse_query(user_text)

    def chat(self, user_message: str) -> str:
        """
        Process one user message and return the agent response.

        Pipeline
        --------
        0. Comparison shortcut — "compare A and B" / "A vs B":
             answered directly from curated biology knowledge (no DB search).
        1. Parse natural language → CellQuery  (metadata + cell-type filters)
        2. Gene detection: scan for a gene name → set target_gene for gene-specific retrieval
        3. "Tell me about X" / general explanation → ensure full info is returned
        4. Fallback: fuzzy match the full message as a cell-type name
        5. Delegate to CellDatabaseAgent.query() which runs:
             SQL candidate filter → build query vector → cosine similarity
             search → C2S NL generation → synthesised natural-language answer
        6. Store turn in history
        """
        # ── Step 0: Comparison query shortcut ────────────────────────────────
        entities = _extract_comparison_entities(user_message)
        if entities:
            entity_a, entity_b = entities
            print(f"  [parser] Comparison detected: '{entity_a}' vs '{entity_b}'")
            comp_response = _build_comparison_response(entity_a, entity_b)
            if comp_response:
                self.history.append((user_message, comp_response))
                return comp_response
            # Fall through to normal pipeline if neither entity is in the KB

        cell_query = self.matcher.parse_query(user_message)

        # ── Gene detection (runs for every query) ─────────────────────────────
        # Scan for a specific gene name anywhere in the message.  When found,
        # set target_gene so the agent retrieves cells that highly express it
        # rather than using a generic cell-type mean vector.
        # This fixes the case where "MUC2" and "goblet cells" returned identical
        # results (both resolved to the same cell-type mean embedding).
        target_gene = _detect_gene_in_query(user_message)
        if target_gene:
            print(f"  [parser] Gene detected: '{target_gene}'")
            cell_query = _dc_replace(cell_query, target_gene=target_gene)

        # ── "Tell me about X" / general explanation ───────────────────────────
        # Strip the filler prefix so the remaining text is treated as a cell-type
        # or gene name by the downstream matchers.
        _EXPLAIN_PREFIXES = (
            "tell me about ", "explain ", "describe ", "what is ", "what are ",
            "give me info on ", "give me information about ", "info on ",
        )
        _stripped = user_message.strip()
        for _pfx in _EXPLAIN_PREFIXES:
            if _stripped.lower().startswith(_pfx):
                _core = _stripped[len(_pfx):].strip()
                if _core:
                    print(f"  [parser] General-explanation query: '{_core}'")
                    _stripped = _core
                break

        # ── Fallback: full message (or stripped core) as fuzzy cell-type name ──
        if cell_query.query_mode == "empty":
            fuzzy_ct = self.matcher._resolve_cell_type(_stripped)
            if fuzzy_ct:
                print(f"  [parser] Fuzzy fallback: '{_stripped}' → cell_type='{fuzzy_ct}'")
                cell_query = CellQuery(cell_type=fuzzy_ct)

        # Also re-run gene detection on the stripped core if still empty
        if cell_query.query_mode == "empty" and _stripped != user_message:
            gene2 = _detect_gene_in_query(_stripped)
            if gene2:
                print(f"  [parser] Gene detected from stripped query: '{gene2}'")
                cell_query = _dc_replace(cell_query, target_gene=gene2)

        if cell_query.query_mode == "empty":
            catalogue = self.agent.get_catalogue()
            ct_preview  = ", ".join(catalogue.get("cell_types", [])[:8])
            reg_preview = ", ".join(catalogue.get("regions",    [])[:6])
            return (
                "I couldn't identify a specific cell type, gene, or metadata filter "
                "in your query.\n\n"
                f"Available cell types : {ct_preview}\n"
                f"Available regions    : {reg_preview}\n\n"
                "You can query by cell type, metadata, or marker gene "
                "(e.g. 'MUC2', 'APOA1', 'TPT1', 'GIP').\n"
                "Try: 'goblet cells in the colon', 'enterocytes', "
                "'L cells (GLP-1)', 'compare goblet cells and enterocytes', "
                "or 'tell me about EECs'."
            )

        print(f"  [parser] filters={cell_query.active_filters()}")
        response = self.agent.query(cell_query, user_question=user_message)
        self.history.append((user_message, response))
        return response

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def run_interactive(self):
        """Start a blocking interactive chat loop in the terminal."""
        print("\nGut Cell Database Chat Agent")
        print("=" * 40)
        print("Commands:  reset — clear history    quit — exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "reset":
                self.reset()
                print("  [chat] History cleared.")
                continue

            response = self.chat(user_input)
            print(f"\nAgent: {response}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gut Cell Database Chat Agent")
    p.add_argument("--db_path",            required=True,
                   help="Path to guts.db (built by build_database.py)")
    p.add_argument("--arrow_dir",          required=True,
                   help="Path to arrow_ds/ (saved by build_database.py)")
    p.add_argument("--nl_model",           required=True,
                   help="Path to C2S NL model (e.g. C2S-Scale-Pythia-1b-pt)")
    p.add_argument("--query",              default=None,
                   help="Single query string. Omit for interactive mode.")
    p.add_argument("--vllm_server_url",    default="http://localhost:8000",
                   help="URL of the running vLLM server")
    p.add_argument("--top_k_genes",        type=int,   default=200)
    p.add_argument("--n_cells_per_prompt", type=int,   default=5)
    p.add_argument("--n_neighbours",       type=int,   default=20)
    p.add_argument("--max_new_tokens",     type=int,   default=512)
    p.add_argument("--temperature",        type=float, default=0.7)
    p.add_argument("--top_k",             type=int,   default=30)
    p.add_argument("--top_p",             type=float, default=0.9)
    return p.parse_args()


def run(args):
    agent = CellDatabaseAgent(
        db_path            = args.db_path,
        arrow_dir          = args.arrow_dir,
        nl_model_path      = args.nl_model,
        top_k_genes        = args.top_k_genes,
        n_neighbours       = args.n_neighbours,
        n_cells_per_prompt = args.n_cells_per_prompt,
        vllm_server_url    = args.vllm_server_url,
    )

    print("\nDatabase catalogue:")
    for dim, vals in agent.get_catalogue().items():
        preview = ", ".join(str(v) for v in vals[:6])
        suffix  = "…" if len(vals) > 6 else ""
        print(f"  {dim:22s}: {preview}{suffix}")

    chatbot = C2SChatbot(
        agent          = agent,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
        top_p          = args.top_p,
    )

    if args.query:
        print(f"\nYou: {args.query}")
        print(f"\nAgent: {chatbot.chat(args.query)}")
        return

    chatbot.run_interactive()


if __name__ == "__main__":
    run(parse_args())