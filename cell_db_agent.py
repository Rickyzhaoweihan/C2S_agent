"""
cell_db_agent.py
----------------
Core agent over the normalized SQLite cell database.

CellQuery fields
----------------
Categorical (substring match against dimension tables):
    cell_type, tissue, batch_condition, sample_type, region,
    orig_ident, age, organism, assay, sex

Seurat cluster (exact integer):
    seurat_cluster

Resolution-specific cluster:
    cluster_resolution  (e.g. 0.35)
    cluster_id          (e.g. 3)

Numeric QC bounds:
    percent_mt_max, n_count_rna_min, n_count_rna_max
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests
import cell2sentence as cs
from cell2sentence.prompt_formatter import C2SMultiCellPromptFormatter


class _NLModelWrapper:
    """
    HTTP client that calls a running vLLM server instead of loading the model
    locally.  Exposes the same generate_from_prompt interface so the rest of
    the agent code is unchanged.

    The vLLM server must be running before starting chat.py:
        sbatch vllm_server.sbatch
    Then note the node hostname from the job output and pass it via
    --vllm_server_url http://<node>:8000
    """

    def __init__(self, server_url: str, model_name: str):
        self.server_url  = server_url.rstrip("/")
        self.model_name  = model_name
        self.completions = f"{self.server_url}/v1/completions"

        # Verify the server is reachable at startup
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            log.info(f"vLLM server healthy at {self.server_url}")
        except Exception as e:
            raise RuntimeError(
                f"vLLM server not reachable at {self.server_url}: {e}\n"
                "Make sure the vLLM server sbatch job is running first."
            )

    def generate_from_prompt(self, model, prompt, max_num_tokens=512, **kwargs):
        """Send one prompt to the vLLM server and return only the new text."""
        payload = {
            "model":       self.model_name,
            "prompt":      prompt,
            "max_tokens":  max_num_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p":       kwargs.get("top_p", 0.9),
            "top_k":       kwargs.get("top_k", 30),
        }
        try:
            resp = requests.post(self.completions, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("vLLM server timed out — the model may still be loading.")
        return resp.json()["choices"][0]["text"].lstrip()

    def generate_from_prompt_batched(self, model, prompt_list, max_num_tokens=512, **kwargs):
        return [
            self.generate_from_prompt(model, p, max_num_tokens=max_num_tokens, **kwargs)
            for p in prompt_list
        ]

import database as db
from pathlib import Path

log = logging.getLogger(__name__)


# ── Query dataclass ────────────────────────────────────────────────────────────

@dataclass
class CellQuery:
    # Categorical
    cell_type:          Optional[str]   = None
    tissue:             Optional[str]   = None
    batch_condition:    Optional[str]   = None
    sample_type:        Optional[str]   = None
    region:             Optional[str]   = None
    orig_ident:         Optional[str]   = None
    age:                Optional[str]   = None
    organism:           Optional[str]   = None
    assay:              Optional[str]   = None
    sex:                Optional[str]   = None
    # Seurat fixed cluster
    seurat_cluster:     Optional[int]   = None
    # Resolution-specific cluster
    cluster_resolution: Optional[float] = None
    cluster_id:         Optional[int]   = None
    # Numeric QC bounds
    percent_mt_max:     Optional[float] = None
    n_count_rna_min:    Optional[float] = None
    n_count_rna_max:    Optional[float] = None
    # Gene-level query (find cells that highly express this specific gene)
    target_gene:        Optional[str]   = None

    @property
    def has_cell_type(self) -> bool:
        return bool(self.cell_type)

    @property
    def has_metadata(self) -> bool:
        return any([
            self.tissue, self.batch_condition, self.sample_type,
            self.region, self.orig_ident, self.age,
            self.organism, self.assay, self.sex,
            self.seurat_cluster is not None,
            self.cluster_resolution is not None,
            self.cluster_id is not None,
            self.percent_mt_max is not None,
            self.n_count_rna_min is not None,
            self.n_count_rna_max is not None,
        ])

    @property
    def query_mode(self) -> str:
        if self.has_cell_type and self.has_metadata:
            return "combined"
        if self.has_cell_type:
            return "by_cell_type"
        if self.has_metadata:
            return "by_metadata"
        if self.target_gene:
            return "by_gene"
        return "empty"

    def active_filters(self) -> dict:
        return {k: v for k, v in {
            "cell_type":          self.cell_type,
            "tissue":             self.tissue,
            "batch_condition":    self.batch_condition,
            "sample_type":        self.sample_type,
            "region":             self.region,
            "orig_ident":         self.orig_ident,
            "age":                self.age,
            "organism":           self.organism,
            "assay":              self.assay,
            "sex":                self.sex,
            "seurat_cluster":     self.seurat_cluster,
            "cluster_resolution": self.cluster_resolution,
            "cluster_id":         self.cluster_id,
            "percent_mt_max":     self.percent_mt_max,
            "n_count_rna_min":    self.n_count_rna_min,
            "n_count_rna_max":    self.n_count_rna_max,
            "target_gene":        self.target_gene,
        }.items() if v is not None}

    def metadata_kwargs(self) -> dict:
        """Kwargs forwarded directly to db.fetch_cells_by_filters()."""
        return {
            "tissue":             self.tissue,
            "batch_condition":    self.batch_condition,
            "sample_type":        self.sample_type,
            "region":             self.region,
            "orig_ident":         self.orig_ident,
            "age":                self.age,
            "organism":           self.organism,
            "assay":              self.assay,
            "sex":                self.sex,
            "seurat_cluster":     self.seurat_cluster,
            "cluster_resolution": self.cluster_resolution,
            "cluster_id":         self.cluster_id,
            "percent_mt_max":     self.percent_mt_max,
            "n_count_rna_min":    self.n_count_rna_min,
            "n_count_rna_max":    self.n_count_rna_max,
        }


# ── Agent ──────────────────────────────────────────────────────────────────────

class CellDatabaseAgent:
    """
    SQL-backed agent for gut cell biology queries.

    Parameters
    ----------
    db_path : str
        SQLite database built by build_database.py.
    arrow_dir : str
        Arrow dataset saved by build_database.py (for prompt formatting).
    nl_model_path : str
        Path to C2S NL generation model ().
    top_k_genes : int
        Genes per cell sentence for the NL prompt (default 200).
    n_neighbours : int
        Cells returned by cosine search (default 20).
    n_cells_per_prompt : int
        Cells grouped per multi-cell NL prompt (default 5).
    seed : int
    """

    def __init__(
        self,
        db_path: str,
        arrow_dir: str,
        nl_model_path: str,
        top_k_genes: int = 200,
        n_neighbours: int = 20,
        n_cells_per_prompt: int = 5,
        seed: int = 1234,
        vllm_server_url: str = "http://localhost:8000",
    ):
        self.db_path            = db_path
        self.top_k_genes        = top_k_genes
        self.n_neighbours       = n_neighbours
        self.n_cells_per_prompt = n_cells_per_prompt
        random.seed(seed)
        np.random.seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {self.device}")
        self._vllm_server_url = vllm_server_url

        # ── Optional: load pre-computed embeddings for fast in-memory cosine search ──
        # Looks for cell_embeddings.npy / cell_embedding_ids.npy next to db_path.
        self._precomputed_emb: Optional[np.ndarray] = None
        self._precomputed_ids: Optional[list] = None
        db_dir = Path(db_path).parent
        npy_emb = db_dir / "cell_embeddings.npy"
        npy_ids = db_dir / "cell_embedding_ids.npy"
        if npy_emb.is_file() and npy_ids.is_file():
            self._precomputed_emb = np.load(str(npy_emb))  # (N, D) float32
            self._precomputed_ids = np.load(str(npy_ids)).tolist()  # [cell_id, …]
            # Build reverse lookup: cell_id → row index in the matrix
            self._id_to_emb_idx: dict[int, int] = {
                cid: i for i, cid in enumerate(self._precomputed_ids)
            }
            log.info(
                f"Loaded pre-computed embeddings: {self._precomputed_emb.shape} "
                f"from {npy_emb}"
            )
        else:
            self._id_to_emb_idx = {}
            log.info("No pre-computed embeddings found — will query DB per request.")

        log.info(f"Loading Arrow dataset from {arrow_dir}")
        self.arrow_ds = load_from_disk(arrow_dir)

        # Build barcode → Arrow row-index lookup (cell_name is obs_names from AnnData)
        self._barcode_to_arrow_idx: dict[str, int] = {
            self.arrow_ds[i]["cell_name"]: i
            for i in range(len(self.arrow_ds))
        }
        log.info(f"  Arrow index built: {len(self._barcode_to_arrow_idx)} cells")

        self._load_nl_model(nl_model_path)

        self.prompt_formatter = C2SMultiCellPromptFormatter(
            task="natural_language_interpretation",
            top_k_genes=top_k_genes,
        )

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_catalogue(self) -> dict:
        return db.get_catalogue(self.db_path)

    def query(self, cell_query: CellQuery, user_question: str = "") -> str:
        """
        Two-path query pipeline:

        • Direct path  — the queried entity IS in the database:
            SQL fetch → aggregate metadata + gene sentences + domain knowledge
            → conversational answer.  No cosine search, no model call.

        • Similarity path — NOT in the database:
            Cosine search across all cells → characterise from top-k matches.
        """
        if cell_query.query_mode == "empty":
            return "Please specify at least a cell type, gene name, or one metadata filter."

        log.info(f"Query mode: {cell_query.query_mode} | {cell_query.active_filters()}")

        candidate_rows = self._fetch_candidate_rows(cell_query)

        if candidate_rows:
            log.info(f"Direct hit: {len(candidate_rows)} cells in DB")
            return self._direct_answer(candidate_rows, cell_query, user_question)

        log.info("No direct match — running cosine similarity search across full DB")
        return self._similarity_answer(cell_query, user_question)

    # ── Direct path ─────────────────────────────────────────────────────────────

    def _fetch_candidate_rows(self, cell_query: CellQuery) -> list[dict]:
        """SQL fetch for the query. Returns [] when nothing matches."""
        if cell_query.target_gene:
            rows = db.fetch_cells_by_gene(
                self.db_path, cell_query.target_gene,
                top_n_position=200, limit=50_000,
            )
            if not rows:
                return []
            if cell_query.cell_type:
                ct_lower = cell_query.cell_type.lower()
                filtered = [r for r in rows if (r.get("cell_type") or "").lower() == ct_lower]
                rows = filtered if filtered else rows
            if cell_query.has_metadata:
                meta_rows = db.fetch_cells_by_filters(
                    self.db_path, limit=50_000, **cell_query.metadata_kwargs()
                )
                meta_ids = {r["cell_id"] for r in meta_rows}
                filtered = [r for r in rows if r["cell_id"] in meta_ids]
                rows = filtered if filtered else rows
            return rows

        if cell_query.has_cell_type or cell_query.has_metadata:
            return db.fetch_cells_by_filters(
                self.db_path, limit=50_000,
                cell_type=cell_query.cell_type,
                **cell_query.metadata_kwargs()
            )
        return []

    def _direct_answer(self, candidate_rows: list[dict],
                       cell_query: CellQuery, user_question: str) -> str:
        """
        Conversational answer built from DB metadata, cell sentences, and
        curated domain knowledge.  No cosine search or model call.
        """
        from collections import Counter

        n_cells    = len(candidate_rows)
        cell_types = Counter(r["cell_type"] for r in candidate_rows if r.get("cell_type"))
        regions    = Counter(r["region"]    for r in candidate_rows if r.get("region"))
        ages_raw   = sorted(set(r["age"]   for r in candidate_rows if r.get("age")))

        dominant_ct     = cell_types.most_common(1)[0][0] if cell_types else None
        dominant_region = regions.most_common(1)[0][0]    if regions    else None

        # Gene frequency across a representative sample of cell sentences
        sample_ids = [r["cell_id"] for r in candidate_rows[:300]]
        sentences  = db.fetch_sentences_for_cells(self.db_path, sample_ids)
        gene_counts: Counter = Counter()
        for sent in sentences.values():
            gene_counts.update(sent.split()[:100])

        specific_genes = [
            g for g, _ in gene_counts.most_common(60)
            if not _is_housekeeping(g)
        ][:12]

        return _build_direct_response(
            n_cells=n_cells,
            cell_types=cell_types,
            dominant_ct=dominant_ct,
            regions=regions,
            ages_raw=ages_raw,
            specific_genes=specific_genes,
            sentences=sentences,
            cell_query=cell_query,
            user_question=user_question,
        )

    # ── Similarity path ──────────────────────────────────────────────────────────

    def _similarity_answer(self, cell_query: CellQuery, user_question: str) -> str:
        """
        Full-DB cosine search used when the queried entity is not directly
        in the database.  Returns a characterisation of the top-k most
        similar cells.
        """
        # Try to build a query vector from whatever anchor we have
        query_vec = self._build_query_vector(cell_query)

        # For genes not in the top-200 window, try widening the search
        if query_vec is None and cell_query.target_gene:
            source_rows = db.fetch_cells_by_gene(
                self.db_path, cell_query.target_gene,
                top_n_position=500, limit=5_000,
            )
            if source_rows:
                source_ids = [r["cell_id"] for r in source_rows]
                if self._precomputed_emb is not None and self._id_to_emb_idx:
                    idxs = [self._id_to_emb_idx[cid] for cid in source_ids
                            if cid in self._id_to_emb_idx]
                    if idxs:
                        v   = self._precomputed_emb[idxs].mean(axis=0).astype(np.float32)
                        n_v = np.linalg.norm(v)
                        query_vec = v / n_v if n_v > 0 else v

        if query_vec is None:
            return self._not_found_message(cell_query)

        # ── Cosine search across all cells ───────────────────────────────────
        all_rows = db.fetch_cells_by_filters(self.db_path, limit=100_000)
        if not all_rows:
            return "The database appears to be empty."

        all_ids = [r["cell_id"] for r in all_rows]

        if self._precomputed_emb is not None and self._id_to_emb_idx:
            cand_idxs   = [self._id_to_emb_idx[cid] for cid in all_ids
                           if cid in self._id_to_emb_idx]
            cand_emb    = self._precomputed_emb[cand_idxs]
            ordered_ids = [self._precomputed_ids[i] for i in cand_idxs]
        else:
            cand_emb, ordered_ids = db.fetch_embeddings_for_cells(self.db_path, all_ids)

        if cand_emb.shape[0] == 0:
            return self._not_found_message(cell_query)

        normed   = _l2_norm(cand_emb)
        sims     = normed @ query_vec
        k        = min(self.n_neighbours, len(sims))
        top_idx  = np.argsort(sims)[::-1][:k]
        top_ids  = [ordered_ids[i] for i in top_idx]
        best_sim = float(sims[top_idx[0]])

        id_to_row = {r["cell_id"]: r for r in all_rows}
        top_cells = [id_to_row[sid] for sid in top_ids if sid in id_to_row]
        sentences = db.fetch_sentences_for_cells(self.db_path, top_ids)

        log.info(
            f"Similarity search: top-{k} cells, best_sim={best_sim:.3f}"
        )

        return _build_similarity_response(
            cell_query=cell_query,
            user_question=user_question,
            top_cells=top_cells,
            sentences=sentences,
            best_sim=best_sim,
        )

    # ── Initialisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_tokenizer(model_path: str) -> AutoTokenizer:
        """
        Load the tokenizer for *model_path*, falling back to a sibling model
        directory if the primary tokenizer files are corrupted.

        Background: some Gemma-2 checkpoints ship a broken tokenizer.model
        (sentencepiece raises RuntimeError on load) and no tokenizer.json.
        AutoTokenizer.from_pretrained silently returns False in that case
        instead of raising, so we need to detect and work around it.
        """
        def _try_load(path: str):
            tok = AutoTokenizer.from_pretrained(path, padding_side="left", use_fast=False)
            if isinstance(tok, bool):
                raise RuntimeError(f"tokenizer.model at {path} is unreadable by sentencepiece")
            return tok

        # Primary attempt
        try:
            return _try_load(model_path)
        except Exception as primary_err:
            log.warning(f"Could not load tokenizer from {model_path}: {primary_err}")

        # Fallback: search sibling directories in the same parent folder
        parent = os.path.dirname(model_path)
        for entry in sorted(os.listdir(parent)):
            candidate = os.path.join(parent, entry)
            if candidate == model_path or not os.path.isdir(candidate):
                continue
            cfg_path = os.path.join(candidate, "tokenizer_config.json")
            if not os.path.isfile(cfg_path):
                continue
            try:
                with open(cfg_path) as f:
                    tc = json.load(f)
                if tc.get("tokenizer_class") != "GemmaTokenizer":
                    continue
                tok = _try_load(candidate)
                log.info(f"Using tokenizer from sibling model: {candidate}")
                return tok
            except Exception:
                continue

        raise RuntimeError(
            f"Could not load a valid GemmaTokenizer from {model_path} "
            "or any sibling model directory."
        )

    def _load_nl_model(self, nl_model_path: str):
        """
        Connect to the vLLM server instead of loading the model locally.
        nl_model_path is used as the model name in the API request —
        it must match what the vLLM server was started with.
        """
        server_url = getattr(self, "_vllm_server_url", "http://localhost:8000")
        log.info(f"Connecting to vLLM server at {server_url}")
        log.info(f"  Model: {nl_model_path}")
        self.csmodel  = _NLModelWrapper(server_url=server_url, model_name=nl_model_path)
        self.nl_model = None   # generation goes through the HTTP client
        log.info("vLLM client ready — no local GPU required for chat.py.")

    # ── Query vector ────────────────────────────────────────────────────────────

    def _build_query_vector(self, cell_query: CellQuery) -> Optional[np.ndarray]:
        if cell_query.target_gene:
            # Query vector = mean embedding of cells that highly express this gene
            source_rows = db.fetch_cells_by_gene(
                self.db_path, cell_query.target_gene, top_n_position=30, limit=2000
            )
            if cell_query.cell_type and source_rows:
                ct_lower = cell_query.cell_type.lower()
                source_rows = [
                    r for r in source_rows
                    if (r.get("cell_type") or "").lower() == ct_lower
                ]
            # Fall back to all cells of the cell type if gene-filtered set is empty
            if not source_rows and cell_query.cell_type:
                source_rows = db.fetch_cells_by_filters(
                    self.db_path, cell_type=cell_query.cell_type, limit=5000
                )
        elif cell_query.has_cell_type:
            source_rows = db.fetch_cells_by_filters(
                self.db_path, cell_type=cell_query.cell_type, limit=5000
            )
        else:
            source_rows = db.fetch_cells_by_filters(
                self.db_path, limit=5000, **cell_query.metadata_kwargs()
            )
        if not source_rows:
            return None

        source_ids = [r["cell_id"] for r in source_rows]

        if self._precomputed_emb is not None and self._id_to_emb_idx:
            idxs = [self._id_to_emb_idx[cid] for cid in source_ids if cid in self._id_to_emb_idx]
            embs = self._precomputed_emb[idxs] if idxs else np.empty((0,), dtype=np.float32)
        else:
            embs, _ = db.fetch_embeddings_for_cells(self.db_path, source_ids)

        if embs.shape[0] == 0:
            return None
        v = embs.mean(axis=0).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ── NL interpretation ───────────────────────────────────────────────────────

    def _interpret(self, top_ids: list[int], meta_by_id: dict, user_question: str = "",
                   target_gene: Optional[str] = None) -> str:
        n = self.n_cells_per_prompt

        # Convert SQL cell IDs → Arrow row indices via barcode lookup.
        # top_ids are cells.id (1-based SQL PKs); Arrow indices are 0-based positions.
        arrow_idxs: list[int] = []
        for sql_id in top_ids:
            row = meta_by_id.get(sql_id)
            if row:
                idx = self._barcode_to_arrow_idx.get(row["barcode"])
                if idx is not None:
                    arrow_idxs.append(idx)

        if not arrow_idxs:
            return "No cells could be mapped to the Arrow dataset for interpretation."

        # Use unique cells only — do NOT repeat indices.
        # Previously `arrow_idxs * N` caused the model to see n identical cells
        # which produced repetitive output.
        chosen = arrow_idxs[:n]
        first  = next(iter(meta_by_id.values())) if meta_by_id else {}

        # Fetch the cell sentences (top-ranked genes) for synthesis later.
        sentences = db.fetch_sentences_for_cells(self.db_path, top_ids)

        multi_ds = Dataset.from_dict({
            "multi_cell_groupings": [chosen],
            "batch_label":          [first.get("batch_condition", "unknown")],
            "tissue":               [first.get("tissue", "unknown")],
            "organism":             [first.get("organism", "Homo sapiens")],
            "abstract":             [""],
        })

        formatted = self.prompt_formatter.format_hf_ds(
            hf_ds=self.arrow_ds,
            multi_cell_indices_ds=multi_ds,
        )

        # ── Augment the prompt for Gemma-2 instruction following ──────────────
        # C2S-Scale-Gemma-2-2B has strong instruction-following capability.
        # We inject:
        #   1. A brief gut-dataset context prefix (so the model knows the tissue)
        #   2. The user's specific question just before "Abstract summary:"
        #      so the generation directly addresses what was asked.
        #
        # The "Abstract summary:" cue is preserved — it is the fine-tuning
        # trigger that keeps the model in abstract-generation mode rather than
        # reverting to gene-list output.
        prompt = formatted[0]["model_input"]

        # 1. Prepend dataset context so the model stays within gut biology
        gut_context = (
            "Context: These cells are from a human gut single-cell RNA dataset "
            "covering fetal and adult intestine (duodenum, jejunum, ileum, colon). "
            "Epithelial cell types include stem cells, TA cells, enterocytes, "
            "goblet cells, and enteroendocrine cells (EECs).\n\n"
        )
        prompt = gut_context + prompt

        # 2. Insert the user question immediately before "Abstract summary:"
        #    so the model's generation is targeted to that specific question.
        if user_question:
            prompt = prompt.replace(
                "\nAbstract summary:",
                f"\nSpecific question to address: {user_question}\nAbstract summary:"
            )

        # print("\n" + "=" * 70)
        # print("  NL MODEL INPUT")
        # print("=" * 70)
        # print(prompt)
        # print("=" * 70)
        # print(f"  Characters : {len(prompt)}")
        # print(f"  Words      : ~{len(prompt.split())}")
        # print(f"  User query : {user_question!r}")
        # print("=" * 70 + "\n")

        raw = self.csmodel.generate_from_prompt(
            model=self.nl_model,
            prompt=prompt,
            do_sample=True,
            max_num_tokens=512,
            temperature=0.7,
            top_k=30,
            top_p=0.9,
        ).strip()

        top_cells = [meta_by_id[sid] for sid in top_ids if sid in meta_by_id]
        return _synthesize_answer(raw, user_question, top_cells, sentences,
                                  target_gene=target_gene)

    # ── Fallback ────────────────────────────────────────────────────────────────

    def _not_found_message(self, cell_query: CellQuery) -> str:
        cat     = self.get_catalogue()
        queried = (cell_query.cell_type or cell_query.target_gene
                   or str(cell_query.active_filters()))
        ct_list = ", ".join(cat.get("cell_types", [])[:8])
        return (
            f"I couldn't find **{queried}** in this gut single-cell database, "
            "and there wasn't enough information to run a similarity search.\n\n"
            f"Available cell types: {ct_list}\n\n"
            "Try querying by cell type, gene name, or metadata filter — for example:\n"
            "  • 'tell me about goblet cells'\n"
            "  • 'MUC2 expression in the colon'\n"
            "  • 'compare enterocytes and EECs'"
        )


# ── Direct-answer response builder ───────────────────────────────────────────

def _build_direct_response(
    n_cells: int,
    cell_types,          # Counter
    dominant_ct: Optional[str],
    regions,             # Counter
    ages_raw: list,
    specific_genes: list,
    sentences: dict,
    cell_query: "CellQuery",
    user_question: str,
) -> str:
    """
    Conversational prose for a query that matched cells in the database.
    Detects intent from user_question and surfaces the most relevant info first.
    """
    from collections import Counter

    q_lower = user_question.lower() if user_question else ""
    _wants_markers  = any(w in q_lower for w in [
        "marker", "gene", "express", "signature", "which gene", "what gene"])
    _wants_function = any(w in q_lower for w in [
        "function", "role", "what do", "what does", "purpose",
        "secrete", "secretes", "produce", "hormone"])
    _wants_pathway  = any(w in q_lower for w in [
        "pathway", "signaling", "signalling", "mechanism",
        "differentiat", "how does", "how is", "promote", "regulat"])
    _wants_location = any(w in q_lower for w in [
        "where", "region", "location", "tissue", "found"])

    # ── Helper: age string ────────────────────────────────────────────────
    def _age_str(ages):
        nums = [a.replace("years", "").strip() for a in ages]
        if not nums:       return ""
        if len(nums) == 1: return f"a donor aged {nums[0]}"
        return f"donors aged {nums[0]}–{nums[-1]}"

    # ── Location string ───────────────────────────────────────────────────
    top_regions = [r for r, _ in regions.most_common(3)]
    if len(top_regions) == 0:   region_str = ""
    elif len(top_regions) == 1: region_str = top_regions[0]
    elif len(top_regions) == 2: region_str = f"{top_regions[0]} and {top_regions[1]}"
    else: region_str = f"{top_regions[0]}, {top_regions[1]}, and {top_regions[2]}"

    age_str    = _age_str(ages_raw)
    ct_display = _ct_display_name(dominant_ct) if dominant_ct else "cells"
    pct        = int(100 * cell_types[dominant_ct] / n_cells) if dominant_ct else 100
    pct_note   = f" ({pct}% of results)" if pct < 90 and len(cell_types) > 1 else ""

    paragraphs: list[str] = []

    # ── Opening paragraph ─────────────────────────────────────────────────
    target = cell_query.target_gene
    if target:
        # Compute median expression rank of the gene across the cell sentences
        ranks = []
        for sent in list(sentences.values())[:20]:
            genes_in_sent = sent.split()
            if target in genes_in_sent:
                ranks.append(genes_in_sent.index(target) + 1)
        rank_note = (
            f"a median expression rank of ~{int(sum(ranks) / len(ranks))} out of 200"
            if ranks else "present in the expression profiles"
        )

        gene_ctx = _get_gene_context_for_gene(target)
        p_open = (
            f"In the gut single-cell database, **{target}** is highly expressed "
            f"in a set of **{n_cells:,}** {ct_display}{pct_note}"
        )
        if region_str: p_open += f" from the **{region_str}**"
        if age_str:    p_open += f", collected from {age_str}"
        p_open += f". {target} shows {rank_note} in these cells"
        if gene_ctx:   p_open += f" — {gene_ctx.rstrip('.')}"
        p_open += "."
    else:
        p_open = (
            f"**{ct_display.capitalize()}** are well-represented in this database — "
            f"there are **{n_cells:,}** of them{pct_note}"
        )
        if region_str: p_open += f", primarily in the **{region_str}**"
        if age_str:    p_open += f", from {age_str}"
        p_open += "."
    paragraphs.append(p_open)

    # ── Gene-signature paragraph ──────────────────────────────────────────
    p_sig = ""
    if specific_genes:
        annotated = _annotate_genes(specific_genes[:8])
        gene_list = (
            ", ".join(annotated[:-1]) + f", and {annotated[-1]}"
            if len(annotated) > 1 else annotated[0]
        )
        if target:
            p_sig = (
                f"These cells also highly express {gene_list}, "
                f"providing the molecular context in which **{target}** is active."
            )
        else:
            p_sig = f"Their most frequently expressed genes are {gene_list}."

    # ── Biology / function / pathway paragraph ───────────────────────────
    # Always included — falls back to a gene-based description when the
    # curated knowledge base has no entry for the dominant cell type.
    p_bio = ""
    if dominant_ct:
        p_bio = _get_domain_context(dominant_ct, highlight_gene=target)
        if not p_bio:
            # Fallback: describe using the most specific genes we observed
            if specific_genes:
                top_annotated = _annotate_genes(specific_genes[:5])
                gene_fallback = ", ".join(top_annotated[:-1]) + f", and {top_annotated[-1]}" \
                    if len(top_annotated) > 1 else top_annotated[0]
                p_bio = (
                    f"{ct_display.capitalize()} are characterised by expression of "
                    f"{gene_fallback}. No additional curated description is available "
                    f"for this cell type in the current knowledge base."
                )
            else:
                p_bio = (
                    f"No additional curated description is available for "
                    f"**{ct_display}** in the current knowledge base."
                )

    # ── Assemble into 1–2 flowing prose paragraphs ───────────────────────
    # Para 1 : opening context + gene signature (or opening + biology for
    #          function/pathway queries where biology is the primary answer).
    # Para 2 : the remaining block (biology or gene signature).
    if _wants_pathway or _wants_function:
        para1_parts = [p for p in [p_open, p_bio] if p]
        para2_parts = [p_sig] if p_sig else []
    else:
        para1_parts = [p for p in [p_open, p_sig] if p]
        para2_parts = [p_bio] if p_bio else []

    para1 = " ".join(para1_parts)
    para2 = " ".join(para2_parts)
    return "\n\n".join(p for p in [para1, para2] if p)


# ── Similarity-answer response builder ───────────────────────────────────────

def _build_similarity_response(
    cell_query: "CellQuery",
    user_question: str,
    top_cells: list[dict],
    sentences: dict,
    best_sim: float,
) -> str:
    """
    Conversational prose for a similarity-search result (cell/gene not in DB).
    Clearly states what was NOT found, what WAS found, and characterises it.
    """
    from collections import Counter

    if not top_cells:
        queried = cell_query.cell_type or cell_query.target_gene or "that entity"
        return (
            f"I couldn't find **{queried}** directly in this database, "
            "and the similarity search returned no results either."
        )

    queried     = cell_query.cell_type or cell_query.target_gene or "the queried entity"
    cell_types  = Counter(c["cell_type"] for c in top_cells if c.get("cell_type"))
    regions     = Counter(c["region"]    for c in top_cells if c.get("region"))
    ages_raw    = sorted(set(c["age"]   for c in top_cells if c.get("age")))
    n_cells     = len(top_cells)

    dominant_ct = cell_types.most_common(1)[0][0] if cell_types else None
    dom_pct     = int(100 * cell_types[dominant_ct] / n_cells) if dominant_ct else 100
    ct_display  = _ct_display_name(dominant_ct) if dominant_ct else "unknown cells"

    top_regions = [r for r, _ in regions.most_common(2)]
    region_str  = " and ".join(top_regions) if top_regions else ""

    sim_label   = "strong" if best_sim > 0.90 else "moderate" if best_sim > 0.75 else "weak"

    # Gene signature of the matched cells
    gene_counts: Counter = Counter()
    for sent in list(sentences.values())[:20]:
        gene_counts.update(sent.split()[:100])
    specific_genes = [g for g, _ in gene_counts.most_common(60) if not _is_housekeeping(g)][:8]

    paragraphs: list[str] = []

    # ── Para 1: not found + what was found ───────────────────────────────
    p1 = (
        f"**{queried}** is not directly represented in this gut single-cell database. "
        f"However, a cosine similarity search across all cells found "
        f"**{ct_display}** as the closest match "
        f"({dom_pct}% of the top {n_cells} results; similarity {best_sim:.2f} — {sim_label})"
    )
    if region_str: p1 += f", primarily from the **{region_str}**"
    p1 += "."
    paragraphs.append(p1)

    # ── Para 2: gene signature of the matched cells ───────────────────────
    if specific_genes:
        annotated = _annotate_genes(specific_genes)
        gene_list = (
            ", ".join(annotated[:-1]) + f", and {annotated[-1]}"
            if len(annotated) > 1 else annotated[0]
        )
        paragraphs.append(
            f"The most similar cells in the database highly express {gene_list}. "
            "This expression profile is what drove the similarity match."
        )

    # ── Para 3: function and description of the matched cell type ────────
    # Always shown — falls back to a gene-based note when there is no
    # curated entry for the matched cell type.
    if dominant_ct:
        bio = _get_domain_context(dominant_ct)
        if bio:
            paragraphs.append(
                f"Here is what is known about **{ct_display}** — "
                f"the closest cell type in this database:\n\n{bio}"
            )
        else:
            # Fallback: characterise from the observed gene signature
            if specific_genes:
                top_ann = _annotate_genes(specific_genes[:5])
                gene_note = ", ".join(top_ann[:-1]) + f", and {top_ann[-1]}" \
                    if len(top_ann) > 1 else top_ann[0]
                paragraphs.append(
                    f"**{ct_display.capitalize()}** are characterised in this database "
                    f"by expression of {gene_note}. "
                    f"No further curated description is available for this cell type."
                )
            else:
                paragraphs.append(
                    f"No curated description is available for **{ct_display}** "
                    f"in the current knowledge base."
                )

    # ── Assemble into 2 flowing prose paragraphs ─────────────────────────
    # Para 1 : not-found note + gene signature of the matched cells
    # Para 2 : biological description of the matched cell type + caveat
    caveat = (
        f"Note that since **{queried}** was not found directly, this description "
        f"reflects the most transcriptomically similar cells in the database, "
        f"and the match may not fully capture the biology of {queried}."
    )

    # paragraphs[0] = p1 (not found + match), paragraphs[1] = p2 (gene sig),
    # paragraphs[2] = p3 (bio description)
    first_parts  = [p for p in paragraphs[:2] if p]   # match + gene sig
    second_parts = [p for p in paragraphs[2:] if p]   # bio description(s)
    second_parts.append(caveat)

    para1 = " ".join(first_parts)
    para2 = " ".join(second_parts)
    return "\n\n".join(p for p in [para1, para2] if p)


# ── Utility ──────────────────────────────────────────────────────────────────

# Abstract-style openers the C2S model commonly produces.
_ABSTRACT_OPENERS = (
    "This study ", "In this study, ", "In this study ", "This paper ",
    "Here, we ", "Here we ", "We report ", "We describe ", "We present ",
    "We show ", "We demonstrate ", "We find ", "We found ",
    "The present study ", "The current study ",
)


def _clean_c2s_output(raw: str) -> str:
    """
    Strip model artefacts, academic openers, and deduplicate sentences from
    the C2S model output.

    Artefacts removed:
    - Control tokens: <ctrl100>, <|endoftext|>, etc.
    - Trailing/leading whitespace and double periods.
    - Verbatim repeated sentences (common when the model sees identical cells).
    - Academic abstract openers.
    """
    import re as _re

    text = raw.strip()

    # ── 1. Strip control / special tokens (e.g. <ctrl100>, <|endoftext|>) ──
    text = _re.sub(r'<[^>]{1,30}>', '', text)
    text = _re.sub(r'\.{2,}', '.', text)   # collapse "..." to "."
    text = text.strip('. ')

    if not text:
        return ""

    # ── 2. Remove academic abstract openers ──────────────────────────────
    for opener in _ABSTRACT_OPENERS:
        if text.lower().startswith(opener.lower()):
            text = text[len(opener):]
            if text:
                text = text[0].upper() + text[1:]
            break

    # ── 3. Deduplicate sentences ──────────────────────────────────────────
    raw_sents = [s.strip() for s in text.replace(". ", ".\n").splitlines() if s.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for s in raw_sents:
        key = s.lower().rstrip(". ").strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(s)

    return " ".join(unique).strip('. ')


def _pluralise_cell_type(ct: str) -> str:
    """Convert a cell-type name to its natural plural form, e.g. 'goblet cell' → 'goblet cells'."""
    if ct.endswith("cells"):
        return ct
    if ct.endswith("cell"):
        return ct + "s"
    if ct.endswith("cyte"):
        return ct + "s"
    if ct.endswith("e"):
        return ct + "s"
    return ct + "s"


def _get_gene_context_for_gene(gene_name: str) -> str:
    """
    Return a concise biological note for a specific gene from biology_context.
    Includes a brief description snippet when available.
    """
    from biology_context import CELL_TYPE_MARKERS, EEC_SUBTYPES

    for ct, info in CELL_TYPE_MARKERS.items():
        ct_plural = _pluralise_cell_type(ct)
        if gene_name in info.get("canonical_markers", []):
            desc = info.get("description", "")
            note = f"{gene_name} is a **canonical marker** of {ct_plural}"
            if desc:
                # Take the first sentence of the description
                first_sent = desc.split(".")[0].strip()
                note += f": {first_sent.rstrip('.')}."
            else:
                note += "."
            return note
        if gene_name in info.get("novel_markers", []):
            desc = info.get("description", "")
            note = f"{gene_name} is a **novel marker** of {ct_plural} (identified via snATAC-seq)"
            if desc:
                first_sent = desc.split(".")[0].strip()
                note += f": {first_sent.rstrip('.')}."
            else:
                note += "."
            return note

    for ct, info in EEC_SUBTYPES.items():
        if gene_name in info.get("markers", []):
            hormone = info.get("hormone", "")
            desc = info.get("description", "")
            note = f"{gene_name} marks **{ct}** cells"
            if hormone and hormone.lower() != "none (uncommitted)":
                note += f", which secrete {hormone}"
            if desc:
                first_sent = desc.split(".")[0].strip()
                note += f": {first_sent.rstrip('.')}."
            else:
                note += "."
            return note

    return ""


# Maps lowercased DB cell_type values to the key used in biology_context.
# Required because the DB stores "EECs" but biology_context uses "enteroendocrine cell".
_CT_NORMALISE: dict[str, str] = {
    "eecs":           "enteroendocrine cell",
    "eec":            "enteroendocrine cell",
    "goblet cells":   "goblet cell",
    "enterocytes":    "enterocyte",
    "ta cells":       "TA cell",
    "ta cell":        "TA cell",
    "iscs":           "intestinal stem cell",
    "isc":            "intestinal stem cell",
}

# Natural plural display names for DB cell_type values.
_CT_LONG_DISPLAY: dict[str, str] = {
    "EECs":         "enteroendocrine cells (EECs)",
    "Goblet Cells": "goblet cells",
    "Enterocytes":  "enterocytes",
    "TA Cells":     "transit-amplifying cells (TA cells)",
    "Stem Cells":   "intestinal stem cells",
}


def _ct_display_name(ct: str) -> str:
    """Return a natural plural display name for a cell type string from the DB."""
    if ct in _CT_LONG_DISPLAY:
        return _CT_LONG_DISPLAY[ct]
    lc = ct.lower()
    return lc if lc.endswith("s") else lc + "s"


def _get_domain_context(cell_type: str, highlight_gene: Optional[str] = None) -> str:
    """
    Return a biological description for a cell type from biology_context.
    When *highlight_gene* is given, the text leads with that gene's specific role
    before listing the other markers, making gene-query responses more focused.
    """
    from biology_context import CELL_TYPE_MARKERS, EEC_SUBTYPES

    # Normalise known DB aliases (e.g. "EECs" → "enteroendocrine cell")
    ct_lower = _CT_NORMALISE.get(cell_type.lower(), cell_type.lower())
    combined = {**CELL_TYPE_MARKERS, **EEC_SUBTYPES}
    for ct_name, info in combined.items():
        if ct_name.lower() in ct_lower or ct_lower in ct_name.lower():
            desc       = info.get("description", "")
            markers    = info.get("canonical_markers") or info.get("markers", [])
            novel      = info.get("novel_markers", [])
            hormone    = info.get("hormone", "")
            parts: list[str] = []

            if highlight_gene and highlight_gene in markers:
                other = [m for m in markers if m != highlight_gene]
                parts.append(
                    f"**{highlight_gene}** is a canonical marker of {ct_name}s; "
                    f"co-markers include {', '.join(other)}."
                    if other else
                    f"**{highlight_gene}** is the defining canonical marker of {ct_name}s."
                )
            elif highlight_gene and highlight_gene in novel:
                parts.append(
                    f"**{highlight_gene}** is a novel {ct_name} marker "
                    f"(identified via snATAC-seq). "
                    f"Canonical markers: {', '.join(markers)}."
                )
            else:
                if markers:
                    parts.append(f"Canonical markers: {', '.join(markers)}.")
                if novel:
                    parts.append(f"Novel markers: {', '.join(novel)}.")

            if hormone and hormone.lower() != "none (uncommitted)":
                parts.append(f"Secreted hormone/product: {hormone}.")
            if desc:
                parts.append(desc.strip())
            return " ".join(parts)
    return ""


# Genes that are constitutively highly expressed in virtually all cells —
# they dominate raw counts but carry no cell-type-specific information.
_HOUSEKEEPING_PREFIXES = ("MT-", "RPL", "RPS", "SNHG", "H3F3")
_HOUSEKEEPING_GENES    = frozenset({
    # Structural / translation
    "MALAT1", "NEAT1", "B2M", "ACTB", "ACTG1", "TUBA1B", "TUBB",
    "EEF1A1", "EEF1B2", "EEF1G", "EEF2", "GAPDH",
    "HSP90AB1", "HSP90AA1", "HSPA8", "RPLP0", "RPLP1", "RPLP2",
    # Iron / redox ubiquitous
    "FTH1", "FTH2", "FTL",
    # Ubiquitin / small proteins
    "UBB", "UBC", "UBA52", "PTMA", "TMSB10", "TMSB4X",
    # Universally expressed in gut epithelia
    "TPT1", "RACK1", "FAU", "NACA", "SSR4", "SERF2",
    "S100A6", "S100A10", "S100A11", "S100A14",
    "PCSK1N", "PHGR1",   # broadly expressed in gut
    "EPCAM",              # pan-epithelial
    # Mitochondrial / OXPHOS (already caught by MT- prefix but explicit list helps)
    "COX4I1", "COX5B", "COX6A1", "COX6B1", "COX6C",
    "COX7A2", "COX7B", "COX7C",
})


def _is_housekeeping(gene: str) -> bool:
    return gene in _HOUSEKEEPING_GENES or any(
        gene.startswith(p) for p in _HOUSEKEEPING_PREFIXES
    )


def _annotate_genes(genes: list[str]) -> list[str]:
    """
    Append a short function note to genes that appear in biology_context.
    e.g. "MUC2 (goblet cell marker)" or "GIP (K cell — GIP hormone)".
    Returns list of annotated strings, one per gene.
    """
    from biology_context import CELL_TYPE_MARKERS, EEC_SUBTYPES, MARKER_TO_CELL_TYPE

    # Build a flat gene → label map from the domain knowledge
    gene_label: dict[str, str] = {}
    for ct, info in CELL_TYPE_MARKERS.items():
        for g in info.get("canonical_markers", []):
            gene_label[g] = f"{ct} marker"
        for g in info.get("novel_markers", []):
            gene_label[g] = f"{ct} novel marker"
    for ct, info in EEC_SUBTYPES.items():
        hormone = info.get("hormone", "")
        # Use the short hormone name (before the parenthetical) to keep labels concise
        if hormone and hormone.lower() != "none (uncommitted)":
            short_h = hormone.split("(")[0].strip()
            label_base = f"{ct} — {short_h}"
        else:
            label_base = ct
        for g in info.get("markers", []):
            gene_label[g] = label_base
    # MARKER_TO_CELL_TYPE as fallback
    for g, ct in MARKER_TO_CELL_TYPE.items():
        if g not in gene_label:
            gene_label[g] = f"{ct} marker"

    annotated = []
    for g in genes:
        label = gene_label.get(g)
        annotated.append(f"{g} ({label})" if label else g)
    return annotated


def _synthesize_answer(
    c2s_raw: str,
    user_question: str,
    top_cells: list[dict],
    sentences: dict[int, str],
    target_gene: Optional[str] = None,
) -> str:
    """
    Build a flowing natural-language answer from the cosine-similarity retrieved
    cells.  The response consists of up to four prose paragraphs:

    1. Introduction — what was retrieved, where, from whom
    2. Expression signature — key marker genes described inline
    3. Biological function — domain-knowledge paragraph
    4. C2S narrative — cleaned model output (when gut-relevant)
    """
    from collections import Counter
    from biology_context import CELL_TYPE_MARKERS, EEC_SUBTYPES, MARKER_TO_CELL_TYPE

    if not top_cells:
        return "No cells found matching the query."

    # ── 1. Aggregate cell metadata ────────────────────────────────────────────
    cell_types = Counter(c["cell_type"] for c in top_cells if c.get("cell_type"))
    regions    = Counter(c["region"]    for c in top_cells if c.get("region"))
    tissues    = Counter(c["tissue"]    for c in top_cells if c.get("tissue"))
    ages_raw   = sorted(set(c["age"] for c in top_cells if c.get("age")))

    dominant_ct     = cell_types.most_common(1)[0][0] if cell_types else None
    dominant_region = regions.most_common(1)[0][0]    if regions    else None
    dominant_tissue = tissues.most_common(1)[0][0]    if tissues    else None
    n_cells         = len(top_cells)

    # Location string
    loc_parts: list[str] = []
    if dominant_region:
        loc_parts.append(dominant_region)
    if dominant_tissue and dominant_tissue != dominant_region:
        loc_parts.append(dominant_tissue)
    loc_str = " / ".join(loc_parts) if loc_parts else ""

    # Age string — ages_raw are strings like "53" or "53 years"
    def _age_str(age_list: list) -> str:
        if not age_list:
            return ""
        # Strip trailing "years" for range display
        nums = [a.replace("years", "").strip() for a in age_list]
        if len(nums) == 1:
            return f"a donor aged {nums[0]}"
        if len(nums) == 2:
            return f"donors aged {nums[0]} and {nums[1]}"
        return f"donors aged {nums[0]}–{nums[-1]}"

    age_str = _age_str(ages_raw)

    # ── 2. Extract cell-type-specific genes from cell sentences ───────────────
    gene_counts: Counter = Counter()
    for sent in list(sentences.values())[:15]:
        gene_counts.update(sent.split()[:100])

    specific_genes = [
        g for g, _ in gene_counts.most_common(50)
        if not _is_housekeeping(g)
    ][:12]
    display_genes = specific_genes if specific_genes else [
        g for g, _ in gene_counts.most_common(12)
    ]

    # ── 3. Build an inline gene → annotation map ──────────────────────────────
    gene_label: dict[str, str] = {}
    for ct, info in CELL_TYPE_MARKERS.items():
        for g in info.get("canonical_markers", []):
            gene_label[g] = f"canonical {ct} marker"
        for g in info.get("novel_markers", []):
            gene_label[g] = f"novel {ct} marker"
    for ct, info in EEC_SUBTYPES.items():
        hormone = info.get("hormone", "")
        short_h = hormone.split("(")[0].strip() if (
            hormone and hormone.lower() != "none (uncommitted)"
        ) else ""
        for g in info.get("markers", []):
            label = f"{ct} marker"
            if short_h:
                label += f", secretes {short_h}"
            gene_label[g] = label
    for g, ct in MARKER_TO_CELL_TYPE.items():
        if g not in gene_label:
            gene_label[g] = f"{ct} marker"

    def _annotate_inline(gene: str) -> str:
        label = gene_label.get(gene)
        return f"**{gene}** ({label})" if label else f"**{gene}**"

    def _gene_list_prose(genes: list[str], max_n: int = 8) -> str:
        """Format up to max_n genes as inline annotated prose list."""
        items = [_annotate_inline(g) for g in genes[:max_n]]
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    # ── 4. Detect user query intent ───────────────────────────────────────────
    # user_question is used here to decide which paragraphs to surface first
    # and to suppress paragraphs that are irrelevant to what was asked.
    q_lower = user_question.lower() if user_question else ""
    _wants_markers  = any(w in q_lower for w in [
        "marker", "markers", "express", "expression", "gene", "genes", "signature",
        "which gene", "what gene",
    ])
    _wants_function = any(w in q_lower for w in [
        "function", "role", "what do", "what does", "purpose",
        "secrete", "secretes", "produce", "produces", "hormone",
    ])
    _wants_pathway  = any(w in q_lower for w in [
        "pathway", "signaling", "signalling", "mechanism",
        "differentiat", "how does", "how is", "promote", "regulat",
    ])
    _wants_location = any(w in q_lower for w in [
        "where", "region", "location", "tissue", "locali", "found",
    ])
    # No specific intent detected → return full response
    _general = not any([_wants_markers, _wants_function, _wants_pathway, _wants_location])

    # ── 5. Build individual named paragraphs ──────────────────────────────────
    p_intro     = ""
    p_signature = ""
    p_biology   = ""
    p_c2s       = ""

    # — Intro paragraph —
    ct_name = _ct_display_name(dominant_ct) if dominant_ct else "cells"
    pct = int(100 * cell_types[dominant_ct] / n_cells) if dominant_ct else 100

    if target_gene:
        # Compute expression rank of the target gene across retrieved cells
        ranks = []
        for sent in list(sentences.values())[:20]:
            genes_in_sent = sent.split()
            if target_gene in genes_in_sent:
                ranks.append(genes_in_sent.index(target_gene) + 1)
        rank_note = (
            f"median expression rank ~{int(sum(ranks)/len(ranks))} out of 200"
            if ranks else "expressed"
        )

        gene_ctx = _get_gene_context_for_gene(target_gene)

        if pct >= 90 or not dominant_ct:
            intro = (
                f"In the gut single-cell database, **{target_gene}** is highly "
                f"expressed in a set of {n_cells} {ct_name}"
            )
        else:
            intro = (
                f"In the gut single-cell database, **{target_gene}** is highly "
                f"expressed across {n_cells} cells; {pct}% of these are {ct_name}"
            )
        if loc_str:
            intro += f" from the {loc_str}"
        if age_str:
            intro += f", collected from {age_str}"
        intro += f". {target_gene} shows a {rank_note} in these cells"
        if gene_ctx:
            # gene_ctx already ends with "."
            intro += f" — {gene_ctx.rstrip('.')}"
        intro += "."
        p_intro = intro

    else:
        # Cell-type or metadata query
        pct_str = f" ({pct}%)" if pct < 95 else ""
        intro = f"The {n_cells} cells retrieved from this database are {ct_name}{pct_str}"
        if loc_str:
            intro += f", sampled from the {loc_str}"
        if age_str:
            intro += f" across {age_str}"
        intro += "."
        p_intro = intro

    # — Expression-signature paragraph —
    if display_genes:
        gene_prose = _gene_list_prose(display_genes)
        if target_gene:
            p_signature = (
                f"These cells co-express {gene_prose}, "
                f"reflecting the molecular context in which **{target_gene}** "
                f"is active."
            )
        else:
            p_signature = (
                f"Their characteristic expression signature includes "
                f"{gene_prose}."
            )

    # — Biology / pathway paragraph (from curated domain knowledge) —
    if dominant_ct:
        bio_ctx = _get_domain_context(dominant_ct, highlight_gene=target_gene)
        if bio_ctx:
            p_biology = bio_ctx

    # ── 6. Assemble paragraphs ordered by user intent ─────────────────────────
    # The C2S model abstract is intentionally excluded: it frequently generates
    # off-topic or generic scientific text (e.g. unrelated studies) that does
    # not add value over the curated biology paragraphs above.
    #
    # Questions about pathways / function → biology paragraph first
    # Questions about markers / genes    → expression signature first
    # Questions about location / region  → intro + biology context
    # General / "tell me about"          → intro + signature + biology
    # ── Assemble into 1–2 flowing prose paragraphs ───────────────────────
    # Para 1 : intro + expression signature (or intro + biology for
    #          function/pathway/location queries).
    # Para 2 : the remaining block.
    if _wants_pathway or _wants_function:
        para1_parts = [p for p in [p_intro, p_biology] if p]
        para2_parts = [p_signature] if p_signature else []
    elif _wants_location:
        para1_parts = [p for p in [p_intro, p_biology] if p]
        para2_parts = []
    else:
        para1_parts = [p for p in [p_intro, p_signature] if p]
        para2_parts = [p_biology] if p_biology else []

    para1 = " ".join(para1_parts)
    para2 = " ".join(para2_parts)
    return "\n\n".join(p for p in [para1, para2] if p)


def _l2_norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1.0, n)