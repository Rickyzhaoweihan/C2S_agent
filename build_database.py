"""
build_database.py
-----------------
Offline pipeline — run once to populate the normalized SQLite database.

Exact obs column names from the h5ad (as confirmed by adata.obs.columns):
    orig.ident, nCount_RNA, nFeature_RNA, percent.mt,
    RNA_snn_res.0.1, seurat_clusters, cell_type, batch_condition,
    RNA_snn_res.1, RNA_snn_res.0.05, RNA_snn_res.0.2, RNA_snn_res.0.35,
    unintegrated_clusters, Age, Tissue, region, SampleType, tissue,
    RNA_snn_res.3.5, RNA_snn_res.2, gene, organism, assay, sex,
    n_genes, n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt

Usage
-----
    python build_database.py \\
        --h5ad_path   /path/to/guts_preprocessed_data.h5ad \\
        --embed_model /path/to/C2S-Pythia-410m-cell-type-prediction \\
        --db_path     ./cell_db/guts.db \\
        --arrow_dir   ./cell_db/arrow_ds \\
        --n_genes     200
"""

from __future__ import annotations

import os
import argparse
import logging
import random
from typing import Optional

import numpy as np
import anndata
import cell2sentence as cs
from cell2sentence.tasks import embed_cells

import database as db
from database import SEURAT_RES_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# ── Exact obs column aliases ───────────────────────────────────────────────────
# Maps internal key → list of raw column names to try in priority order.
# The first name that exists in adata.obs.columns wins.
# Priority matters where two columns carry the same concept (e.g. Tissue/tissue).

OBS_ALIASES: dict[str, list[str]] = {
    # Categorical → dimension tables
    "cell_type":             ["cell_type"],
    "tissue":                ["Tissue", "tissue"],          # capital takes priority
    "batch_condition":       ["batch_condition"],
    "sample_type":           ["SampleType", "sample_type"],
    "region":                ["region"],
    "orig_ident":            ["orig.ident"],
    "age":                   ["Age", "age"],
    "organism":              ["organism"],
    "assay":                 ["assay"],
    "sex":                   ["sex"],
    # Numeric QC → direct columns on cells
    "n_count_rna":           ["nCount_RNA"],
    "n_feature_rna":         ["nFeature_RNA"],
    "percent_mt":            ["percent.mt", "pct_counts_mt"],
    "gene":                  ["gene"],
    "n_genes":               ["n_genes"],
    "n_genes_by_counts":     ["n_genes_by_counts"],
    "total_counts":          ["total_counts"],
    "total_counts_mt":       ["total_counts_mt"],
    "pct_counts_mt":         ["pct_counts_mt", "percent.mt"],
    # Fixed cluster columns
    "seurat_clusters":       ["seurat_clusters"],
    "unintegrated_clusters": ["unintegrated_clusters"],
}

# Columns passed to C2S as label metadata (only include what exists in the obs)
C2S_LABEL_KEYS = [
    "cell_type", "tissue", "batch_condition", "organism", "sex", "region",
]


def _resolve(obs_df, aliases: list[str]) -> Optional[str]:
    """Return the first alias that exists as a column in obs_df, else None."""
    for a in aliases:
        if a in obs_df.columns:
            return a
    return None


def _get(obs_row, col: Optional[str]):
    """Safely retrieve a value from an obs row by resolved column name."""
    return obs_row[col] if col and col in obs_row.index else None


def build_row(barcode: str, obs_row, col_map: dict) -> dict:
    """Build the flat insert dict for one cell."""
    row = {
        "barcode":               barcode,
        # Categorical
        "cell_type":             _get(obs_row, col_map.get("cell_type")),
        "tissue":                _get(obs_row, col_map.get("tissue")),
        "batch_condition":       _get(obs_row, col_map.get("batch_condition")),
        "sample_type":           _get(obs_row, col_map.get("sample_type")),
        "region":                _get(obs_row, col_map.get("region")),
        "orig_ident":            _get(obs_row, col_map.get("orig_ident")),
        "age":                   _get(obs_row, col_map.get("age")),
        "organism":              _get(obs_row, col_map.get("organism")),
        "assay":                 _get(obs_row, col_map.get("assay")),
        "sex":                   _get(obs_row, col_map.get("sex")),
        # Numeric QC
        "n_count_rna":           _get(obs_row, col_map.get("n_count_rna")),
        "n_feature_rna":         _get(obs_row, col_map.get("n_feature_rna")),
        "percent_mt":            _get(obs_row, col_map.get("percent_mt")),
        "gene":                  _get(obs_row, col_map.get("gene")),
        "n_genes":               _get(obs_row, col_map.get("n_genes")),
        "n_genes_by_counts":     _get(obs_row, col_map.get("n_genes_by_counts")),
        "total_counts":          _get(obs_row, col_map.get("total_counts")),
        "total_counts_mt":       _get(obs_row, col_map.get("total_counts_mt")),
        "pct_counts_mt":         _get(obs_row, col_map.get("pct_counts_mt")),
        # Fixed clusters
        "seurat_clusters":       _get(obs_row, col_map.get("seurat_clusters")),
        "unintegrated_clusters": _get(obs_row, col_map.get("unintegrated_clusters")),
    }
    # RNA_snn_res.* columns
    for res_col in SEURAT_RES_COLS:
        row[res_col] = obs_row[res_col] if res_col in obs_row.index else None
    return row


def run(
    h5ad_path: str,
    embed_model_path: str,
    db_path: str,
    arrow_dir: str,
    n_genes: int = 200,
):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(arrow_dir, exist_ok=True)

    # ── 1. Load AnnData ──────────────────────────────────────────────────────
    log.info(f"Loading AnnData from {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path)
    log.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    log.info(f"obs columns: {list(adata.obs.columns)}")

    # ── 2. Resolve exact column names ────────────────────────────────────────
    col_map: dict[str, Optional[str]] = {
        key: _resolve(adata.obs, aliases)
        for key, aliases in OBS_ALIASES.items()
    }
    log.info("Column mapping resolved:")
    for k, v in col_map.items():
        status = f"→ {v}" if v else "NOT FOUND"
        log.info(f"  {k:25s} {status}")

    # Columns to pass to C2S as label metadata
    c2s_label_cols = [
        col_map[k] for k in C2S_LABEL_KEYS
        if col_map.get(k) is not None
    ]
    log.info(f"C2S label columns: {c2s_label_cols}")

    # ── 3. Initialise database + gene reference table ────────────────────────
    db.init_db(db_path)

    gene_records = []
    for gene_name in adata.var_names:
        ensembl_id = (
            str(adata.var.loc[gene_name, "ensembl_id"])
            if "ensembl_id" in adata.var.columns else ""
        )
        gene_records.append({
            "ensembl_id":       ensembl_id,
            "gene_name":        str(gene_name),
            "is_mitochondrial": int(str(gene_name).startswith("MT-")),
        })
    db.insert_genes(db_path, gene_records)

    # ── 4. Convert AnnData → CSData (Arrow format) ───────────────────────────
    log.info("Converting AnnData → Arrow dataset …")
    obs_c2s = adata.obs.copy()

    # C2S adata_to_arrow requires every label column to be a clean string —
    # any raw float NaN causes a PyArrow ArrowTypeError.
    # Cast ALL label columns to str and replace every NaN/None variant with "NA".
    NAN_STRINGS = {"nan", "none", "nat", "<na>", ""}
    for key in C2S_LABEL_KEYS:
        actual_col = col_map.get(key)
        if actual_col and actual_col in obs_c2s.columns:
            obs_c2s[actual_col] = (
                obs_c2s[actual_col]
                .astype(str)
                .apply(lambda v: "NA" if v.strip().lower() in NAN_STRINGS else v)
            )
            log.info(f"  Cleaned label column: {actual_col}")

    adata_c2s = adata.copy()
    adata_c2s.obs = obs_c2s

    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata_c2s,
        random_state=SEED,
        sentence_delimiter=" ",
        label_col_names=c2s_label_cols,
    )

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocabulary,
        save_dir=os.path.join(os.path.dirname(db_path), "csdata"),
        save_name="guts_csdata",
        dataset_backend="arrow",
    )
    arrow_ds.save_to_disk(arrow_dir)
    log.info(f"Arrow dataset saved → {arrow_dir}")

    # ── 5. Cell sentences ────────────────────────────────────────────────────
    log.info("Extracting cell sentence strings …")
    all_sentences: list[str] = csdata.get_sentence_strings()
    log.info(f"Got {len(all_sentences)} sentences.")

    # ── 6. Embed cells ───────────────────────────────────────────────────────
    log.info(f"Loading embedding model from {embed_model_path} …")
    csmodel = cs.CSModel(
        model_name_or_path=embed_model_path,
        save_dir=os.path.join(os.path.dirname(db_path), "csmodel_cache"),
        save_name="embedding_model",
    )
    log.info(f"Embedding {adata.n_obs} cells (top {n_genes} genes) …")
    embeddings = embed_cells(
        csdata=csdata, csmodel=csmodel, n_genes=n_genes
    ).astype(np.float32)
    log.info(f"Embeddings shape: {embeddings.shape}")

    # ── 7. Build row dicts ───────────────────────────────────────────────────
    barcodes = adata.obs_names.tolist()
    rows = [
        build_row(barcode, adata.obs.iloc[i], col_map)
        for i, barcode in enumerate(barcodes)
    ]

    # ── 8. Bulk insert into SQL DB ───────────────────────────────────────────
    db.insert_cells_batch(
        db_path=db_path,
        rows=rows,
        embeddings=embeddings,
        sentences=all_sentences,
        model_name=os.path.basename(embed_model_path),
        n_genes=n_genes,
    )

    # ── 9. Summary ───────────────────────────────────────────────────────────
    catalogue = db.get_catalogue(db_path)
    log.info("Build complete. Catalogue:")
    for dim, vals in catalogue.items():
        preview = [str(v) for v in vals[:6]]
        log.info(f"  {dim:22s}: {preview}{'…' if len(vals) > 6 else ''}")

    log.info(f"\nDatabase  : {db_path}")
    log.info(f"Arrow DS  : {arrow_dir}")
    log.info("Run chat.py to start the agent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad_path",   required=True)
    parser.add_argument("--embed_model", required=True,
                        help="C2S-Pythia-410m embedding model path")
    parser.add_argument("--db_path",     default="./cell_db/guts.db")
    parser.add_argument("--arrow_dir",   default="./cell_db/arrow_ds")
    parser.add_argument("--n_genes",     type=int, default=200)
    args = parser.parse_args()
    run(args.h5ad_path, args.embed_model, args.db_path, args.arrow_dir, args.n_genes)