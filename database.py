"""
database.py
-----------
Normalized SQLite schema and all I/O operations for the gut cell database.

Schema overview
---------------
Dimension tables (one row per unique string value, referenced by FK):
    cell_types, tissues, batches, sample_types, regions,
    orig_idents, ages, organisms, assays, sexes

Fact table:
    cells — one row per barcode; FK into every dimension table;
            all numeric QC columns stored directly

Repeating-group table (3NF — one row per resolution per cell):
    cell_cluster_assignments (cell_id, resolution, cluster_id)

Derived / BLOB tables:
    cell_embeddings  — float32 C2S embedding as raw bytes
    cell_sentences   — space-delimited gene sentence string

Reference table:
    genes — ensembl_id, gene_name, is_mitochondrial

Obs column → DB mapping (exact column names from the h5ad)
-----------------------------------------------------------
Categorical (→ dimension FKs):
    orig.ident          → orig_idents          / cells.orig_ident_id
    cell_type           → cell_types           / cells.cell_type_id
    batch_condition     → batches              / cells.batch_id
    Age                 → ages                 / cells.age_id
    Tissue / tissue     → tissues              / cells.tissue_id  (merged)
    region              → regions              / cells.region_id
    SampleType          → sample_types         / cells.sample_type_id
    organism            → organisms            / cells.organism_id
    assay               → assays               / cells.assay_id
    sex                 → sexes                / cells.sex_id

Numeric QC (→ direct columns on cells):
    nCount_RNA          → cells.n_count_rna
    nFeature_RNA        → cells.n_feature_rna
    percent.mt          → cells.percent_mt
    gene                → cells.gene
    n_genes             → cells.n_genes
    n_genes_by_counts   → cells.n_genes_by_counts
    total_counts        → cells.total_counts
    total_counts_mt     → cells.total_counts_mt
    pct_counts_mt       → cells.pct_counts_mt

Fixed cluster columns:
    seurat_clusters        → cells.seurat_clusters
    unintegrated_clusters  → cells.unintegrated_clusters

Repeating group:
    RNA_snn_res.0.05 / 0.1 / 0.2 / 0.35 / 1 / 2 / 3.5
                        → cell_cluster_assignments(cell_id, resolution, cluster_id)
"""

from __future__ import annotations

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SEURAT_RES_COLS = [
    "RNA_snn_res.0.05",
    "RNA_snn_res.0.1",
    "RNA_snn_res.0.2",
    "RNA_snn_res.0.35",
    "RNA_snn_res.1",
    "RNA_snn_res.2",
    "RNA_snn_res.3.5",
]

# ── DDL ────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Dimension / lookup tables ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cell_types (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tissues (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS batches (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sample_types (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS regions (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS orig_idents (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS ages (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS organisms (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS assays (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sexes (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE
);

-- ── Reference table ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS genes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ensembl_id       TEXT,
    gene_name        TEXT NOT NULL UNIQUE,
    is_mitochondrial INTEGER NOT NULL DEFAULT 0
);

-- ── Fact table ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cells (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    barcode               TEXT    NOT NULL UNIQUE,

    -- Dimension FKs
    cell_type_id          INTEGER REFERENCES cell_types(id),
    tissue_id             INTEGER REFERENCES tissues(id),
    batch_id              INTEGER REFERENCES batches(id),
    sample_type_id        INTEGER REFERENCES sample_types(id),
    region_id             INTEGER REFERENCES regions(id),
    orig_ident_id         INTEGER REFERENCES orig_idents(id),
    age_id                INTEGER REFERENCES ages(id),
    organism_id           INTEGER REFERENCES organisms(id),
    assay_id              INTEGER REFERENCES assays(id),
    sex_id                INTEGER REFERENCES sexes(id),

    -- Numeric QC (single value per cell — not categorical, not repeated)
    n_count_rna           REAL,
    n_feature_rna         INTEGER,
    percent_mt            REAL,
    gene                  INTEGER,
    n_genes               INTEGER,
    n_genes_by_counts     INTEGER,
    total_counts          REAL,
    total_counts_mt       REAL,
    pct_counts_mt         REAL,

    -- Fixed cluster assignments
    seurat_clusters       INTEGER,
    unintegrated_clusters INTEGER
);

CREATE INDEX IF NOT EXISTS idx_cells_cell_type   ON cells(cell_type_id);
CREATE INDEX IF NOT EXISTS idx_cells_tissue       ON cells(tissue_id);
CREATE INDEX IF NOT EXISTS idx_cells_batch        ON cells(batch_id);
CREATE INDEX IF NOT EXISTS idx_cells_sample_type  ON cells(sample_type_id);
CREATE INDEX IF NOT EXISTS idx_cells_region       ON cells(region_id);
CREATE INDEX IF NOT EXISTS idx_cells_orig_ident   ON cells(orig_ident_id);
CREATE INDEX IF NOT EXISTS idx_cells_age          ON cells(age_id);
CREATE INDEX IF NOT EXISTS idx_cells_organism     ON cells(organism_id);
CREATE INDEX IF NOT EXISTS idx_cells_assay        ON cells(assay_id);
CREATE INDEX IF NOT EXISTS idx_cells_sex          ON cells(sex_id);
CREATE INDEX IF NOT EXISTS idx_cells_seurat       ON cells(seurat_clusters);

-- ── Repeating-group table: RNA_snn_res.* ────────────────────────────────────

CREATE TABLE IF NOT EXISTS cell_cluster_assignments (
    cell_id    INTEGER NOT NULL REFERENCES cells(id),
    resolution REAL    NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (cell_id, resolution)
);

CREATE INDEX IF NOT EXISTS idx_cca_res_cluster
    ON cell_cluster_assignments(resolution, cluster_id);

-- ── Derived / BLOB tables ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cell_embeddings (
    cell_id       INTEGER PRIMARY KEY REFERENCES cells(id),
    embedding     BLOB    NOT NULL,
    embedding_dim INTEGER NOT NULL,
    model_name    TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cell_sentences (
    cell_id   INTEGER PRIMARY KEY REFERENCES cells(id),
    sentence  TEXT    NOT NULL,
    n_genes   INTEGER NOT NULL DEFAULT 200,
    delimiter TEXT    NOT NULL DEFAULT ' '
);
"""

# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    log.info(f"Database initialised: {db_path}")


# ── Dimension upsert ───────────────────────────────────────────────────────────

def _upsert_dim(conn: sqlite3.Connection, table: str, name) -> Optional[int]:
    """Insert name into a dimension table if absent; return its id.
    Returns None for blank / NaN values."""
    if name is None:
        return None
    s = str(name).strip()
    if s.lower() in ("", "nan", "none", "na", "unknown"):
        return None
    conn.execute(f"INSERT OR IGNORE INTO {table}(name) VALUES (?)", (s,))
    return conn.execute(f"SELECT id FROM {table} WHERE name=?", (s,)).fetchone()["id"]


# ── Gene insert ────────────────────────────────────────────────────────────────

def insert_genes(db_path: str, gene_records: list[dict]) -> None:
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO genes(ensembl_id, gene_name, is_mitochondrial)
               VALUES(:ensembl_id, :gene_name, :is_mitochondrial)""",
            gene_records,
        )
    log.info(f"Inserted {len(gene_records)} gene records.")


# ── Bulk cell insert ───────────────────────────────────────────────────────────

def insert_cells_batch(
    db_path: str,
    rows: list[dict],
    embeddings: np.ndarray,
    sentences: list[str],
    model_name: str = "",
    n_genes: int = 200,
    batch_size: int = 500,
) -> None:
    """
    Insert cells, cluster assignments, embeddings, and sentences.

    Each row dict keys (exact mapping from build_database.py):
        barcode, cell_type, tissue, batch_condition, sample_type, region,
        orig_ident, age, organism, assay, sex,
        n_count_rna, n_feature_rna, percent_mt, gene,
        n_genes, n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt,
        seurat_clusters, unintegrated_clusters,
        RNA_snn_res.0.05, .0.1, .0.2, .0.35, .1, .2, .3.5
    """
    assert len(rows) == len(embeddings) == len(sentences)
    total = len(rows)
    log.info(f"Inserting {total} cells …")

    for start in range(0, total, batch_size):
        chunk_rows = rows[start: start + batch_size]
        chunk_emb  = embeddings[start: start + batch_size]
        chunk_sent = sentences[start: start + batch_size]

        with get_connection(db_path) as conn:
            for i, row in enumerate(chunk_rows):

                # Upsert all dimension FKs
                ct_id  = _upsert_dim(conn, "cell_types",   row.get("cell_type"))
                ti_id  = _upsert_dim(conn, "tissues",      row.get("tissue"))
                bt_id  = _upsert_dim(conn, "batches",      row.get("batch_condition"))
                st_id  = _upsert_dim(conn, "sample_types", row.get("sample_type"))
                rg_id  = _upsert_dim(conn, "regions",      row.get("region"))
                oi_id  = _upsert_dim(conn, "orig_idents",  row.get("orig_ident"))
                ag_id  = _upsert_dim(conn, "ages",         row.get("age"))
                org_id = _upsert_dim(conn, "organisms",    row.get("organism"))
                as_id  = _upsert_dim(conn, "assays",       row.get("assay"))
                sx_id  = _upsert_dim(conn, "sexes",        row.get("sex"))

                cur = conn.execute(
                    """INSERT OR IGNORE INTO cells (
                           barcode,
                           cell_type_id, tissue_id, batch_id, sample_type_id,
                           region_id, orig_ident_id, age_id,
                           organism_id, assay_id, sex_id,
                           n_count_rna, n_feature_rna, percent_mt, gene,
                           n_genes, n_genes_by_counts,
                           total_counts, total_counts_mt, pct_counts_mt,
                           seurat_clusters, unintegrated_clusters
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row["barcode"],
                        ct_id, ti_id, bt_id, st_id, rg_id, oi_id, ag_id,
                        org_id, as_id, sx_id,
                        _f(row.get("n_count_rna")),
                        _i(row.get("n_feature_rna")),
                        _f(row.get("percent_mt")),
                        _i(row.get("gene")),
                        _i(row.get("n_genes")),
                        _i(row.get("n_genes_by_counts")),
                        _f(row.get("total_counts")),
                        _f(row.get("total_counts_mt")),
                        _f(row.get("pct_counts_mt")),
                        _i(row.get("seurat_clusters")),
                        _i(row.get("unintegrated_clusters")),
                    ),
                )
                cell_id = cur.lastrowid or conn.execute(
                    "SELECT id FROM cells WHERE barcode=?", (row["barcode"],)
                ).fetchone()["id"]

                # Resolution cluster assignments
                for res_col in SEURAT_RES_COLS:
                    val = row.get(res_col)
                    if val is not None and str(val).strip() not in ("", "nan", "None"):
                        resolution = float(res_col.replace("RNA_snn_res.", ""))
                        conn.execute(
                            """INSERT OR REPLACE INTO cell_cluster_assignments
                               (cell_id, resolution, cluster_id) VALUES (?,?,?)""",
                            (cell_id, resolution, int(float(val))),
                        )

                # Embedding
                emb = chunk_emb[i].astype(np.float32)
                conn.execute(
                    """INSERT OR REPLACE INTO cell_embeddings
                       (cell_id, embedding, embedding_dim, model_name)
                       VALUES (?,?,?,?)""",
                    (cell_id, emb.tobytes(), emb.shape[0], model_name),
                )

                # Sentence
                conn.execute(
                    """INSERT OR REPLACE INTO cell_sentences
                       (cell_id, sentence, n_genes, delimiter)
                       VALUES (?,?,?,?)""",
                    (cell_id, chunk_sent[i], n_genes, " "),
                )

        log.info(f"  … {min(start + batch_size, total)}/{total} committed")

    log.info("All cells inserted.")


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_catalogue(db_path: str) -> dict:
    """Return all unique values per filterable dimension + available resolutions."""
    with get_connection(db_path) as conn:
        def _names(table):
            return [r["name"] for r in
                    conn.execute(f"SELECT name FROM {table} ORDER BY name")]

        resolutions = [r[0] for r in conn.execute(
            "SELECT DISTINCT resolution FROM cell_cluster_assignments "
            "ORDER BY resolution")]

        return {
            "cell_types":          _names("cell_types"),
            "tissues":             _names("tissues"),
            "batches":             _names("batches"),
            "sample_types":        _names("sample_types"),
            "regions":             _names("regions"),
            "orig_idents":         _names("orig_idents"),
            "ages":                _names("ages"),
            "organisms":           _names("organisms"),
            "assays":              _names("assays"),
            "sexes":               _names("sexes"),
            "cluster_resolutions": resolutions,
        }


def fetch_cells_by_filters(
    db_path: str,
    cell_type:          Optional[str]   = None,
    tissue:             Optional[str]   = None,
    batch_condition:    Optional[str]   = None,
    sample_type:        Optional[str]   = None,
    region:             Optional[str]   = None,
    orig_ident:         Optional[str]   = None,
    age:                Optional[str]   = None,
    organism:           Optional[str]   = None,
    assay:              Optional[str]   = None,
    sex:                Optional[str]   = None,
    seurat_cluster:     Optional[int]   = None,
    cluster_resolution: Optional[float] = None,
    cluster_id:         Optional[int]   = None,
    percent_mt_max:     Optional[float] = None,
    n_count_rna_min:    Optional[float] = None,
    n_count_rna_max:    Optional[float] = None,
    limit:              int             = 10_000,
) -> list[dict]:
    """
    Return cell rows matching ALL provided filters (AND logic).
    Categorical filters use case-insensitive LIKE substring matching.
    Numeric filters are range bounds applied directly on cells columns.
    Resolution/cluster filters join into cell_cluster_assignments.
    """
    joins      = []
    conditions = []
    params: list = []

    # Categorical dimension filters
    _dim_f("ct",  "cell_types",   "cell_type_id",   cell_type,       joins, conditions, params)
    _dim_f("ti",  "tissues",      "tissue_id",       tissue,          joins, conditions, params)
    _dim_f("bt",  "batches",      "batch_id",        batch_condition, joins, conditions, params)
    _dim_f("st",  "sample_types", "sample_type_id",  sample_type,     joins, conditions, params)
    _dim_f("rg",  "regions",      "region_id",       region,          joins, conditions, params)
    _dim_f("oi",  "orig_idents",  "orig_ident_id",   orig_ident,      joins, conditions, params)
    _dim_f("ag",  "ages",         "age_id",          age,             joins, conditions, params)
    _dim_f("org", "organisms",    "organism_id",     organism,        joins, conditions, params)
    _dim_f("ay",  "assays",       "assay_id",        assay,           joins, conditions, params)
    _dim_f("sx",  "sexes",        "sex_id",          sex,             joins, conditions, params)

    # Numeric filters
    if seurat_cluster is not None:
        conditions.append("c.seurat_clusters = ?");     params.append(seurat_cluster)
    if percent_mt_max is not None:
        conditions.append("c.percent_mt <= ?");          params.append(percent_mt_max)
    if n_count_rna_min is not None:
        conditions.append("c.n_count_rna >= ?");         params.append(n_count_rna_min)
    if n_count_rna_max is not None:
        conditions.append("c.n_count_rna <= ?");         params.append(n_count_rna_max)

    # Resolution + cluster filter
    if cluster_resolution is not None or cluster_id is not None:
        joins.append("JOIN cell_cluster_assignments cca ON cca.cell_id = c.id")
        if cluster_resolution is not None:
            conditions.append("cca.resolution = ?");    params.append(cluster_resolution)
        if cluster_id is not None:
            conditions.append("cca.cluster_id = ?");    params.append(cluster_id)

    # LEFT JOIN any dimension table not already in a WHERE join
    # so the SELECT can always reference the name columns
    used = {j.split()[2] for j in joins}
    for alias, table, fk in [
        ("ct",  "cell_types",   "cell_type_id"),
        ("ti",  "tissues",      "tissue_id"),
        ("bt",  "batches",      "batch_id"),
        ("st",  "sample_types", "sample_type_id"),
        ("rg",  "regions",      "region_id"),
        ("oi",  "orig_idents",  "orig_ident_id"),
        ("ag",  "ages",         "age_id"),
        ("org", "organisms",    "organism_id"),
        ("ay",  "assays",       "assay_id"),
        ("sx",  "sexes",        "sex_id"),
    ]:
        if alias not in used:
            joins.append(f"LEFT JOIN {table} {alias} ON {alias}.id = c.{fk}")

    where   = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    join_sq = "\n        ".join(joins)
    params.append(limit)

    sql = f"""
        SELECT
            c.id  AS cell_id, c.barcode,
            ct.name  AS cell_type,   ti.name  AS tissue,
            bt.name  AS batch_condition,
            st.name  AS sample_type, rg.name  AS region,
            oi.name  AS orig_ident,  ag.name  AS age,
            org.name AS organism,    ay.name  AS assay,
            sx.name  AS sex,
            c.n_count_rna, c.n_feature_rna, c.percent_mt, c.gene,
            c.n_genes, c.n_genes_by_counts,
            c.total_counts, c.total_counts_mt, c.pct_counts_mt,
            c.seurat_clusters, c.unintegrated_clusters
        FROM cells c
        {join_sq}
        {where}
        LIMIT ?
    """

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def fetch_embeddings_for_cells(
    db_path: str, cell_ids: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Return (M, D) float32 matrix and aligned cell_id list."""
    if not cell_ids:
        return np.empty((0,), dtype=np.float32), []
    ph = ",".join("?" * len(cell_ids))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT cell_id, embedding FROM cell_embeddings WHERE cell_id IN ({ph})",
            cell_ids,
        ).fetchall()
    ids, arrs = [], []
    for r in rows:
        arrs.append(np.frombuffer(r["embedding"], dtype=np.float32).copy())
        ids.append(r["cell_id"])
    if not arrs:
        return np.empty((0,), dtype=np.float32), []
    return np.stack(arrs, axis=0), ids


def fetch_sentences_for_cells(db_path: str, cell_ids: list[int]) -> dict[int, str]:
    if not cell_ids:
        return {}
    ph = ",".join("?" * len(cell_ids))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT cell_id, sentence FROM cell_sentences WHERE cell_id IN ({ph})",
            cell_ids,
        ).fetchall()
    return {r["cell_id"]: r["sentence"] for r in rows}


def fetch_cells_by_ids(db_path: str, cell_ids: list[int]) -> list[dict]:
    """Fetch full cell metadata rows for an explicit list of cell_ids."""
    if not cell_ids:
        return []
    ph = ",".join("?" * len(cell_ids))
    joins = "\n        ".join([
        "LEFT JOIN cell_types   ct  ON ct.id  = c.cell_type_id",
        "LEFT JOIN tissues      ti  ON ti.id  = c.tissue_id",
        "LEFT JOIN batches      bt  ON bt.id  = c.batch_id",
        "LEFT JOIN sample_types st  ON st.id  = c.sample_type_id",
        "LEFT JOIN regions      rg  ON rg.id  = c.region_id",
        "LEFT JOIN orig_idents  oi  ON oi.id  = c.orig_ident_id",
        "LEFT JOIN ages         ag  ON ag.id  = c.age_id",
        "LEFT JOIN organisms    org ON org.id = c.organism_id",
        "LEFT JOIN assays       ay  ON ay.id  = c.assay_id",
        "LEFT JOIN sexes        sx  ON sx.id  = c.sex_id",
    ])
    sql = f"""
        SELECT
            c.id  AS cell_id, c.barcode,
            ct.name  AS cell_type,   ti.name  AS tissue,
            bt.name  AS batch_condition,
            st.name  AS sample_type, rg.name  AS region,
            oi.name  AS orig_ident,  ag.name  AS age,
            org.name AS organism,    ay.name  AS assay,
            sx.name  AS sex,
            c.n_count_rna, c.n_feature_rna, c.percent_mt, c.gene,
            c.n_genes, c.n_genes_by_counts,
            c.total_counts, c.total_counts_mt, c.pct_counts_mt,
            c.seurat_clusters, c.unintegrated_clusters
        FROM cells c
        {joins}
        WHERE c.id IN ({ph})
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, cell_ids).fetchall()
    return [dict(r) for r in rows]


def fetch_cells_by_gene(
    db_path: str,
    gene_name: str,
    top_n_position: int = 200,
    limit: int = 5000,
) -> list[dict]:
    """
    Return cells where *gene_name* appears in the top-N expressed genes.

    Cells are returned sorted by expression rank (lower rank = higher expression).
    A fallback with a wider position window is tried automatically if the strict
    search yields no results.
    """
    # Fetch all cells whose sentence contains the gene (anywhere in the 200-gene list)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT cell_id, sentence FROM cell_sentences "
            "WHERE sentence LIKE ? OR sentence LIKE ? OR sentence LIKE ?",
            (f"{gene_name} %", f"% {gene_name} %", f"% {gene_name}"),
        ).fetchall()

    # Determine expression rank for each cell and filter by position window.
    # Try progressively wider windows until we find enough cells.
    all_ranked: list[tuple[int, int]] = []
    for r in rows:
        genes = r["sentence"].split()
        try:
            rank = genes.index(gene_name)
        except ValueError:
            continue
        all_ranked.append((r["cell_id"], rank))

    if not all_ranked:
        return []

    for pos_limit in (top_n_position, 100, 200):
        ranked = [(cid, rank) for cid, rank in all_ranked if rank < pos_limit]
        if ranked:
            ranked.sort(key=lambda x: x[1])
            top_ids = [cid for cid, _ in ranked[:limit]]
            return fetch_cells_by_ids(db_path, top_ids)

    return []


def fetch_cluster_assignments(db_path: str, cell_id: int) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT resolution, cluster_id FROM cell_cluster_assignments "
            "WHERE cell_id=? ORDER BY resolution",
            (cell_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Private helpers ────────────────────────────────────────────────────────────

def _dim_f(alias, table, fk_col, value, joins, conditions, params):
    """Add INNER JOIN + LIKE condition when value is provided."""
    if value:
        joins.append(f"JOIN {table} {alias} ON {alias}.id = c.{fk_col}")
        conditions.append(f"{alias}.name LIKE ?")
        params.append(f"%{value}%")


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return None if x != x else x   # NaN → None
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None