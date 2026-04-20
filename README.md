# Gut Cell Database — Build & Chat Agent

A single-cell RNA-seq database and conversational query agent for the human gut, grounded in:

> Vandana et al. *"Integrative Single-Cell Spatial Multi-Omics Analysis of Human Fetal Intestine."*

The pipeline converts an AnnData (`h5ad`) file into a normalized SQLite database with C2S embeddings, then exposes a natural-language chat interface backed by a vLLM-served C2S model.

---

## Project Structure

```
cell_data_base/
├── build_database.py       # Offline pipeline: h5ad → SQLite + Arrow dataset
├── build_cell_database.sh  # SLURM batch script to run the build pipeline
├── database.py             # SQLite schema, insert helpers, and query functions
├── cell_db_agent.py        # Core query agent (SQL + cosine search + NL generation)
├── biology_context.py      # Curated biology knowledge base (cell types, markers, pathways)
├── chat.py                 # Chat wrapper + CLI entry point
├── data/
│   ├── guts.db             # Built SQLite database
│   ├── arrow_ds/           # Arrow dataset (required by the NL model)
│   └── csmodel_cache/      # Cached C2S embedding model weights
├── cell_embeddings.npy     # (Optional) Pre-computed embeddings for fast cosine search
└── cell_embedding_ids.npy  # (Optional) Corresponding cell IDs for the above
```

---

## Prerequisites

- **Conda environment:** `cell2sentence` (with `cell2sentence`, `anndata`, `torch`, `transformers`, `datasets`, `numpy`, `requests` installed)
- **Models:**
  - Embedding model: `C2S-Pythia-410m-cell-type-prediction`
  - NL generation model: `CC2S-Scale-Gemma-2-2B` (or a Gemma-2 variant)
- **Input data:** A preprocessed AnnData `.h5ad` file (see `build_database.py` for expected `obs` columns)
- **Cluster:** SLURM with GPU access (L40S or equivalent) for building and vLLM serving

---

## Step 1 — Build the Database

The build step is a one-time offline pipeline. It:
1. Loads the AnnData file and resolves `obs` column aliases
2. Converts cells to Cell Sentence format (Arrow dataset)
3. Embeds all cells using the C2S embedding model
4. Inserts cells, embeddings, and gene sentences into a normalized SQLite database

### Option A: Submit SLURM job (recommended)

Edit paths in `build_cell_database.sh` if needed, then submit:

```bash
sbatch build_cell_database.sh
```

The script requests 1 GPU, 16 CPUs, 256 GB RAM, and up to 20 hours. Output is written to `cell_db_build-<jobid>.out`.

### Option B: Run directly

```bash
source activate cell2sentence

python build_database.py \
    --h5ad_path   /path/to/guts_preprocessed_data.h5ad \
    --embed_model /path/to/C2S-Pythia-410m-cell-type-prediction \
    --db_path     ./data/guts.db \
    --arrow_dir   ./data/arrow_ds \
    --n_genes     200
```

| Argument | Description |
|---|---|
| `--h5ad_path` | Path to the preprocessed AnnData `.h5ad` file |
| `--embed_model` | Path to the C2S embedding model directory |
| `--db_path` | Output path for the SQLite database |
| `--arrow_dir` | Output directory for the Arrow dataset |
| `--n_genes` | Number of top genes per cell sentence (default: 200) |

After a successful build the script prints a catalogue of all unique values per filterable dimension.

---

## Step 2 — Start the vLLM Server

The chat agent sends generation requests to a running vLLM server rather than loading the NL model in-process. Start the server **before** launching `chat.py`.

```bash
# Example: start vLLM on a GPU node
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/C2S-Scale-Pythia-1b-pt \
    --port 8000 \
    --host 0.0.0.0
```

On a SLURM cluster, wrap this in a batch script and note the node hostname from the job output — you will pass it as `--vllm_server_url`.

The agent connects to `http://<node>:8000` and verifies `/health` at startup. If the server is unreachable, `chat.py` will exit immediately with a clear error message.

---

## Step 3 — Run the Chat Agent

### Interactive mode (REPL)

```bash
python chat.py \
    --db_path    ./data/guts.db \
    --arrow_dir  ./data/arrow_ds \
    --nl_model   /path/to/C2S-Scale-Gemma-2-2B \
    --vllm_server_url http://<node>:8000
```

Type queries at the `You:` prompt. Special commands:
- `reset` — clear conversation history
- `quit` / `exit` / `q` — exit

### Single-query mode

```bash
python chat.py \
    --db_path    ./data/guts.db \
    --arrow_dir  ./data/arrow_ds \
    --nl_model   /path/to/C2S-Scale-Gemma-2-2B \
    --vllm_server_url http://<node>:8000 \
    --query "tell me about goblet cells in the colon"
```

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--db_path` | (required) | Path to the SQLite database |
| `--arrow_dir` | (required) | Path to the Arrow dataset directory |
| `--nl_model` | (required) | Path/name of the vLLM-served NL model |
| `--vllm_server_url` | `http://localhost:8000` | URL of the running vLLM server |
| `--query` | None | Single query (omit for interactive mode) |
| `--top_k_genes` | 200 | Genes per cell sentence in NL prompts |
| `--n_cells_per_prompt` | 5 | Cells grouped per multi-cell NL prompt |
| `--n_neighbours` | 20 | Cells returned by cosine similarity search |
| `--max_new_tokens` | 512 | Maximum tokens for NL generation |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_k` | 30 | Top-k sampling |
| `--top_p` | 0.9 | Top-p (nucleus) sampling |

---

## Example Queries

```
# Cell type queries
tell me about goblet cells
what are enteroendocrine cells?
describe stem cells in the duodenum

# Gene queries
MUC2 expression in the colon
which cells express GIP?
APOA1

# Comparison queries
compare goblet cells and enterocytes
L cells vs K cells
difference between stem cells and TA cells

# Metadata + filter queries
goblet cells in the colon from adult donors
enterocytes in the jejunum
cells in seurat cluster 3
resolution 0.35 cluster 2

# QC-filtered queries
cells with percent_mt < 5
n_count_rna > 1000
```

---

## Python API

You can use the agent programmatically in a notebook or script:

```python
from cell_db_agent import CellDatabaseAgent
from chat import C2SChatbot

agent = CellDatabaseAgent(
    db_path            = "./data/guts.db",
    arrow_dir          = "./data/arrow_ds",
    nl_model_path      = "/path/to/C2S-Scale-Gemma-2-2B",
    vllm_server_url    = "http://<node>:8000",
    n_neighbours       = 20,
    n_cells_per_prompt = 5,
)

chatbot = C2SChatbot(agent=agent, max_new_tokens=512)

# Single turn
response = chatbot.chat("tell me about goblet cells")
print(response)

# Parse a query into structured filters (useful for debugging)
cell_query = chatbot.parse_query("goblet cells in the colon with percent_mt < 5")
print(cell_query.active_filters())

# Reset conversation history between sessions
chatbot.reset()
```

---

## Database Schema Overview

The SQLite database uses a normalized star schema:

| Table | Description |
|---|---|
| `cells` | Fact table — one row per cell barcode; numeric QC columns; FK into dimension tables |
| `cell_types`, `tissues`, `batches`, … | Dimension tables — one row per unique categorical value |
| `cell_cluster_assignments` | Repeating-group table — one row per (cell, Seurat resolution) pair |
| `cell_embeddings` | BLOB table — float32 C2S embedding stored as raw bytes |
| `cell_sentences` | Space-delimited gene sentence string per cell |
| `genes` | Reference table — Ensembl ID, gene name, mitochondrial flag |

All categorical filters use case-insensitive `LIKE` substring matching. Numeric and cluster filters use exact or range bounds.

---

## How the Query Pipeline Works

```
User text
    │
    ▼
CatalogueMatcher.parse_query()          ← three-tier NL → CellQuery resolution
    │   1. Marker gene lookup (MECOM → stem cell)
    │   2. Biology alias     (GLP-1 → L cell)
    │   3. Fuzzy n-gram match against live catalogue
    │
    ▼
CellDatabaseAgent.query(cell_query)
    │
    ├─ Direct path (cell/gene IS in DB):
    │       SQL fetch → aggregate metadata + gene sentences
    │       + curated biology knowledge → conversational prose
    │
    └─ Similarity path (NOT in DB):
            Build query vector (mean embedding of matching cells)
            → L2 cosine search across all cell embeddings
            → characterise top-k matches
            → conversational prose with similarity caveat
```

---

## Curated Biology Knowledge Base

`biology_context.py` encodes cell-type markers, EEC subtypes, regional markers, and key signalling pathways extracted from the Vandana et al. paper. It is used for:

- Query parsing (alias and marker-to-cell-type resolution)
- Response enrichment (inline gene annotation, function paragraphs)
- Comparison queries (answered directly without a database search)

No external LLM is required for these responses — they are assembled from curated dictionaries.
