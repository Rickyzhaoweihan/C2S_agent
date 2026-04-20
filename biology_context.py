"""
biology_context.py
------------------
Biological knowledge base extracted from:

  Vandana et al. "Integrative Single-Cell Spatial Multi-Omics Analysis of
  Human Fetal Intestine."

This module provides:
  - SYSTEM_PREAMBLE : biology context string for use with a general-purpose LLM
                      (NOT injected into the C2S model; reserved for future use)
  - CELL_TYPE_ALIASES : maps common/shorthand names → canonical cell_type values
  - MARKER_TO_CELL_TYPE : known marker genes → cell type (for query parsing)
  - REGION_ALIASES : shorthand region names → canonical values

Nothing here imports from the other project files — it is a pure data module.
"""
from __future__ import annotations


# ── Dataset summary ────────────────────────────────────────────────────────────

DATASET_SUMMARY = """
Dataset: Human fetal and adult intestine, integrated multi-omics.
  - snATAC-seq: 86,159 epithelial cells, 9 fetal samples (11–17 weeks
    post-conception), regions: duodenum, jejunum, ileum, colon.
  - scRNA-seq: 374,445 cells from 6 published datasets, ages 6 weeks
    post-conception to 70 years, all gut regions.
  - Spatial transcriptomics: Xenium (10X), duodenum and colon,
    17 and 20 weeks post-conception.
  - Spatial metabolomics: 295 metabolites, matched duodenum/colon sections,
    17-week donor.
"""

# ── Epithelial cell types and their canonical markers ─────────────────────────

CELL_TYPE_MARKERS = {
    "stem cell": {
        "canonical_markers": ["LGR5", "MYC"],
        "novel_markers":     ["MECOM"],
        "description": (
            "LGR5+ intestinal stem cells reside in crypts and self-renew to "
            "replenish the epithelium. MECOM was identified as a previously "
            "unknown stem cell marker via snATAC-seq differentially accessible "
            "regions (DARs)."
        ),
    },
    "TA cell": {
        "canonical_markers": ["TOP2A"],
        "novel_markers":     ["NUSAP1"],
        "description": (
            "Transiently amplifying (TA) cells are rapidly dividing progenitors "
            "that sit between stem cells and differentiated cells in the crypt. "
            "NUSAP1 was identified as a novel TA marker. TA cells also express "
            "DBH, producing norepinephrine that may activate cAMP-CREB signalling "
            "to promote adjacent K cell differentiation."
        ),
    },
    "enterocyte": {
        "canonical_markers": ["APOA1", "APOC3"],
        "novel_markers":     ["ACSL5"],
        "description": (
            "Absorptive enterocytes line the villi and are responsible for "
            "nutrient absorption. APOA1 localises to villi in spatial "
            "transcriptomics. ACSL5 was identified as a novel enterocyte marker."
        ),
    },
    "goblet cell": {
        "canonical_markers": ["MUC2", "CLCA1", "SPINK4"],
        "novel_markers":     ["GALNT8"],
        "description": (
            "Mucus-secreting goblet cells are dispersed along the villi and "
            "play a critical role in forming the protective mucus layer. "
            "JAK-STAT3 signalling (activated by Colivelin-TFA) promotes goblet "
            "cell differentiation. GALNT8 is a novel goblet marker; GABRB2 is "
            "also expressed in goblet cells as part of GABAergic signalling."
        ),
    },
    "enteroendocrine cell": {
        "canonical_markers": ["CHGA", "NEUROD1"],
        "novel_markers":     ["RIMBP2"],
        "description": (
            "Enteroendocrine cells (EECs) are rare, hormone-secreting epithelial "
            "cells. GAD2 is predominantly expressed in EECs, making them the "
            "likely source of GABA in the duodenum. GABRB3 is highly expressed "
            "in EECs. RIMBP2 is a novel EEC marker."
        ),
    },
}

# ── EEC subtypes ───────────────────────────────────────────────────────────────

EEC_SUBTYPES = {
    "L cell": {
        "markers":   ["PYY", "GCG", "THSD7A"],
        "hormone":   "GLP-1 (glucagon-like peptide 1)",
        "location":  "lower intestine (colon and ileum); duodenum and colon",
        "pathway":   "ROCK inhibition promotes L cell differentiation",
        "description": (
            "L cells secrete GLP-1 and are relevant to obesity/diabetes "
            "therapeutics (semaglutide targets GLP-1R). ROCK1 expression is "
            "relatively low in L cells. Rho GTPase inhibition (MLS000532223) "
            "increases PYY and NTS expression, promoting L and N cell formation."
        ),
    },
    "K cell": {
        "markers":   ["GIP"],
        "hormone":   "GIP (glucose-dependent insulinotropic polypeptide)",
        "location":  "upper intestine (duodenum), near crypts",
        "pathway":   "cAMP-CREB activation promotes K cell differentiation",
        "description": (
            "K cells are localized near the crypt base close to stem cells and "
            "TA cells. TA cells produce norepinephrine via DBH, activating the "
            "cAMP-CREB pathway to promote K cell specification. EP300, a CREB "
            "coactivator, is enriched in K cells. PNMT is highly expressed in K "
            "cells, converting norepinephrine to epinephrine. Tirzepatide "
            "targets both GLP-1R and GIPR."
        ),
    },
    "enterochromaffin cell": {
        "markers":   ["TPH1", "CES1", "SLC38A11", "CDHR3", "LMX1A"],
        "hormone":   "Serotonin",
        "location":  "duodenum and colon",
        "description": "Produces serotonin; distributed throughout both regions.",
    },
    "D cell": {
        "markers":   ["SST", "REG1A", "HHEX"],
        "hormone":   "Somatostatin",
        "location":  "duodenum and colon",
        "description": "CREB activation (8-Br-cAMP) enhances SST expression.",
    },
    "M/X/A cell": {
        "markers":   ["MLN", "GHRL"],
        "hormone":   "Motilin / Ghrelin",
        "location":  "predominantly fetal",
        "description": "Enriched in fetal samples; ghrelin-producing cell type.",
    },
    "N cell": {
        "markers":   ["NTS"],
        "hormone":   "Neurotensin",
        "location":  "duodenum and colon",
        "description": "ROCK inhibition promotes NTS expression and N cell formation.",
    },
    "I cell": {
        "markers":   ["CCK", "ONECUT3"],
        "hormone":   "Cholecystokinin",
        "location":  "human fetal duodenum only",
        "description": "Duodenum-restricted; CCK secretion supports digestion.",
    },
    "endocrine progenitor": {
        "markers":   [],
        "hormone":   "None (uncommitted)",
        "description": (
            "The most abundant EEC subtype in fetal samples, reflecting "
            "incomplete maturation of the EEC lineage at the fetal stage."
        ),
    },
}

# ── Region-specific markers ────────────────────────────────────────────────────

REGION_MARKERS = {
    "duodenum": {
        "all_cell_types": ["ANXA13", "CCL25", "RBP2", "SERPINA1"],
        "differentiated_cells": ["AFP", "APOA1", "APOC3", "APOB"],
        "stem_cells": ["EDN1", "STXBP6", "CD9", "CD24"],
        "key_pathway": "GABAergic receptor signalling (GAD2, GABRB2, GABRB3)",
        "description": (
            "The duodenum shows enhanced GABAergic receptor signalling and "
            "elevated GABA metabolite levels. GAD2 (encodes GAD65, converts "
            "glutamate to GABA) is enriched in duodenal EECs. These "
            "GABAergic genes are enriched only in fetal, not adult, tissue. "
            "K cells and I cells are specifically localised to the duodenum."
        ),
    },
    "colon": {
        "all_cell_types": ["C10orf99", "C15orf48", "SATB2", "SLC26A2"],
        "differentiated_cells": ["CEACAM7", "COL17A1", "GUCA2A"],
        "stem_cells": ["ADAMTSL1", "LEFTY1", "LITD1", "ALDH1B1"],
        "key_pathway": "JAK-STAT3 promotes goblet cell differentiation",
        "description": (
            "SATB2 is a key colon-specific marker that preserves colon stem "
            "cell identity. GABA treatment significantly downregulates SATB2, "
            "suggesting a shift toward duodenal identity. Colonic stem cells "
            "express ADAMTSL1, LEFTY1, and LITD1."
        ),
    },
}

# ── Key signalling pathways ────────────────────────────────────────────────────

PATHWAYS = {
    "GABAergic receptor signalling": {
        "region": "duodenum",
        "genes":  ["GAD2", "GABRB2", "GABRB3", "GABRA1", "GABRA4"],
        "description": (
            "Enriched in fetal duodenum vs. colon. GAD65 (encoded by GAD2) "
            "converts glutamate to GABA. Spatial metabolomics confirmed elevated "
            "GABA in duodenal sections. Adding 100 µM GABA to hESC-derived "
            "organoids shifts differentiation away from colon (reduced SATB2). "
            "This pathway is fetal-specific — not enriched in adult tissue."
        ),
    },
    "JAK-STAT3 signalling": {
        "cell_type": "goblet cell",
        "genes":     ["STAT3"],
        "treatment": "Colivelin-TFA (STAT3 activator, 200 nM, Day 48–70)",
        "description": (
            "STAT3 activation with Colivelin-TFA significantly increases "
            "CLCA1, SPINK4, and MUC2 expression in hESC-derived organoids, "
            "promoting goblet cell differentiation. STAT3 blockade causes "
            "severe maturation defects in intestinal organoids."
        ),
    },
    "ROCK / Rho GTPase signalling": {
        "cell_type": "L cell (EEC subtype)",
        "genes":     ["ROCK1", "RHOA"],
        "treatment": "MLS000532223 (Rho inhibitor, 10 µM, Day 48–70)",
        "description": (
            "ROCK inhibition promotes L cell and N cell formation, increasing "
            "PYY and NTS expression. ROCK1 is relatively low in L cells. "
            "Consistent with Y-27632 (ROCK1/2 inhibitor) increasing L cell "
            "numbers in mouse intestinal organoids."
        ),
    },
    "cAMP-CREB signalling": {
        "cell_type": "K cell (EEC subtype)",
        "genes":     ["EP300", "PNMT", "DBH"],
        "treatment": "8-Br-cAMP (CREB activator, 100 µM, Day 48–70)",
        "description": (
            "CREB activation promotes K cell and D cell formation (GIP, SST). "
            "EP300 (CREB coactivator) is enriched in K cells. TA cells produce "
            "norepinephrine (via DBH) which activates cAMP-CREB in adjacent "
            "K cells. PNMT (converts norepinephrine to epinephrine) is highly "
            "expressed in EECs, especially K cells."
        ),
    },
}

# ── Aliases used by CatalogueMatcher ──────────────────────────────────────────
# Maps what a user might type → substring to search in the catalogue

CELL_TYPE_ALIASES: dict[str, list[str]] = {
    # Stem cells
    "stem":          ["stem"],
    "lgr5":          ["stem"],
    "mecom":         ["stem"],
    # TA cells
    "ta cell":       ["TA", "transient"],
    "transient":     ["TA", "transient"],
    "top2a":         ["TA", "transient"],
    "nusap1":        ["TA", "transient"],
    # Enterocytes
    "enterocyte":    ["enterocyte"],
    "apoa1":         ["enterocyte"],
    "absorptive":    ["enterocyte"],
    # Goblet cells
    "goblet":        ["goblet"],
    "muc2":          ["goblet"],
    "mucus":         ["goblet"],
    # EEC / enteroendocrine
    "eec":           ["enteroendocrine", "EEC"],
    "enteroendocrine": ["enteroendocrine", "EEC"],
    "chga":          ["enteroendocrine", "EEC"],
    "hormone":       ["enteroendocrine", "EEC"],
    # EEC subtypes
    "l cell":        ["L cell", "enteroendocrine"],
    "glp-1":         ["L cell", "enteroendocrine"],
    "glp1":          ["L cell", "enteroendocrine"],
    "pyy":           ["L cell", "enteroendocrine"],
    "k cell":        ["K cell", "enteroendocrine"],
    "gip":           ["K cell", "enteroendocrine"],
    "incretin":      ["K cell", "L cell", "enteroendocrine"],
    "enterochromaffin": ["enterochromaffin", "enteroendocrine"],
    "serotonin":     ["enterochromaffin", "enteroendocrine"],
    "tph1":          ["enterochromaffin", "enteroendocrine"],
    "d cell":        ["D cell", "enteroendocrine"],
    "somatostatin":  ["D cell", "enteroendocrine"],
    "sst":           ["D cell", "enteroendocrine"],
    "i cell":        ["I cell", "enteroendocrine"],
    "cck":           ["I cell", "enteroendocrine"],
    "n cell":        ["N cell", "enteroendocrine"],
    "neurotensin":   ["N cell", "enteroendocrine"],
    "nts":           ["N cell", "enteroendocrine"],
}

REGION_ALIASES: dict[str, str] = {
    "small intestine": "duodenum",
    "upper intestine": "duodenum",
    "lower intestine": "colon",
    "large intestine": "colon",
    "colon":           "colon",
    "duodenum":        "duodenum",
    "ileum":           "ileum",
    "jejunum":         "jejunum",
}

MARKER_TO_CELL_TYPE: dict[str, str] = {
    # Stem
    "LGR5": "stem cell", "MYC": "stem cell", "MECOM": "stem cell",
    # TA
    "TOP2A": "TA cell", "NUSAP1": "TA cell",
    # Enterocyte
    "APOA1": "enterocyte", "APOC3": "enterocyte", "ACSL5": "enterocyte",
    # Goblet
    "MUC2": "goblet cell", "CLCA1": "goblet cell",
    "SPINK4": "goblet cell", "GALNT8": "goblet cell",
    # EEC
    "CHGA": "enteroendocrine cell", "NEUROD1": "enteroendocrine cell",
    "RIMBP2": "enteroendocrine cell",
    # EEC subtypes
    "PYY": "L cell", "GCG": "L cell",
    "GIP": "K cell",
    "TPH1": "enterochromaffin cell",
    "SST": "D cell",
    "NTS": "N cell",
    "CCK": "I cell",
    "MLN": "M/X/A cell", "GHRL": "M/X/A cell",
    # Regional
    "SATB2": "colon marker", "GUCA2A": "colon marker",
    "CEACAM7": "colon marker",
    "ANXA13": "duodenum marker", "CCL25": "duodenum marker",
    "SERPINA1": "duodenum marker", "GAD2": "duodenum EEC marker",
}

# ── System preamble for chat.py ────────────────────────────────────────────────

SYSTEM_PREAMBLE = """You are an expert single-cell RNA biologist with deep knowledge \
of the human gut, specialising in the dataset described below.

DATASET
-------
{dataset_summary}

EPITHELIAL CELL TYPES AND KEY MARKERS
--------------------------------------
• Stem cells       : LGR5, MYC (canonical); MECOM (novel, snATAC-seq)
• TA cells         : TOP2A (canonical); NUSAP1 (novel); produce norepinephrine via DBH
• Enterocytes      : APOA1, APOC3 (canonical); ACSL5 (novel); localise to villi
• Goblet cells     : MUC2, CLCA1, SPINK4 (canonical); GALNT8 (novel);
                     JAK-STAT3 activation (Colivelin-TFA) drives differentiation
• EECs             : CHGA, NEUROD1 (canonical); RIMBP2 (novel);
                     GAD2 (primary GABA source in duodenum); GABRB3 highly expressed

EEC SUBTYPES
------------
• L cells          : PYY, GCG, THSD7A → GLP-1; ROCK inhibition promotes these
• K cells          : GIP → incretin; cAMP-CREB activation (EP300 enriched); duodenum crypt
• Enterochromaffin : TPH1, LMX1A → serotonin; duodenum and colon
• D cells          : SST, HHEX → somatostatin; CREB activation promotes these
• I cells          : CCK, ONECUT3 → cholecystokinin; duodenum only
• N cells          : NTS → neurotensin; ROCK inhibition promotes these
• M/X/A cells      : MLN, GHRL → motilin/ghrelin; enriched in fetal samples
• Endocrine prog.  : uncommitted; most abundant EEC type in fetal tissue

REGIONAL MARKERS
----------------
Duodenum (all cells) : ANXA13, CCL25, RBP2, SERPINA1
Duodenum (diff.)     : AFP, APOA1, APOC3, APOB
Duodenum (stem)      : EDN1, STXBP6, CD9, CD24
Colon (all cells)    : C10orf99, C15orf48, SATB2, SLC26A2
Colon (diff.)        : CEACAM7, COL17A1, GUCA2A
Colon (stem)         : ADAMTSL1, LEFTY1, LITD1, ALDH1B1

KEY SIGNALLING PATHWAYS
------------------------
• GABAergic signalling → duodenum specification (fetal only; GAD2, GABRB2, GABRB3)
  GABA addition shifts hESC organoids away from colon (↓ SATB2)
• JAK-STAT3 → goblet cell differentiation (Colivelin-TFA activates STAT3)
• ROCK inhibition → L cell and N cell promotion (↑ PYY, NTS)
• cAMP-CREB activation → K cell and D cell promotion (↑ GIP, SST; EP300 enriched in K cells)
• TA cells → norepinephrine (DBH) → cAMP-CREB → K cell specification (spatial proximity)

TASK
----
Use the retrieved cell context to answer the researcher's question accurately and \
concisely. Focus on cell identity, marker genes, tissue function, signalling pathways, \
and regional specificity. When discussing EECs, specify the subtype if possible. \
Cite novel markers (MECOM, NUSAP1, ACSL5, GALNT8, RIMBP2) where relevant.""".format(
    dataset_summary=DATASET_SUMMARY.strip()
)