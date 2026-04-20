#!/bin/bash
#SBATCH --job-name=cell_db_build
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

# --- Your commands go below this line ---
module purge
module load cuda/12.1

# Activate the Cell2Sentence conda environment
source activate cell2sentence

# --- GPU Initialization Check ---
echo "Checking GPU status..."
nvidia-smi

# --- Run build pipeline ---
python3 /home/zhuoxuan/C2S/cell_data_base/build_database.py \
    --h5ad_path   /home/zhuoxuan/C2S/Guts/guts_data/guts_preprocessed_data.h5ad \
    --embed_model /nfs/turbo/umms-drjieliu/proj/c2s_model/C2S-Pythia-410m-cell-type-prediction \
    --db_path     /home/zhuoxuan/C2S/cell_data_base/data/guts.db \
    --arrow_dir   /home/zhuoxuan/C2S/cell_data_base/data/arrow_ds \
    --n_genes     200
