#!/bin/bash

#SBATCH --job-name=PostHocExplanations         
#SBATCH --output=job_main_output_%j.txt         
#SBATCH --time=48:00:00                  
#SBATCH --partition=luna-gpu-long
#SBATCH --gres=gpu:1g.10gb:1                    
#SBATCH --mem=32GB                         
#SBATCH --cpus-per-task=8                

# Load required modules
module load Anaconda3/2024.02-1
module load cuda/11.8

# Activate your conda environment
source ~/my-scratch/miniconda3/etc/profile.d/conda.sh
conda activate GNN

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Optional: Configure PyTorch CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diagnostic: GPU info
echo "=== GPU info ==="
nvidia-smi
nvidia-smi -L
echo "================"

# Change to your project directory
cd /home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/

# Run your main script
python MainProject/Modules/ExplainabilityLayerAnalysis_updated.py
