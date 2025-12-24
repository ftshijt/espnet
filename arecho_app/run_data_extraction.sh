#!/bin/bash
#SBATCH --job-name=data_extraction
#SBATCH --time=48:00:00
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000M
#SBATCH --account=bbjs-dtai-gh
#SBATCH --output=logs/data_extraction_%j.out
#SBATCH --error=logs/data_extraction_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment if activate script exists
if [ -f "../tools/activate_python.sh" ]; then
    source ./../tools/activate_python.sh
fi

# Set CUDA visible devices (optional, SLURM handles this automatically)
# export CUDA_VISIBLE_DEVICES=0

# Run the data extraction script
echo "Starting data extraction..."
python data_extraction_2.py

echo "End Time: $(date)"
echo "Job completed!"

