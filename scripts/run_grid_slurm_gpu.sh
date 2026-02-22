#!/bin/bash
#SBATCH --job-name=grid_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/grid_gpu_%A_%a.out
#SBATCH --error=slurm/grid_gpu_%A_%a.err

# GPU version - 5 parallel jobs, one per experiment type.
# Submit: sbatch scripts/run_grid_slurm_gpu.sh

EXP_TYPES=(teacher_wd student_teacher_w batch_student_w batch_student_d student_wd)
# Submit from project root: cd NodePerturbation && sbatch scripts/run_grid_slurm_gpu.sh
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${PROJECT_DIR}/results/grid_10x10_200ep"

mkdir -p slurm

# module load python/3.9
# module load cuda/11.8  # if needed
# source activate your_env

cd "${PROJECT_DIR}" || exit 1

EXP_TYPE=${EXP_TYPES[$SLURM_ARRAY_TASK_ID]}
echo "Running ${EXP_TYPE} on GPU (task ${SLURM_ARRAY_TASK_ID})"

python3 experiments/grid_experiments.py \
    --exp_type ${EXP_TYPE} \
    --grid_size 10 \
    --max_epochs 200 \
    --output_base "${OUTPUT_BASE}"

echo "Done: ${EXP_TYPE}"
