#!/bin/bash
#SBATCH --job-name=grid_exp
#SBATCH --partition=mit_normal
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/grid_%A_%a.out
#SBATCH --error=slurm/grid_%A_%a.err

# Run grid experiments on Engaging cluster.
# Array index maps to: 0=teacher_wd, 1=student_teacher_w, 2=batch_student_w, 3=batch_student_d, 4=student_wd
# Submit: sbatch scripts/run_grid_slurm.sh

EXP_TYPES=(teacher_wd student_teacher_w batch_student_w batch_student_d student_wd)
# Submit from project root: cd NodePerturbation && sbatch scripts/run_grid_slurm.sh
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${PROJECT_DIR}/results/grid_10x10_200ep"

mkdir -p slurm

# Load modules if needed (customize for your environment)
# module load python/3.9
# module load anaconda3/2022.05
# source activate your_env  # if using conda

cd "${PROJECT_DIR}" || exit 1

EXP_TYPE=${EXP_TYPES[$SLURM_ARRAY_TASK_ID]}
echo "Running ${EXP_TYPE} (task ${SLURM_ARRAY_TASK_ID})"

python3 experiments/grid_experiments.py \
    --exp_type ${EXP_TYPE} \
    --grid_size 10 \
    --max_epochs 200 \
    --output_base "${OUTPUT_BASE}"

echo "Done: ${EXP_TYPE}"
