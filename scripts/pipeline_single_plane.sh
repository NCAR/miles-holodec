#!/bin/bash
### End-to-end single-plane pipeline: train → inference → evaluate
### Single-plane model predicts mask only (no depth), so z_um = z_center.
#PBS -N holo_val_pipe
#PBS -A NAML0001
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

set -e

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
CONFIG=$REPO/config/single_plane_validate.yml
SAVE=/glade/derecho/scratch/schreck/holodec/single_plane_validate
INFER_OUT=/glade/derecho/scratch/schreck/holodec/inference_single_plane
EVAL_OUT=/glade/derecho/scratch/schreck/holodec/evaluation_single_plane
VAL_NC=/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc

mkdir -p $SAVE $INFER_OUT $EVAL_OUT

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

echo "=== STEP 1: Training ==="
python applications/trainer_unet_planer.py -c $CONFIG

echo "=== STEP 2: Inference (full validation set) ==="
python applications/inference_planar.py \
    -c $CONFIG \
    -k $SAVE/checkpoint.pt \
    -o $INFER_OUT \
    --z-stride 1 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda

echo "=== STEP 3: Evaluation ==="
python applications/evaluate_detections.py \
    -d $INFER_OUT/detections.csv \
    -n $VAL_NC \
    -o $EVAL_OUT \
    --cluster \
    --xy-radius 50 \
    --z-radius 200 \
    --xy-tol 50 \
    --z-tol 360

echo "=== Done. Results in $EVAL_OUT ==="
