#!/bin/bash
### Inference + evaluation for multiplane model (runs after training)
#PBS -N holo_multi_eval
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=64GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

set -e

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
SAVE=/glade/derecho/scratch/schreck/holodec/multiplane
INFER_OUT=/glade/derecho/scratch/schreck/holodec/inference_validation
EVAL_OUT=/glade/derecho/scratch/schreck/holodec/evaluation
VAL_NC=/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc

mkdir -p $INFER_OUT $EVAL_OUT

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

echo "=== Inference ==="
python applications/inference_multiplane.py \
    -c config/multiplane.yml \
    -k $SAVE/checkpoint.pt \
    -o $INFER_OUT \
    --z-stride 5 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda

echo "=== Evaluation ==="
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
