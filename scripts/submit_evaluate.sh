#!/bin/bash
### Evaluate multiplane detections against ground truth
#PBS -N holo_evaluate
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=32GB
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
DET=/glade/derecho/scratch/schreck/holodec/inference_validation/detections.csv
NC=/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc
OUT=/glade/derecho/scratch/schreck/holodec/evaluation

mkdir -p $OUT

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

python applications/evaluate_detections.py \
    -d $DET \
    -n $NC \
    -o $OUT \
    --cluster \
    --xy-radius 50 \
    --z-radius 200 \
    --xy-tol 50 \
    --z-tol 360
