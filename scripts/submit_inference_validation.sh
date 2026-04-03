#!/bin/bash
### Full validation set inference — Matt's multiplane checkpoint, all 10 holograms
#PBS -N holo_infer_val
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=64GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
OUT=/glade/derecho/scratch/schreck/holodec/inference_validation
mkdir -p $OUT

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

python applications/inference_multiplane.py \
    -c config/inference_multiplane_matt.yml \
    -k results/checkpoint.pt \
    -o $OUT \
    --z-stride 5 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda
# No --max-holograms: processes all holograms in the file
