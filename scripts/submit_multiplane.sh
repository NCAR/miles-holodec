#!/bin/bash
### Multi-plane training — live GPU wave propagation, 1x A100, 12h
### Usage: qsub scripts/submit_multiplane.sh
#PBS -N holo_multi
#PBS -A NAML0001
#PBS -l walltime=12:00:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
CONFIG=$REPO/config/multiplane.yml

mkdir -p /glade/derecho/scratch/schreck/holodec/multiplane

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

python applications/trainer_unet.py -c $CONFIG
