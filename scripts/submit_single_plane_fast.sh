#!/bin/bash
### Single-plane fast run — pre-computed tiles, no live wave propagation
### Usage: qsub scripts/submit_single_plane_fast.sh
#PBS -N holo_fast
#PBS -A NAML0001
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
CONFIG=$REPO/config/single_plane_fast.yml

mkdir -p /glade/derecho/scratch/schreck/holodec/single_plane_fast

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

python applications/trainer_unet_planer.py -c $CONFIG
