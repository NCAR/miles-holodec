#!/bin/bash
### AMP smoke test: run 2 training batches then inference on 1 hologram
#PBS -N holo_amp_test
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=64GB:ngpus=1:gpu_type=a100
#PBS -j oe
#PBS -k eod
#PBS -M jsschreck@gmail.com

set -e

export TMPDIR=/glade/derecho/scratch/schreck/temp
mkdir -p $TMPDIR

REPO=/glade/work/schreck/repos/miles-holodec
OUT=/glade/derecho/scratch/schreck/holodec/amp_test

mkdir -p $OUT

module load conda
conda activate /glade/work/schreck/conda-envs/holodec

cd $REPO

# Write a minimal config with amp=true, only 2 batches/epoch, 1 epoch
python - << 'PYEOF'
import yaml, copy

with open("config/multiplane.yml") as f:
    conf = yaml.safe_load(f)

conf["save_loc"] = "/glade/derecho/scratch/schreck/holodec/amp_test"
conf["trainer"]["batches_per_epoch"] = 2
conf["trainer"]["valid_batches_per_epoch"] = 2
conf["trainer"]["epochs"] = 1
conf["trainer"]["stopping_patience"] = 999
conf["trainer"]["amp"] = True

with open("/glade/derecho/scratch/schreck/holodec/amp_test/amp_test.yml", "w") as f:
    yaml.dump(conf, f)

print("Config written.")
PYEOF

echo "=== Training (1 epoch, 2 batches, AMP=True) ==="
python applications/trainer_unet.py \
    -c /glade/derecho/scratch/schreck/holodec/amp_test/amp_test.yml

echo "=== Inference (1 hologram, AMP not used in inference) ==="
python applications/inference_multiplane.py \
    -c config/multiplane.yml \
    -k /glade/derecho/scratch/schreck/holodec/amp_test/checkpoint.pt \
    -o $OUT/inference \
    --max-holograms 1 \
    --z-stride 5 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda

echo "=== AMP test passed ==="
