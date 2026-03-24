#!/bin/bash
# Train R2Dreamer on the vineyard active reconstruction task.
# Usage:  bash runs/vineyard.sh [extra hydra overrides]

python train.py env=vineyard model=size12M "$@"
