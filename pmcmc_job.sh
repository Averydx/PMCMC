#!/bin/bash
#SBATCH --job-name=PMCMC_Movement
#SBATCH --output=./out/PMCMC_Log.txt
#SBATCH --time=1:00
#SBATCH --mem=192
#SBATCH --cpus-per-task=1
set -euo pipefail

# Run a Python script given as the first argument
if [ $# -eq 0 ]; then
    echo "Error: No Python script provided."
    echo "Usage: $0 <script.py>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: File not found: $1"
    echo "Usage: $0 <script.py>"
    exit 1
fi

source .venv/bin/activate

env_file=.env
if [ -f "$env_file" ]; then
    export $(cat "$env_file" | xargs)
    echo "loaded $env_file"
fi

echo "starting job: $(pwd)/$1"
echo "which python: $(which python) ($(python -V))"

python "$1"

deactivate
echo "job complete"