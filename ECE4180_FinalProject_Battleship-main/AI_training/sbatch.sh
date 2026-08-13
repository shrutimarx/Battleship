#!/bin/bash
#SBATCH --job-name=GAT_1
#SBATCH --output=/home/hice1/rsiddique32/4180/logs/%j.log
#SBATCH --error=/home/hice1/rsiddique32/4180/logs/%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=224GB
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00

mkdir -p /home/hice1/rsiddique32/4180/logs
source /home/hice1/rsiddique32/scratch/envs/4180/bin/activate
python3 /home/hice1/rsiddique32/4180/model.py
