#!/bin/bash
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     PYTHON3="/usr/bin/python3";;
    Darwin*)    PYTHON3="/usr/local/bin/python3";;
esac
ENV_NAME=Hopper-v2
DATASET_LIST=${1-"ryan ryan0904 jdonati0904"}
MODE=${2-full}
DATASET=""
FRAMEWORK=${3-"keras"}
SEED_MIN=${4-0}
SEED_NB=${5-1}
cd ..
for dataset in $DATASET_LIST
do
	${PYTHON3} quantilizer.py --dataset_name $dataset --env_name=$ENV_NAME --mode $MODE  --seed_min $SEED_MIN --seed_nb $SEED_NB
done
