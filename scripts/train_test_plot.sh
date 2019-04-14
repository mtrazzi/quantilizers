#!/bin/bash
DEFAULT_DATASETS=
ENV_NAME=Hopper-v2
DATASET_LIST=${1-"ryan ryan0904 jdonati0904"}
MODE=${2-full}
DATASET=""
#echo $DATASET_LIST | (read datasets;)
cd ..
for dataset in $DATASET_LIST
do 
	/usr/bin/python3 quantilizer.py --dataset_name $dataset --env_name=$ENV_NAME --mode $MODE
done