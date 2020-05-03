#!/bin/bash

set -e

if [ -z "${PYTHON}" ]; then 
    # by default it should be run on an AWS server
    source activate pytorch_p36
    PYTHON='python'
fi

current_date="$(date +"%H:%M")"

$PYTHON train.py --scenario defend_the_center --logfile "${current_date}_center_initial" --total_episodes 500
$PYTHON train.py --scenario deadly_corridor --logfile "${current_date}_corridor_initial" --total_episodes 1000
$PYTHON train.py --scenario defend_the_center --logfile "${current_date}_center_retrain" --load_weights "weights/${current_date}_corridor_initial/1000.pth" --total_episodes 500
$PYTHON train.py --scenario deadly_corridor --logfile "${current_date}_corridor_retrain" --load_weights "weights/${current_date}_center_initial/500.pth" --total_episodes 1000