#!/bin/bash
# to train on the MCTest160 dataset - using mBert
python3 ./models/run_mbert.py --data_dir=./models/Data/MCTest160/ --output_dir=./models/outputs/MBERT160/ --learning_rate=1e-5 --weight_decay=0.017 --num_epochs=20 --action=train
# to train on MCTest160 - using XLMR
python3 ./models/run_xlmr.py --data_dir=./models/Data/MCTest160/ --output_dir=./models/outputs/XLMR160/ --learning_rate=1e-5 --weight_decay=0.008 --num_epochs=20 --action=train
# to train on MCTest160 - using Flaubert
python3 ./models/run_flaubert.py --data_dir=./models/Data/MCTest160/ --output_dir=./models/outputs/FLAUBERT/ --learning_rate=1e-5 --weight_decay=0.008 --num_epochs=20 --action=train
