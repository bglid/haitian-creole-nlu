#!/bin/bash
# Evaluating MBert on Creole Dev sets
python3 run_mbert.py --data_dir=./models/Data/MCTestHat1/ --output_dir=./models/outputs/MBERT160/ --learning_rate=1e-5 --weight_decay=0.01 --num_epochs=10 --action=evaluate --from_checkpoint=./models/outputs/MBERT160/
python3 run_mbert.py --data_dir=./models/Data/MCTestHat2/ --output_dir=./models/outputs/MBERT160/ --learning_rate=1e-5 --weight_decay=0.01 --num_epochs=10 --action=evaluate --from_checkpoint=./models/outputs/MBERT160/

# XLMR
python3 run_xlmr.py --data_dir=./models/Data/MCTestHat1/ --output_dir=./models/outputs/XLMR160/ --learning_rate=1e-5 --weight_decay=0.01 --num_epochs=10 --action=evaluate --from_checkpoint=./models/outputs/XLMR160/
python3 run_xlmr.py --data_dir=./models/Data/MCTestHat2/ --output_dir=./models/outputs/XLMR160/ --learning_rate=1e-5 --weight_decay=0.01 --num_epochs=10 --action=evaluate --from_checkpoint=./models/outputs/XLMR160/
