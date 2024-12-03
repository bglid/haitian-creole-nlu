# to train on the MCTest160 dataset - in English
python3 ./models/run_mbert.py --data_dir=./models/Data/MCTest160/ --output_dir=./models/outputs --learning_rate=1e-5 --weight_decay=0.01 --num_epochs=10 --action=train
