# BERT-Fine-Tuning
BERT fine-tuning for text classification 

First downoad the dataset using the commands:

!gdown --id 1S6qMioqPJjyBLpLVz4gmRTnJHnjitnuV
!gdown --id 1zdmewp7ayS4js4VtrJEHzAheSW-5NBZv

To run the code use this command:

!python BERT_finetunning/train.py \
  --data_path=./reviews.csv \
  --batch_size=32 \
  --epochs=2 \
  --num_classes=3 \
  --max_len=160 \
  --pretrained_bert_model_name=bert-base-cased \

