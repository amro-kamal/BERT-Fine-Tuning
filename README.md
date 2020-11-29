# BERT-Fine-Tuning
BERT fine-tuning for text classification 

First clone this repo and downoad the dataset using this command:

```
!gdown --id 1zdmewp7ayS4js4VtrJEHzAheSW-5NBZv
```
This command will download Google Apps reviews dataset as a .csv file. You can use any dataset you want, but don't forget to modify the `read_data_from_path()` function in the `utils.py` file.

To run the code use this command:

````
!python BERT_finetunning/train.py \`
  --data_path=./reviews.csv \
  --batch_size=32 \
  --epochs=2 \
  --num_classes=3 \
  --max_len=160 \
  --pretrained_bert_model_name=bert-base-cased \
  
 ````

