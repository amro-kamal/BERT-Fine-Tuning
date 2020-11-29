import argparse
import numpy as np
import pandas as pd
import torch
from  torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
from utils import to_sentiment, read_data_from_path
from model import BertClassfier
from data import BertDataset




def train_epoch(
  model, 
  data_loader,
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:

    if device.type=='cuda':
        #move the data to GPU
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["targets"].to(device)

    #predict
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    ##compute the loss
    # if loss_fn=='triplet_loss':
    #   _, preds = torch.max(outputs, dim=1)
    #   loss=batch_hard_triplet_loss(labels, outputs, margin=0.2)

    # else  :
    _, preds = torch.max(outputs, dim=1)

    # loss_fn= nn.CrossEntropyLoss().to(device)
    loss = loss_fn(outputs, labels)
    # loss=batch_hard_triplet_loss(targets, outputs, margin=0.2,device=device)

    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples
  # , np.mean(losses)

####################################################################
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples
#   , np.mean(losses)

  ######################################################################

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--data_path", required=True, type=str, help="train dataset for train bert")


    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--pretrained_bert_model_name", required=True, type=str, help="pretrained bert model")
    parser.add_argument("--num_classes", required=True, type=int, help="number of classes in the data")
    parser.add_argument("--max_len", required=True, type=int, help="maximum sequence length")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

#################################
    # read the data
    train_reviews, train_labels, val_reviews, val_labels, test_reviews, test_labels=read_data_from_path(args.data_path)

    print("Loading  Training  Dataset")
    
    train_ds = BertDataset(
    reviews=train_reviews,
    labels=train_labels,
    max_len=args.max_len,
    PRE_TRAINED_MODEL_NAME=args.pretrained_bert_model_name
    )
    train_dataloader = DataLoader( train_ds, args.batch_size, num_workers=5)
##########################################################################
    print("Loading val Dataset")
    
    val_ds = BertDataset(
    reviews=val_reviews,
    labels=val_labels,
    max_len=args.max_len,
    PRE_TRAINED_MODEL_NAME=args.pretrained_bert_model_name
    )
    val_dataloader = DataLoader( val_ds, args.batch_size, num_workers=5)



    print("Creating BERT Model")

    if args.with_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print ('you are using {} : {}'.format("GPU" if device.type=="cuda" else device , torch.cuda.get_device_name(0)))
        model=BertClassfier(pretrained_bert_model_name=args.pretrained_bert_model_name, num_classes=args.num_classes).to(device)
    else:
        model=BertClassfier(pretrained_bert_model_name=args.pretrained_bert_model_name, num_classes=args.num_classes)


    print("Start Training ")

    
    history = defaultdict(list)
    best_accuracy = 0
    loss_fn = nn.CrossEntropyLoss().to(device)
    #loss_fn=batch_hard_triplet_loss(targets, outputs, margin=0.2,device=device)


    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    EPOCHS=args.epochs
    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    for epoch in range(args.epochs):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc = train_epoch(
        model,
        train_dataloader,     
        loss_fn,
        optimizer, 
        device, 
        scheduler, 
        train_reviews.size
        )
        train_loss=0
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc = eval_model(
        model,
        val_dataloader,
        loss_fn,
        device, 
        val_reviews.size
        )
        
        val_loss=0
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            # torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

#main
if __name__ == "__main__":
    train()
  
