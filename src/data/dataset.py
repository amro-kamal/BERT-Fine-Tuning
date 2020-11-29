from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torch

class BertDataset(Dataset):

  def __init__(self, reviews, labels, max_len, PRE_TRAINED_MODEL_NAME):
    self.reviews = reviews
    self.labels = labels
    self.max_len = max_len
    self.PRE_TRAINED_MODEL_NAME=PRE_TRAINED_MODEL_NAME
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    label = self.labels[item]

    tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    encoding =  tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(label, dtype=torch.long)
    }

