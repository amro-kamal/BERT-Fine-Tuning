from torch import nn
from transformers import BertModel

class BertClassfier(nn.Module):

  def __init__(self, pretrained_bert_model_name, num_classes):
    super(BertClassfier, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_bert_model_name)
    self.dropout = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, num_classes)    #763 x n_classes
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    x = self.dropout(pooled_output)
    return self.out(x)