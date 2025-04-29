import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class MF(nn.Module):
  def __init__(self, num_users, num_items, embedding_k = 4, *args, **kwargs):
    super(MF, self).__init__()
    self.num_users = num_users
    self.num_items = num_items
    self.embedding_k = embedding_k
    self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
    self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

    self.sigmoid = torch.nn.Sigmoid()
    self.xent_func = torch.nn.BCELoss()

  def forward(self, x):
    user_idx = torch.LongTensor(x[:,0])
    item_idx = torch.LongTensor(x[:,1])
    U_emb = self.W(user_idx)
    V_emb = self.H(item_idx)

    out = self.sigmoid(torch.sum(U_emb.mul(V_emb),1)).squeeze()
    return out,U_emb,V_emb

  def predict(self, x):
    pred,_,_ = self.forward(x)
    pred = self.sigmoid(pred).squeeze()
    return pred.detach().numpy()
  
class NCF(nn.Module):
  def __init__(self, num_users, num_items, embedding_k = 4, *args, **kwargs):
    super(NCF, self).__init__()
    self.num_users = num_users
    self.num_items = num_items
    self.embedding_k = embedding_k
    self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
    self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
    self.linear1 = nn.Linear(self.embedding_k * 2, self.embedding_k)
    self.linear2 = nn.Linear(self.embedding_k, 1, bias = False)
    self.sigmoid = torch.nn.Sigmoid()
    self.relu = torch.nn.ReLU()
    self.xent_func = torch.nn.BCELoss()
  def forward(self, x):
    user_idx = torch.LongTensor(x[:,0])
    item_idx = torch.LongTensor(x[:,1])
    U_emb = self.W(user_idx)
    V_emb = self.H(item_idx)
    out = torch.cat((U_emb, V_emb), axis = 1)
    h1 = self.linear1(out)
    h1 = self.relu(h1)
    output = self.linear2(h1)
    output = self.sigmoid(output).squeeze()

    return output, U_emb, V_emb
    
  def predict(self, x):
    pred, _, _ = self.forward(x)
    pred = self.sigmoid(pred)
    pred = pred.squeeze()
    return pred.detach().numpy()    
    
    