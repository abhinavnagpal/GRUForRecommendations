!pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d chadgostopp/recsys-challenge-2015
!unzip /content/recsys-challenge-2015.zip
!rm /content/recsys-challenge-2015.zip
!rm kaggle.json

import numpy as np
import pandas as pd
import datetime

import torch
import lib
import os
from torch import nn

pd_tr = '/content/yoochoose-data/yoochoose-clicks.dat' 
pd_test = '/content/yoochoose-data/yoochoose-test.dat' 
pd_processed = '/content/data_processed/' 
time = 86400

def preprocess_sessions(pd):
    byLen = pd.groupby('SessionID').size() 
    k = np.in1d(pd.SessionID, byLen[byLen > 1].index)
    return pd[k]

def preprocess_items(pd):
  #delete records of items which appeared less than 5 times
  byItem = pd.groupby('ItemID').size() #groupby itemID and get size of each item
  k = np.in1d(pd.ItemID, byItem[byItem > 4].index)
  return pd[k]

data_train = pd.read_csv(pd_tr, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64}, names =  ['SessionID', 'Time', 'ItemID'] )
data_test = pd.read_csv(pd_test, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64}, names =  ['SessionID', 'Time', 'ItemID'] )

data_train['Time']= data_train.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) 
data_test['Time'] = data_test.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())

data_train = preprocess_sessions(data_train)
data_train = preprocess_items(data_train)
data_train = preprocess_sessions(data_train)

k = np.in1d(data_test['ItemID'], data_train['ItemID'])

data_test = data_test[k]
data_test = preprocess_sessions(data_test)
test.to_csv(pd_processed + 'Test_file.txt', sep=',', index=False)

session_time_max = data_train.groupby('SessionID').Time.max()
Train_data = sessionMaxTime[sessionMaxTime < (data_train.Time.max() - dayTime)].index 
Val_data = sessionMaxTime[sessionMaxTime >= (data_train.Time.max() - dayTime)].index

Train_data = data_train[np.in1d(data_train.SessionID, Train_data)]
Val_data = data_train[np.in1d(data_train.SessionID, Val_data)]

k=np.in1d(Train_data['ItemID'], Train_data['ItemID'])
Train_data = Train_data[k]
Val_data = preprocess_sessions(Val_data)
Train_data.to_csv(pd_processed + 'Train_file.txt', sep=',', index=False)
Val_data.to_csv(pd_processed + 'Val_file.txt', sep=',', index=False)

def main():
  train_data = Dataset(os.path.join(data_folder, train_data))
  valid_data = Dataset(os.path.join(data_folder, val_data), itemmap=train_data.itemmap)
  
  cuda = torch.cuda.is_available()
  data_folder =  '/content/data_processed/' 
  train_data =  'Train_file.txt'
  val_data =  'Val_file.txt'
  input = len(train_data.items)
  output = input_size
  hidden = 100
  num_layers = 3
  optimizer = 'Adagrad'
  lr = 0.05
  n_epochs = 10
  Eval = False
  model_load = False
  bs = 32
  dt_input = 0.5
  dt_hidden = 0.3
  activation = 'tanh'
  loss_type = 'TOP1-max'
  sort_by_time = True

  
      

  model = gru_model(input, hidden, output, activation=activation, use_cuda=cuda, bs=bs,
                      dt_input=dt_input, dt_hidden=dt_hidden)
                      
  # define loss function
  
  
  

class gru_model(nn.Module):
    def __init__(self, input, hidden, output, activation='tanh',
                 dt_hidden=0.5, dt_input=0, bs=50, use_cuda=False):
      
        super(GRU4REC, self).__init__()
        self.input = input
        self.hidden = hidden
        self.output = output
        self.num_layers = 1
        self.dt_hidden = dt_hidden
        self.dt_input = dt_input
        self.num_embedding_dim = -1
        self.bs = bs
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.emb_init = self.init_emb()
        self.h2o = nn.Linear(hidden, output)
        self.create_activation(activation)
        self.gru = nn.GRU(self.input, self.hidden, 1 , dt=self.dt_hidden)
        self = self.to(self.device)

    def create_activation(self, activation):
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input, hidden):

        if self.num_embedding_dim == -1:
            embed= self.encode(input)
            if self.dt_input > 0: 
              embedded = self.embed_dropout(embedded)
            embedded = embedded.unsqueeze(0)

        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logits = self.activation(self.h2o(output))

        return logits, hidden

    def init_emb(self):

        buffer = torch.FloatTensor(self.bs, self.output)
        buffer = buffer.to(self.device)
        return buffer

    def encode(self, input):

        id = input.view(-1, 1)
        buffer = self.buffer.scatter_(1, d, 1)
        return buffer