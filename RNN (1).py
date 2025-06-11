import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

device='cpu'

base_csv = 'IMDB_Dataset.csv'
df = pd.read_csv(base_csv)

X,y = df['review'].values,df['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                    if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label =='positive' else 0 for label in y_train]
    encoded_test = [1 if label =='positive' else 0 for label in y_val]
    return np.array(final_list_train,dtype=object), np.array(encoded_train),np.array(final_list_test,dtype=object), np.array(encoded_test),onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)
x_train_pad = padding_(x_train,100)
x_test_pad = padding_(x_test,100)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, hidden_size, output_size,n_layers,embedding_dim):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers
        self.rnn = nn.RNN(embedding_dim, hidden_size,n_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.embed=nn.Embedding(len(vocab)+1,embedding_dim=embedding_dim)
    def forward(self, x):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embed(x)
        # print(embeds.shape)
        # print(embeds[0])
        hidden=self.init_hidden(batch_size)

        # pass in the rnn layer
        out,hidden=self.rnn(embeds,hidden)
        # out=out.contiguous().view(-1,self.hidden_dim)
        # out=self.fc(out)
        
        output = self.sigmoid(self.fc(hidden.squeeze(0)))
        if self.n_layers==1:
            return output
        else:
            return output[-1]

    def init_hidden(self,batch_size):
        hidden=torch.zeros(self.n_layers,batch_size,self.hidden_size).to(device)
        return hidden

def train_acc(model):
    acc=0
    for instances,labels in train_loader_acc:
        
        output=model(instances.to(device))
        for i in range(len(output)):
            if output[i]>=0.5 and labels[i]==1:
                acc+=1
            elif output[i]<0.5 and labels[i]==0:
                acc+=1
    del instances
    del labels
    return acc/len(train_data)
def valid_acc(model):
    acc=0
    for instances,labels in valid_loader:
        
        output=model(instances.to(device))
        for i in range(len(output)):
            if output[i]>=0.5 and labels[i]==1:
                acc+=1
            elif output[i]<0.5 and labels[i]==0:
                acc+=1
    del instances
    del labels
    return acc/len(valid_data)
def train_loss(model,batch_size,criterion):
    loss=0
    for instances,labels in train_loader_acc:
        labels=labels.reshape(len(labels),-1).to(device).to(torch.float32)
        output=model(instances.to(device))
        loss+=criterion(output,labels)
    del instances
    del labels
    return loss/batch_size
def valid_loss(model,batch_size,criterion):
    loss=0
    for instances,labels in valid_loader:
        labels=labels.reshape(len(labels),-1).to(device).to(torch.float32)
        output=model(instances.to(device))
        loss+=criterion(output,labels)
    del instances
    del labels
    return loss/batch_size
    
hidden_size=100
output_size=1
n_layers=1
embedding_dim=64
model=RNN(hidden_size,output_size,n_layers,embedding_dim).to(device)
criterion =nn.BCELoss()
lr=0.0001
train_acc_list=[]
valid_acc_list=[]
loss_list=[]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
epochs=100
num_epochs=tqdm(range(epochs))

# n_epochs=tqdm(range(epochs))
for epoch in num_epochs:

    running_loss = 0
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    train_loader_acc=DataLoader(train_data, shuffle=False, batch_size=batch_size)
    
    for i,(instances, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        instances=instances.to(device)
        labels=labels.reshape(len(labels),-1).to(device).to(torch.float32)
        
        # print(instances.shape)
        output = model(instances)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_list.append(running_loss/batch_size)
    a=train_acc(model)
    train_acc_list.append(a)
    
    b=valid_acc(model)
    valid_acc_list.append(b)
    print('epoch is:',epoch+1,'accuracy over training/validation data set:',a,'/',b,' || running loss is:',running_loss/batch_size)
 
plt.plot(loss_list,label='loss in model',color='midnightblue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(train_acc_list,label='train_data',color='red')
plt.plot(valid_acc_list,label='valid_data',color='midnightblue')
plt.xlabel('epoch')
plt.ylabel('accruacy')
plt.legend()
plt.show()