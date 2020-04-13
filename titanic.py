import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df_labelled = pd.read_csv('train.csv')
df_unlabelled  = pd.read_csv('test.csv')
df_sub   = pd.read_csv('gender_submission.csv')

df_labelled.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df_unlabelled.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)

sex = pd.get_dummies(df_labelled['Sex'],drop_first=True)
embark = pd.get_dummies(df_labelled['Embarked'],drop_first=True)
df_labelled = pd.concat([df_labelled,sex,embark],axis=1)
df_labelled.drop(['Sex','Embarked'],axis=1,inplace=True)

sex = pd.get_dummies(df_unlabelled['Sex'],drop_first=True)
embark = pd.get_dummies(df_unlabelled['Embarked'],drop_first=True)
df_unlabelled = pd.concat([df_unlabelled,sex,embark],axis=1)
df_unlabelled.drop(['Sex','Embarked'],axis=1,inplace=True)

df_labelled.fillna(df_labelled.mean(),inplace=True)
df_unlabelled.fillna(df_unlabelled.mean(),inplace=True)

Scaler1 = StandardScaler()
Scaler2 = StandardScaler()

labelled_columns = df_labelled.columns
unlabelled_columns  = df_unlabelled.columns

labels = df_labelled.loc[:, "Survived"]
df_labelled.drop(['Survived'],axis=1,inplace=True)

df_labelled = pd.DataFrame(Scaler1.fit_transform(df_labelled))
df_unlabelled  = pd.DataFrame(Scaler2.fit_transform(df_unlabelled))

df_labelled.columns = unlabelled_columns
df_unlabelled.columns  = unlabelled_columns

df_labelled = pd.concat([df_labelled,labels],axis=1)


features = df_labelled.iloc[:,:-1].columns.tolist()
target   = df_labelled.loc[:, 'Survived'].name

# train/validation split
df_train = df_labelled.iloc[:701, :]
df_valid = df_labelled.iloc[701:, :]

y_train = df_train.loc[:, "Survived"].values
df_train.drop(["Survived"], axis=1, inplace=True)
X_train = df_train.values

y_valid = df_valid.loc[:, "Survived"].values
df_valid.drop(["Survived"], axis=1, inplace=True)
X_valid = df_valid.values

class Net(nn.Module):
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


dropouts = [0, 0.1, 0.2, 0.3]
n_epochss = [600, 1000, 1600, 2000, 3000]

best_accuracy = 0

for n_epochs in n_epochss:
    for dropout in dropouts:
        for wd in range(3):
            wd = 10**np.random.uniform(low=-5, high=-1)
            for lr in range(4):
               lr = 10**np.random.uniform(low=-3, high=-1)

               model = Net(dropout)
               print(model)
               
               criterion = nn.CrossEntropyLoss()
               optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
               
               
               batch_size = 64
               batch_no = len(X_train) // batch_size
               
               train_loss = 0
               train_loss_min = np.Inf
               for epoch in range(n_epochs):
                   for i in range(batch_no):
                       start = i*batch_size
                       end = start+batch_size
                       x_var = Variable(torch.FloatTensor(X_train[start:end]))
                       y_var = Variable(torch.LongTensor(y_train[start:end]))
               
                       optimizer.zero_grad()
                       output = model(x_var)
                       loss = criterion(output,y_var)
                       loss.backward()
                       optimizer.step()
               
                       values, labels = torch.max(output, 1)
                       num_right = np.sum(labels.data.numpy() == y_train[start:end])
                       train_loss += loss.item()*batch_size
               
                   train_loss = train_loss / len(X_train)
                   if train_loss <= train_loss_min:
                       #print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
                       torch.save(model.state_dict(), "model.pt")
                       train_loss_min = train_loss
               
                   if epoch % 200 == 0:
                       print('')
                       print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y_train[start:end]) ))
               print('Training Ended! ')
               
               line = " lr: " + str(lr)
               line += " wd: "+ str(wd) 
               line += " epochs: " + str(n_epochs)
               line += " dropout: " + str(dropout)
               print(line)

               x_var = Variable(torch.FloatTensor(X_valid))
               y_var = Variable(torch.LongTensor(y_valid))
               output = model(x_var)
               values, labels = torch.max(output, 1)
               num_right = np.sum(labels.data.numpy() == y_valid)
               valid_accuracy = num_right/len(y_valid)
               print("Validaion Accuracy: ", valid_accuracy)
               if valid_accuracy > best_accuracy:
                   torch.save(model.state_dict(), "best_model.pt")
                   best_accuracy = valid_accuracy
                   with open("parameters.txt", "a") as f:
                       line = "Accuracy: " + str(best_accuracy)
                       line += " lr: " + str(lr)
                       line += " wd: "+ str(wd) 
                       line += " epochs: " + str(n_epochs)
                       line += " dropout: " + str(dropout) + "\n"
                       f.write(line)
                       
                       X_predict = df_unlabelled.values
                       X_predict_var = Variable(torch.FloatTensor(X_predict), requires_grad=False) 
                       with torch.no_grad():
                           predict_result = model(X_predict_var)
                           values, labels = torch.max(predict_result, 1)
                           survived = labels.data.numpy()

                           submission = pd.DataFrame({'PassengerId': df_sub['PassengerId'], 'Survived': survived})
                           submission.to_csv('submission.csv', index=False)
