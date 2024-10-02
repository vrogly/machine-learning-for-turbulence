#!/usr/bin/env python
# coding: utf-8

# # Setup 🏗️
# 

# In[1]:


import numpy as np
import torch 
import sys 
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load

plt.rcParams.update({'font.size': 22})
plt.interactive(True)
plt.close('all')


init_time = time.time()

# load DNS data
vel_DNS=np.genfromtxt("vel_11000_DNS_no-text.dat", dtype=None,comments="%")

# % Wall-normal profiles:
# y/\delta_{99}       y+          U+          urms+       vrms+       wrms+       uv+         prms+       pu+         pv+         S(u)        F(u)        dU+/dy+     V+


y_DNS=vel_DNS[:,0]
yplus_DNS=vel_DNS[:,1]
u_DNS=vel_DNS[:,2]
uu_DNS=vel_DNS[:,3]**2
vv_DNS=vel_DNS[:,4]**2
ww_DNS=vel_DNS[:,5]**2
uv_DNS=vel_DNS[:,6]


dudy_DNS  = np.gradient(u_DNS,yplus_DNS)


k_DNS  = 0.5*(uu_DNS+vv_DNS+ww_DNS)

# y/d99           y+              Produc.         Advect.         Tur. flux       Pres. flux      Dissip
#DNS_RSTE = np.genfromtxt("/chalmers/users/lada/DNS-boundary-layers-jimenez/balances_6500_Re_theta.6500.bal.uu.txt",comments="%")
#
#prod_DNS = -DNS_RSTE[:,2]*3/2 #multiply by 3/2 to get P^k from P_11
#eps_DNS = -DNS_RSTE[:,6]*3/2  #multiply by 3/2 to get eps from eps_11
#yplus_DNS_uu = DNS_RSTE[:,1]

DNS_RSTE = np.genfromtxt("bud_11000.prof",comments="%")

eps_DNS = -DNS_RSTE[:,4]
yplus_DNS_uu = yplus_DNS


# fix wall
eps_DNS[0]=eps_DNS[1]


#-----------------Data_manipulation--------------------


# choose values for 30 < y+ < 1000
#index_choose=np.nonzero((yplus_DNS > 30 )  & (yplus_DNS< 1000 ))
index_choose=np.nonzero((yplus_DNS > 9 )  & (yplus_DNS< 2200 ))

# set a min on dudy
# set a min on dudy
dudy_DNS = np.maximum(dudy_DNS,4e-4)


uv_DNS    =  uv_DNS[index_choose]
uu_DNS    =  uu_DNS[index_choose]
vv_DNS    =  vv_DNS[index_choose]
ww_DNS    =  ww_DNS[index_choose]
k_DNS     =  k_DNS[index_choose]
eps_DNS   =  eps_DNS[index_choose]
dudy_DNS  =  dudy_DNS[index_choose]
yplus_DNS =  yplus_DNS[index_choose]
y_DNS     =  y_DNS[index_choose]
u_DNS     =  u_DNS[index_choose]

# Calculate ny_t and time-scale tau
viscous_t = k_DNS**2/eps_DNS 
# tau       = viscous_t/abs(uv_DNS)
#DNS

dudy_DNS_org = np.copy(dudy_DNS)

tau_DNS = k_DNS/eps_DNS

# make dudy non-dimensional
#dudy_DNS = dudy_DNS*tau_DNS
#tau_DNS = np.ones(len(dudy_DNS))

# Calculate c_0 & c_2 of the Non-linear Eddy Viscosity Model

a11_DNS=uu_DNS/k_DNS-0.66666
a22_DNS=vv_DNS/k_DNS-0.66666
a33_DNS=ww_DNS/k_DNS-0.66666

c_2_DNS=(2*a11_DNS+a33_DNS)/tau_DNS**2/dudy_DNS**2
c_0_DNS=-6*a33_DNS/tau_DNS**2/dudy_DNS**2

c = np.array([c_0_DNS,c_2_DNS])


########################## 2*a11_DNS+a33_DNS
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(2*a11_DNS+a33_DNS,yplus_DNS, marker="o", s=10, c="red", label="Neural Network")
plt.xlabel("$2a_{11}+a_{33}$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('2a11_DNS+a33_DNS-dudy2-and-tau-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


prod_DNS_1 = -uv_DNS*dudy_DNS


########################## k-bal
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
#ax1.plot(yplus_DNS_uu,prod_DNS, 'b-', label="prod")
ax1.plot(yplus_DNS,prod_DNS_1, 'b-', label="$-\overline{u'v'} \partial U/\partial y$")
ax1.plot(yplus_DNS,eps_DNS,'r--', label="diss")
plt.axis([0,200,0,0.3])
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('prod-diss-DNS-dudy2-and-tau-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-ustar-and-nu-BL.png')




# transpose the target vector to make it a column vector  
y = c.transpose()

dudy_squared_DNS = (dudy_DNS**2)
# scale with k and eps 
# dudy [1/T]
# dudy**2 [1/T**2]
T = tau_DNS
dudy_squared_DNS_scaled = dudy_squared_DNS*T**2
dudy_DNS_inv = 1/dudy_DNS/T
# re-shape
dudy_squared_DNS_scaled = dudy_squared_DNS_scaled.reshape(-1,1)
dudy_DNS_inv_scaled = dudy_DNS_inv.reshape(-1,1)
# use MinMax scaler
#scaler_dudy2 = StandardScaler()
#scaler_tau = StandardScaler()
scaler_dudy2 = MinMaxScaler()
scaler_dudy = MinMaxScaler()
X=np.zeros((len(dudy_DNS),2))
X[:,0] = scaler_dudy2.fit_transform(dudy_squared_DNS_scaled)[:,0]
X[:,1] = scaler_dudy.fit_transform(dudy_DNS_inv_scaled)[:,0]


# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

random_state = randrange(100)

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices,test_size=0.2,shuffle=True,random_state=42)

# create text index 
#index= np.arange(0,len(X), dtype=int)
## pick every 5th elements 
#index_test=index[::5]
#
#X_test = X[::5]
#y_test = y[::5]
#
## pick every element except every 5th
#index_train = np.array([i for i in range(len(X)) if i%5!=0])
#
#X_train = X[index_train]
#y_train = y[index_train]



dudy_DNS_train = dudy_DNS[index_train]
dudy_DNS_inv_train = dudy_DNS_inv[index_train]
k_DNS_train = k_DNS[index_train]
uu_DNS_train = uu_DNS[index_train]
vv_DNS_train = vv_DNS[index_train]
ww_DNS_train = ww_DNS[index_train]
yplus_DNS_train = yplus_DNS[index_train]
c0_DNS_train = c_0_DNS[index_train]
c2_DNS_train = c_2_DNS[index_train]
tau_DNS_train = tau_DNS[index_train]

dudy_DNS_test = dudy_DNS[index_test]
dudy_DNS_inv_test = dudy_DNS_inv[index_test]
k_DNS_test = k_DNS[index_test]
uu_DNS_test = uu_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
ww_DNS_test = ww_DNS[index_test]
yplus_DNS_test = yplus_DNS[index_test]
c0_DNS_test = c_0_DNS[index_test]
c2_DNS_test = c_2_DNS[index_test]
tau_DNS_test = tau_DNS[index_test]


# Set up hyperparameters
learning_rate = 1e-1
learning_rate = 5.e-1  
#learning_rate = 10e-1
#learning_rate = 0.9
#learning_rate = 0.1
#learning_rate = 0.2
#learning_rate = 0.2
my_batch_size = 5
#my_batch_size = 30
#my_batch_size = 5
#my_batch_size = 3
epochs = 10000
epochs = 40000
#epochs = 30

# convert the numpy arrays to PyTorch tensors with float32 data type
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create PyTorch datasets and dataloaders for the training and validation sets
# a TensorDataset wraps the feature and target tensors into a single dataset
# a DataLoader loads the data in batches and shuffles the batches if shuffle=True
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=my_batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=my_batch_size)

# show dataset
print('len(test_dataset)',len(test_dataset))
print('len(train_dataset)',len(train_dataset))
print('train_dataset[0:2]',train_dataset[0:2])
#
# show dataloader
k=5
#train_k = next(itertools.islice(train_loader, k, None))
train_k_from_dataset = train_dataset[k]
print ('X_train[5]',X_train[k])
print ('train_k_from_dataset',train_k_from_dataset)
print ('y_train[5]',y_train[k])

#test_k = next(itertools.islice(test_loader, k, None))
test_k_from_dataset = test_dataset[k]
print ('X_test[5]',X_test[k])
print ('test_k_from_dataset',test_k_from_dataset)
print ('y_test[5]',y_test[k])

# In[4]:

########################## check
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
Nx=len(X_train)
Nx=5
for k in range(0,Nx):
# train_k = next(itertools.islice(train_loader, k, None))
  train_k = train_dataset[k]
  k_train = index_train[k]
  yplus = yplus_DNS[k_train]
# print ('k,k_train,c_0_train,yplus',k,k_train,train_k[1][0][0],yplus)
  print ('k,k_train,c_0_train,yplus',k,k_train,train_dataset[k][1][0],yplus)
  if k == 0: 
     plt.plot(c_0_DNS[k_train],yplus, 'ro',label='target')
     plt.plot(train_dataset[k][1][0],yplus, 'b+',label='trained')
  else:
     plt.plot(c_0_DNS[k_train],yplus, 'ro')
     plt.plot(train_dataset[k][1][0],yplus, 'b+')
plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('c0-and-cNN-train-and-test-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-ustar-and-nu-BL.png')

# Let's set up a neural network:

class ThePredictionMachine(nn.Module):

    def __init__(self):
        
        super(ThePredictionMachine, self).__init__()

        self.input   = nn.Linear(2, 50)
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 2)

#       self.input   = nn.Linear(2, 50)
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 25)
#       self.hidden3 = nn.Linear(25, 2)

#       self.input   = nn.Linear(2, 50)
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 50)
#       self.hidden3 = nn.Linear(50, 25)
#       self.hidden4 = nn.Linear(25, 2)

#       self.input   = nn.Linear(2, 50)
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 50)
#       self.hidden3 = nn.Linear(50, 50)
#       self.hidden4 = nn.Linear(50, 25)
#       self.hidden5 = nn.Linear(25, 2)

#       self.input   = nn.Linear(2, 50)
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 50)
#       self.hidden3 = nn.Linear(50, 50)
#       self.hidden4 = nn.Linear(50, 50)
#       self.hidden5 = nn.Linear(50, 25)
#       self.hidden6 = nn.Linear(25, 2)


#       self.input   = nn.Linear(2, 50)
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 50)
#       self.hidden3 = nn.Linear(50, 50)
#       self.hidden4 = nn.Linear(50, 50)
#       self.hidden5 = nn.Linear(50, 50)
#       self.hidden6 = nn.Linear(50, 50)
#       self.hidden7 = nn.Linear(50, 25)
#       self.hidden8 = nn.Linear(25, 2)



    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.hidden1(x))
        x = self.hidden2(x)

#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = self.hidden3(x)

#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = nn.functional.relu(self.hidden3(x))
#       x = self.hidden4(x)

#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = nn.functional.relu(self.hidden3(x))
#       x = nn.functional.relu(self.hidden4(x))
#       x = self.hidden5(x)

#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = nn.functional.relu(self.hidden3(x))
#       x = nn.functional.relu(self.hidden4(x))
#       x = nn.functional.relu(self.hidden5(x))
#       x = self.hidden6(x)

#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = nn.functional.relu(self.hidden3(x))
#       x = nn.functional.relu(self.hidden4(x))
#       x = nn.functional.relu(self.hidden5(x))
#       x = nn.functional.relu(self.hidden6(x))
#       x = nn.functional.relu(self.hidden7(x))
#       x = self.hidden8(x)

        return x

# In[6]:


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print('in train_loop: len(dataloader)',len(dataloader))
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
#       optimizer.zero_grad()
# https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        optimizer.zero_grad(None)
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    global pred_numpy,pred1,size1
    size = len(dataloader.dataset)
    size1 = size
    num_batches = len(dataloader)
    test_loss = 0
    print('in test_loop: len(dataloader)',len(dataloader))

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
#transform from tensor to numpy
            pred_numpy = pred.detach().numpy()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>.2e} \n")

    return test_loss

start_time = time.time()

# Instantiate a neural network
neural_net = ThePredictionMachine()

# Initialize the loss function
loss_fn = nn.MSELoss()

# Choose loss function, check out https://pytorch.org/docs/stable/optim.html for more info
# In this case we choose Stocastic Gradient Descent
optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, neural_net, loss_fn, optimizer)
    test_loss = test_loop(test_loader, neural_net, loss_fn)
print("Done!")

preds = neural_net(X_test_tensor)

print(f"{'time ML: '}{time.time()-start_time:.2e}")

#transform from tensor to numpy
c_NN = preds.detach().numpy()
 
c_NN_old = c_NN

c0=c_NN[:,0]
c2=c_NN[:,1]

a_11 = 1/12*tau_DNS_test**2*dudy_DNS_test**2*(c0 + 6*c2)
uu_NN = (a_11+0.6666)*k_DNS_test

#a_{22} = \frac{1}{12} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 - 6c_2 + c_3)
a_22 = 1/12*tau_DNS_test**2*dudy_DNS_test**2*(c0 - 6*c2)
vv_NN = (a_22+0.6666)*k_DNS_test

# a_{33} = -\frac{1}{6} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 + c_3)
a_33 = -1/6*tau_DNS_test**2*dudy_DNS_test**2*c0
ww_NN = (a_33+0.6666)*k_DNS_test

c0_std=np.std(c0-c0_DNS_test)/(np.mean(c0.flatten()**2))**0.5
c2_std=np.std(c2-c2_DNS_test)/(np.mean(c2.flatten()**2))**0.5

print('\nc0_error_std',c0_std)
print('\nc2_error_std',c2_std)

np.savetxt('error-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.txt', [test_loss,c0_std,c2_std] )

filename = 'model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.pth'
torch.save(neural_net, filename)
dump(scaler_dudy2,'model-channel-DNS-dudy-and-dudy2_scaler-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')
dump(scaler_dudy,'model-channel-DNS-dudy-and-dudy2_scaler-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')

dudy2_max = np.max(dudy_squared_DNS)
dudy2_min = np.min(dudy_squared_DNS)
dudy_min = np.min(dudy_DNS)
dudy_max = np.max(dudy_DNS)
c0_min = np.min(c0)
c0_max = np.max(c0)
c2_min = np.min(c2)
c2_max = np.max(c2)

np.savetxt('min-max-model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.6-scale-with-k-eps-units-BL.txt', [dudy2_min, dudy2_max, dudy_min, dudy_max, c0_min, c0_max, c2_min, c2_max] )



########################## c0
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for k in range(0,len(X_test)):
  k_test = index_test[k]
  yplus = yplus_DNS[k_test]
  if k == 0: 
     plt.plot(c_0_DNS[k_test],yplus, 'bo',label='target')
     plt.plot(c0[k],yplus, 'r+',label='NN')
  else:
     plt.plot(c_0_DNS[k_test],yplus, 'bo')
     plt.plot(c0[k],yplus, 'r+')
plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('c0-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


########################## c0 v dudy**2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
dudy2_inverted=scaler_dudy2.inverse_transform(X_test)
for k in range(0,len(X_test)):
  if k == 0:
     plt.plot(c_0_DNS[index_test[k]],dudy_DNS[index_test[k]]**2, 'bo',label='target')
     plt.plot(c0[k],dudy2_inverted[k,0], 'r+',label='NN')
  else:
     plt.plot(c_0_DNS[index_test[k]],dudy_DNS[index_test[k]]**2,'bo')
     plt.plot(c0[k],dudy2_inverted[k,0], 'r+')
plt.xlabel("$c_0$")
plt.ylabel(r"$\left(\partial U/\partial y\right)^2$")
plt.legend(loc="best",fontsize=12)
plt.savefig('c0-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


########################## c2 v dudy**2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for k in range(0,len(X_test)):
  if k == 0:
     plt.plot(c_2_DNS[index_test[k]],dudy_DNS[index_test[k]]**2, 'bo',label='target')
     plt.plot(c_NN[k,1],dudy_DNS[index_test[k]]**2, 'r+',label='NN')
  else:
     plt.plot(c_2_DNS[index_test[k]],dudy_DNS[index_test[k]]**2,'bo')
     plt.plot(c_NN[k,1],dudy_DNS[index_test[k]]**2, 'r+')
plt.xlabel("$c_2$")
plt.ylabel(r"$\left(\partial U/\partial y\right)^2$")
plt.legend(loc="best",fontsize=12)
plt.savefig('c2-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


########################## c2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for k in range(0,len(X_test)):
  k_test = index_test[k]
  yplus = yplus_DNS[k_test]
  if k == 0: 
     plt.plot(c_2_DNS[k_test],yplus, 'bo',label='target')
     plt.plot(c_NN[k,1],yplus, 'r+',label='NN')
  else:
     plt.plot(c_2_DNS[k_test],yplus, 'bo')
     plt.plot(c_NN[k,1],yplus, 'r+')
# ax4.axis([-2000, 0, 0,5000])
# ax5.axis([-2000, 0, 0,5000])
plt.xlabel("$c_2$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('c2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')




########################## uu
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(uu_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(uu_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('uu-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


########################## vv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(vv_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(vv_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{v'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('vv-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

########################## ww
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(ww_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(ww_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{w'w'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('ww-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

########################## time scales
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.plot(dudy_DNS_org,yplus_DNS,'r-', label=r"$dudy$")
ax1.plot(1/dudy_DNS_org,yplus_DNS,'b-', label=r"$\left(\partial U/\partial y\right)^{-1}$")
plt.xlabel("time scsles")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig('time-scales-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

########################## time scales
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.plot(dudy_DNS_org*dudy_DNS_org,yplus_DNS,'b-')
plt.xlabel(r"$\left(\partial U/\partial y\right)^{-1} dudy$")
plt.ylabel("$y^+$")
plt.savefig('dudy-times-dudy-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

print(f"{'total time: '}{time.time()-init_time:.2e}")


