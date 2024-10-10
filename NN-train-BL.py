
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch 
import sys 
import time
import os
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

trainset = 'BL' #CF5200
valset = 'CF5200' #BL

# Add unique run ID here
savedir = "renders/run1/"
os.makedirs(os.path.dirname(savedir), exist_ok=True)

init_time = time.time()

# Load data for relevant y+ interval
def loaddict(data_set,yplusmin,yplusmax):
   dict_temp = {}
   if data_set == 'BL':
      vel_DNS=np.genfromtxt("datasets/vel_11000_DNS.dat", dtype=None,comments="%")
      DNS_RSTE = np.genfromtxt("datasets/bud_11000.prof",comments="%")

      dict_temp["y"]=vel_DNS[:,0]
      dict_temp["yplus"]=vel_DNS[:,1]
      dict_temp["u"]=vel_DNS[:,2]
      dict_temp["uu"]=vel_DNS[:,3]**2
      dict_temp["vv"]=vel_DNS[:,4]**2
      dict_temp["ww"]=vel_DNS[:,5]**2
      dict_temp["uv"]=vel_DNS[:,6]
      dict_temp["k"]  = 0.5*(dict_temp["uu"]+dict_temp["vv"]+dict_temp["ww"])

      dict_temp["eps"] = -DNS_RSTE[:,4]
      dict_temp["eps"][0]=dict_temp["eps"][1]

   # y/d99           y+              Produc.         Advect.         Tur. flux       Pres. flux      Dissip
   #DNS_RSTE = np.genfromtxt("/chalmers/users/lada/DNS-boundary-layers-jimenez/balances_6500_Re_theta.6500.bal.uu.txt",comments="%")
   #
   #prod_DNS = -DNS_RSTE[:,2]*3/2 #multiply by 3/2 to get P^k from P_11
   #eps_DNS = -DNS_RSTE[:,6]*3/2  #multiply by 3/2 to get eps from eps_11
   #yplus_DNS_uu = DNS_RSTE[:,1]
      #yplus_DNS_uu = yplus_DNS

   elif data_set == 'CF5200':
      # load DNS channel data
      DNS_mean  = np.genfromtxt("datasets/LM_Channel_5200_mean_prof.dat",comments="%").transpose()
      DNS_stress = np.genfromtxt("datasets/LM_Channel_5200_vel_fluc_prof.dat",comments="%").transpose()
      DNS_RSTE = np.genfromtxt("datasets/LM_Channel_5200_RSTE_k_prof.dat",comments="%")

      # Load mean file
      dict_temp["y"]     = DNS_mean[0]
      dict_temp["yplus"] = DNS_mean[1]
      dict_temp["u"]     = DNS_mean[2]
      # Load fluc file
      dict_temp["uu"] = DNS_stress[2]
      dict_temp["vv"] = DNS_stress[3]
      dict_temp["ww"] = DNS_stress[4]
      dict_temp["uv"] = DNS_stress[5]
      dict_temp["uw"] = DNS_stress[6]
      dict_temp["vw"] = DNS_stress[7]
      dict_temp["k"]  = 0.5*(dict_temp["uu"]+dict_temp["vv"]+dict_temp["ww"])


      dict_temp["eps"] = DNS_RSTE[:,7]
      dict_temp["eps"][0]=dict_temp["eps"][1]
      dict_temp["visc_diff"] =  DNS_RSTE[:,4]
   else:
      print(f'{data_set} is not a valid code word for a data set')

   # set a min on dudy
   dict_temp["dudy"]  = np.maximum(np.gradient(dict_temp["u"],dict_temp["yplus"]),4e-4)

   # Calculate ny_t and time-scale tau
   dict_temp["viscous_t"] = dict_temp["k"]**2/dict_temp["eps"] 
   # tau       = viscous_t/abs(uv_DNS)
   #DNS

   dict_temp["dudy_org"] = np.copy(dict_temp["dudy"])

   dict_temp["tau"] = dict_temp["k"]/dict_temp["eps"]

   # make dudy non-dimensional
   #dudy_DNS = dudy_DNS*tau_DNS
   #tau_DNS = np.ones(len(dudy_DNS))

   # Calculate c_0 & c_2 of the Non-linear Eddy Viscosity Model

   dict_temp["a11"] = dict_temp["uu"]/dict_temp["k"]-0.66666
   dict_temp["a22"] = dict_temp["vv"]/dict_temp["k"]-0.66666
   dict_temp["a33"] = dict_temp["ww"]/dict_temp["k"]-0.66666

   dict_temp["c_2"] = (2*dict_temp["a11"]+dict_temp["a33"])/dict_temp["tau"]**2/dict_temp["dudy"]**2
   dict_temp["c_0"] = -6*dict_temp["a33"]/dict_temp["tau"]**2/dict_temp["dudy"]**2


   




   # Crop values with yplusmin < y+ < yplusmax
   index_choose=np.nonzero((dict_temp["yplus"] > yplusmin )  & (dict_temp["yplus"] < yplusmax ))
   for key in dict_temp:
      dict_temp[key] = dict_temp[key][index_choose]
      
   c = np.array([dict_temp["c_0"],dict_temp["c_2"]])
   
   # Don't put these in dictionary, they will be in X
   dict_temp["dudy_squared"] = (dict_temp["dudy"]**2)
   #scale with k and eps 
   # dudy [1/T]
   # dudy**2 [1/T**2]
   
   # N.b. T and tau are the same thing!
   dict_temp["dudy_squared_scaled"] = dict_temp["dudy_squared"]*dict_temp["tau"]**2
   dict_temp["dudy_squared_scaled"] = dict_temp["dudy_squared_scaled"].reshape(-1,1)
   
   dict_temp["dudy_inv"] = 1/dict_temp["dudy"]/dict_temp["tau"]

   dict_temp["dudy_inv_scaled"] = dict_temp["dudy_inv"].reshape(-1,1)
   
   # use MinMax scaler
   #scaler_dudy2 = StandardScaler()
   #scaler_tau = StandardScaler()
   dict_temp["scaler_dudy2"] = MinMaxScaler()
   dict_temp["scaler_dudy"] = MinMaxScaler()
   
   X=np.zeros((len(dict_temp["dudy"]),2))
   X[:,0] = dict_temp["scaler_dudy2"].fit_transform(dudy_squared_scaled)[:,0]
   X[:,1] = dict_temp["scaler_dudy"].fit_transform(dudy_inv_scaled)[:,0]

   dict_temp["y"] = c.transpose()
   dict_temp["X"] = X
   
   dict_temp["prod"] = -dict_temp["uv"]*dict_temp["dudy"]

   return dict_temp

dict_train_full = loaddict(trainset,30,1000)
dict_val = loaddict(valset,30,1000)

#uv_DNS = dict_train["uv"]
#uu_DNS = dict_train["uu"]
#vv_DNS =  dict_train["vv"]
#ww_DNS =  dict_train["ww"]
#k_DNS =  dict_train["k"]
#eps_DNS =  dict_train["eps"]
#dudy_DNS = dict_train["dudy"]
#yplus_DNS =  dict_train["yplus"]
#y_DNS =  dict_train["y"]
#u_DNS =  dict_train["u"]

#uv_VAL = dict_val["uv"]
#uu_VAL = dict_val["uu"]
#vv_VAL = dict_val["vv"]
#ww_VAL = dict_val["ww"]
#k_VAL = dict_val["k"]
#eps_VAL = dict_val["eps"]
#dudy_VAL = dict_val["dudy"]
#yplus_VAL = dict_val["yplus"]
#y_VAL = dict_val["y"]
#u_VAL = dict_val["u"]

#-----------------Data_manipulation--------------------



# Originally this one
#index_choose=np.nonzero((yplus_DNS > 9 )  & (yplus_DNS< 2200 ))


# Calculate ny_t and time-scale tau
#viscous_t = k_DNS**2/eps_DNS 
# tau       = viscous_t/abs(uv_DNS)
#DNS

#dudy_DNS_org = np.copy(dudy_DNS)

#tau_DNS = k_DNS/eps_DNS

# make dudy non-dimensional
#dudy_DNS = dudy_DNS*tau_DNS
#tau_DNS = np.ones(len(dudy_DNS))

# Calculate c_0 & c_2 of the Non-linear Eddy Viscosity Model

#a11_DNS=uu_DNS/k_DNS-0.66666
#a22_DNS=vv_DNS/k_DNS-0.66666
#a33_DNS=ww_DNS/k_DNS-0.66666

#c_2_DNS=(2*a11_DNS+a33_DNS)/tau_DNS**2/dudy_DNS**2
#c_0_DNS=-6*a33_DNS/tau_DNS**2/dudy_DNS**2

#c = np.array([c_0_DNS,c_2_DNS])


# ------------------------------- Data manipulation for validation data ------------------------------------------
#tau_VAL = k_VAL/eps_VAL

#a11_VAL=uu_VAL/k_VAL-0.66666
#a22_VAL=vv_VAL/k_VAL-0.66666
#a33_VAL=ww_VAL/k_VAL-0.66666

#c_2_VAL=(2*a11_VAL+a33_VAL)/tau_VAL**2/dudy_VAL**2
#c_0_VAL=-6*a33_VAL/tau_VAL**2/dudy_VAL**2

#### EVERYTHING TO THIS POINT HAS JUST CROPPED INITIAL DATASET TO CERTAIN y+ VALUES AND CALCULATED c:s ####


########################## 2*a11_DNS+a33_DNS

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(2*dict_train_full["a11"]+dict_train_full["a33"],dict_train_full["yplus"], marker="o", s=10, c="red", label="Cropped initial dataset")
plt.xlabel("$2a_{11}+a_{33}$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}2a11_DNS+a33_DNS-dudy2-and-tau-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')




########################## k-bal
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
#ax1.plot(yplus_DNS_uu,prod_DNS, 'b-', label="prod")
ax1.plot(dict_train_full["prod"],dict_train_full["yplus"], 'b-', label="$-\overline{u'v'} \partial U/\partial y$ Cropped Original")
ax1.plot(dict_train_full["eps"],dict_train_full["yplus"],'r--', label="dissipation")
plt.axis([0,200,0,0.3])
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}prod-diss-DNS-dudy2-and-tau-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-ustar-and-nu-BL.png')


# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility
# random_state = randrange(100)

indices = np.arange(len(dict_train_full["X"]))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(dict_train_full["X"],dict_train_full["y"], indices,test_size=0.2,shuffle=True,random_state=42)

# create test index 
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

dict_test = {}
dict_train = {}
for key in dict_train_full:
   if key != "X" and key != "y":
      dict_test[key] = dict_train_full[key][index_test]
      dict_train[key] = dict_train_full[key][index_train]

dict_test["X"] = X_test
dict_test["y"] = y_test
dict_train["X"] = X_train
dict_train["y"] = y_train

# Set up hyperparameters

# Suggestions : 1e-1, 2e-1, 5e-1, 9e-1, 1e1 
learning_rate = 1e-1

# Suggestions : 3, 5, 30
my_batch_size = 5

# Suggestions : 3e1, 1e4, 4e4
epochs = 100

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

########################## check
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

# OBS
# Suggestions : 5, len(X_train)
Nx=len(X_train)

for k in range(0,Nx):
# train_k = next(itertools.islice(train_loader, k, None))
  train_k = train_dataset[k]
  #k_train = index_train[k]
  yplus = dict_train["yplus"][k]
# print ('k,k_train,c_0_train,yplus',k,k_train,train_k[1][0][0],yplus)
  print ('k,k_train,c_0_train,yplus',k,train_dataset[k][1][0],yplus)
  if k == 0: 
     #plt.plot(c_0_DNS[k_train],yplus, 'ro',label='target')
     plt.plot(train_dataset[k][1][0],yplus, 'b+',label='Train dataset')
  else:
     #plt.plot(c_0_DNS[k_train],yplus, 'ro')
     plt.plot(train_dataset[k][1][0],yplus, 'b+')
Mx=len(X_test)
for k in range(0,Mx):
   #k_test = index_test[k]
   yplus = dict_test["yplus"][k]
   if k == 0:
      plt.plot(test_dataset[k][1][0],yplus, 'ro',label='Test dataset')
   else:
      plt.plot(test_dataset[k][1][0],yplus, 'ro')

plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}c0-and-cNN-train-and-test-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-ustar-and-nu-BL.png')

# Everything to this point makes sense and splits the dataset between test and train data

# Let's set up a neural network:

class ThePredictionMachine(nn.Module):

    def __init__(self):
        
        super(ThePredictionMachine, self).__init__()

        self.input   = nn.Linear(2, 50)
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 2)

#        self.input   = nn.Linear(2, 50)
#        self.hidden1 = nn.Linear(50, 50)
#        self.hidden2 = nn.Linear(50, 25)
#        self.hidden3 = nn.Linear(25, 2)

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


        #self.input   = nn.Linear(2, 50)
        #self.hidden1 = nn.Linear(50, 50)
        #self.hidden2 = nn.Linear(50, 50)
        #self.hidden3 = nn.Linear(50, 50)
        #self.hidden4 = nn.Linear(50, 50)
        #self.hidden5 = nn.Linear(50, 50)
        #self.hidden6 = nn.Linear(50, 50)
        #self.hidden7 = nn.Linear(50, 25)
        #self.hidden8 = nn.Linear(25, 2)



    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(nn.functional.relu(x))
        x = self.hidden2(nn.functional.relu(x))

#        x = nn.functional.relu(self.input(x))
#        x = nn.functional.relu(self.hidden1(x))
#        x = nn.functional.relu(self.hidden2(x))
#        x = self.hidden3(x)

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

#        x = nn.functional.relu(self.input(x))
 #       x = nn.functional.relu(self.hidden1(x))
  #      x = nn.functional.relu(self.hidden2(x))
   #     x = nn.functional.relu(self.hidden3(x))
    #    x = nn.functional.relu(self.hidden4(x))
     #   x = nn.functional.relu(self.hidden5(x))
      #  x = nn.functional.relu(self.hidden6(x))
       # x = nn.functional.relu(self.hidden7(x))
        #x = self.hidden8(x)

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

#y = c.transpose()

'''
dudy_squared_VAL = (dudy_VAL**2)
# scale with k and eps 
# dudy [1/T]
# dudy**2 [1/T**2]
T_VAL = tau_VAL
dudy_squared_VAL_scaled = dudy_squared_VAL*T_VAL**2
dudy_VAL_inv = 1/dudy_VAL/T_VAL
# re-shape
dudy_squared_VAL_scaled = dudy_squared_VAL_scaled.reshape(-1,1)
dudy_VAL_inv_scaled = dudy_VAL_inv.reshape(-1,1)
# use MinMax scaler
#scaler_dudy2 = StandardScaler()
#scaler_tau = StandardScaler()
scaler_dudy2_VAL = MinMaxScaler()
scaler_dudy_VAL = MinMaxScaler()
X_VAL=np.zeros((len(dudy_VAL),2))
X_VAL[:,0] = scaler_dudy2_VAL.fit_transform(dudy_squared_VAL_scaled)[:,0]
X_VAL[:,1] = scaler_dudy_VAL.fit_transform(dudy_VAL_inv_scaled)[:,0]
'''
X_VAL_tensor = torch.tensor(dict_val["X"], dtype=torch.float32)
#preds_VAL = neural_net(X_VAL_tensor)


   



print(f"{'time ML: '}{time.time()-start_time:.2e}")

#transform from tensor to numpy


def calc_dict(dict_temp, X_tens, model,typelabel):

   preds = model(X_tens)


   c_NN = preds.detach().numpy()
 
#c_NN_old = c_NN

   dict_temp["c0_NN"] = c_NN[:,0]
   dict_temp["c2_NN"] =c_NN[:,1]
   
   dict_temp["a_11_NN"] = 1/12*dict_temp["tau"]**2*dict_temp["dudy"]**2*(dict_temp["c0"] + 6*dict_temp["c2"])
   dict_temp["uu_NN"] = (dict_temp["a_11_NN"]+0.6666)*dict_temp["k"]

   #a_{22} = \frac{1}{12} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 - 6c_2 + c_3)
   dict_temp["a_22_NN"] = 1/12*dict_temp["tau"]**2*dict_temp["dudy"]**2*(dict_temp["c0"] - 6*dict_temp["c2"])
   dict_temp["vv_NN"] = (dict_temp["a_22_NN"]+0.6666)*dict_temp["k"]

   # a_{33} = -\frac{1}{6} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 + c_3)
   dict_temp["a_33"] = -1/6*dict_temp["tau"]**2*dict_temp["dudy"]**2*dict_temp["c0"]
   dict_temp["ww_NN"] = (dict_temp["a_33"]+0.6666)*dict_temp["k"]

   # Compare NN values to original data
   dict_temp["c0_std"] = np.std(dict_temp["c0_NN"]-dict_temp["c0"])/(np.mean(dict_temp["c0"].flatten()**2))**0.5
   dict_temp["c2_std"] = np.std(dict_temp["c2_NN"]-dict_temp["c2"])/(np.mean(dict_temp["c2"].flatten()**2))**0.5

   print('\nc0_error_std',dict_temp["c0_std"])
   print('\nc2_error_std',dict_temp["c2_std"])

   np.savetxt(f'{savedir}{typelabel}error-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.txt', [test_loss,dict_temp["c0_std"],dict_temp["c2_std"]] )
   #return dict_temp


calc_dict(dict_test,X_test_tensor,neural_net,"Test_")
calc_dict(dict_val,X_VAL_tensor,neural_net,"val_")


def plot_dict(dict_temp,typelabel):

   filename = f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.pth'
   torch.save(neural_net, filename)
   dump(scaler_dudy2,f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2_scaler-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')
   dump(scaler_dudy,f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2_scaler-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')


   dudy2_max = np.max(dict_temp["dudy_squared"])
   dudy2_min = np.min(dict_temp["dudy_squared"])
   dudy_min = np.min(dict_temp["dudy"])
   dudy_max = np.max(dict_temp["dudy_DNS"])
   c0_min = np.min(dict_temp["c0"])
   c0_max = np.max(dict_temp["c0"])
   c2_min = np.min(dict_temp["c2"])
   c2_max = np.max(dict_temp["c2"])

   np.savetxt(f'{savedir}{typelabel}min-max-model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.6-scale-with-k-eps-units-BL.txt', [dudy2_min, dudy2_max, dudy_min, dudy_max, c0_min, c0_max, c2_min, c2_max] )



   ########################## c0
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   for k in range(0,len(X_test)):
   yplus = yplus_DNS[k]
   if k == 0: 
      plt.plot(dict_temp["c0"][k],dict_temp["yplus"], 'bo',label='target')
      plt.plot(dict_temp["c0_NN"][k],yplus, 'r+',label='NN')
   else:
      plt.plot(dict_temp["c0"][k],dict_temp["yplus"], 'bo')
      plt.plot(dict_temp["c0_NN"][k],dict_temp["yplus"], 'r+')
   plt.xlabel("$c_0$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}c0-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


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
   plt.savefig(f'{savedir}{typelabel}c0-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


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
   plt.savefig(f'{savedir}{typelabel}c2-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


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
   plt.savefig(f'{savedir}{typelabel}c2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')




   ########################## uu
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(uu_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
   ax1.plot(uu_DNS,yplus_DNS,'b-', label="Target")
   plt.xlabel("$\overline{u'u'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}uu-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


   ########################## vv
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(vv_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
   ax1.plot(vv_DNS,yplus_DNS,'b-', label="Target")
   plt.xlabel("$\overline{v'v'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}vv-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## ww
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(ww_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
   ax1.plot(ww_DNS,yplus_DNS,'b-', label="Target")
   plt.xlabel("$\overline{w'w'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}ww-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## time scales
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.plot(dudy_DNS_org,yplus_DNS,'r-', label=r"$dudy$")
   ax1.plot(1/dudy_DNS_org,yplus_DNS,'b-', label=r"$\left(\partial U/\partial y\right)^{-1}$")
   plt.xlabel("time scsles")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}time-scales-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## time scales
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.plot(dudy_DNS_org*dudy_DNS_org,yplus_DNS,'b-')
   plt.xlabel(r"$\left(\partial U/\partial y\right)^{-1} dudy$")
   plt.ylabel("$y^+$")
   plt.savefig(f'{savedir}{typelabel}dudy-times-dudy-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

plot_dict(dict_val,"val_")
plot_dict(dict_test,"test_")

print(f"{'total time: '}{time.time()-init_time:.2e}")

