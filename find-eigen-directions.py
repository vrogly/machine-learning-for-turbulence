# %%
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
import itertools

plt.rcParams.update({"font.size": 12})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["figure.figsize"] = (3.2, 2.4)
plt.rcParams["savefig.bbox"] = "tight"
plt.interactive(False)

plt.close('all')

# Set up hyperparameters

# Suggestions : 1e-1, 2e-1, 5e-1, 9e-1, 1e1 
learning_rate = 1e-1

# Suggestions : 3, 5, 30
my_batch_size = 5

# Suggestions : 3e1, 1e4, 4e4
epochs = 5000


trainset = 'CF5200' #CF5200
valset = 'BL' #BL

yplusmin_train = 5
yplusmax_train = 1000

yplusmin_val = 5
yplusmax_val = 1000

# Add unique run ID here
savedir = "renders/xyplusk-CF-BL/"
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
   dict_temp["dudy"]  = np.maximum(np.gradient(dict_temp["u"],dict_temp["yplus"]),0)

   # Calculate ny_t and time-scale tau
   dict_temp["viscous_t"] = dict_temp["k"]**2/dict_temp["eps"] 
   # tau       = viscous_t/abs(uv_DNS)
   #DNS

   dict_temp["dudy_org"] = np.copy(dict_temp["dudy"])

   dict_temp["tau"] = dict_temp["k"]/dict_temp["eps"]
   # Calculate c_0 & c_2 of the Non-linear Eddy Viscosity Model

   dict_temp["a11"] = dict_temp["uu"]/dict_temp["k"]-0.66666
   dict_temp["a22"] = dict_temp["vv"]/dict_temp["k"]-0.66666
   dict_temp["a33"] = dict_temp["ww"]/dict_temp["k"]-0.66666

   dict_temp["c2"] = (2*dict_temp["a11"]+dict_temp["a33"])/dict_temp["tau"]**2/dict_temp["dudy"]**2
   dict_temp["c0"] = -6*dict_temp["a33"]/dict_temp["tau"]**2/dict_temp["dudy"]**2



   # Crop values with yplusmin < y+ < yplusmax
   index_choose=np.nonzero((dict_temp["yplus"] > yplusmin )  & (dict_temp["yplus"] < yplusmax ))
   for key in dict_temp:
      dict_temp[key] = dict_temp[key][index_choose]
      
   c = np.array([dict_temp["c0"],dict_temp["c2"]])
   
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

   dict_temp["L"] = dict_temp["k"]**(3/2)/dict_temp["eps"]
   dict_temp["quotient"] = dict_temp["viscous_t"]/(dict_temp["L"]**2*dict_temp["tau"])
   
   # use MinMax scaler
   dict_temp["scaler_dudy2"] = MinMaxScaler()
   dict_temp["scaler_dudy"] = MinMaxScaler()
   dict_temp["scaler_yplus"] = MinMaxScaler()
   dict_temp["scaler_tau"] = MinMaxScaler()
   dict_temp["scaler_L"] = MinMaxScaler()
   dict_temp["scaler_quotient"] = MinMaxScaler()
   dict_temp["scaler_k"] = MinMaxScaler()
   dict_temp["scaler_dudy_pure"] = MinMaxScaler()
   dict_temp["scaler_u"] = MinMaxScaler()

   
   X=np.zeros((len(dict_temp["dudy"]),2))
   X[:,0] = dict_temp["scaler_dudy2"].fit_transform(dict_temp["dudy_squared_scaled"])[:,0]
   X[:,1] = dict_temp["scaler_dudy"].fit_transform(dict_temp["dudy_inv_scaled"])[:,0]


   dict_temp["minmax_dudy_squared_scaled"] = X[:,0]
   dict_temp["minmax_dudy_inv_scaled"] = X[:,1]
   dict_temp["minmax_yplus"] = dict_temp["scaler_yplus"].fit_transform(dict_temp["yplus"].reshape(-1,1))[:,0]
   dict_temp["minmax_tau"] = dict_temp["scaler_tau"].fit_transform(dict_temp["tau"].reshape(-1,1))[:,0]
   dict_temp["minmax_L"] = dict_temp["scaler_L"].fit_transform(dict_temp["L"].reshape(-1,1))[:,0]
   dict_temp["minmax_quotient"] = dict_temp["scaler_quotient"].fit_transform(dict_temp["quotient"].reshape(-1,1))[:,0]
   dict_temp["minmax_k"] = dict_temp["scaler_k"].fit_transform(dict_temp["k"].reshape(-1,1))[:,0]
   dict_temp["minmax_dudy"] = dict_temp["scaler_dudy_pure"].fit_transform(dict_temp["dudy"].reshape(-1,1))[:,0]
   dict_temp["minmax_u"] = dict_temp["scaler_u"].fit_transform(dict_temp["u"].reshape(-1,1))[:,0]

   X2=np.zeros((len(dict_temp["dudy"]),2))
   X2[:,0] = dict_temp["minmax_k"]
   X2[:,1] = dict_temp["minmax_yplus"]


   dict_temp["y"] = c.transpose()
   dict_temp["X"] = X2
   
   dict_temp["prod"] = -dict_temp["uv"]*dict_temp["dudy"]

   return dict_temp

dict_train_full = loaddict(trainset,yplusmin_train,yplusmax_train)
dict_val = loaddict(valset,yplusmin_val,yplusmax_val)

def delta_metric(dataset1:dict,dataset2:dict,x1:str,x2:str):
   #scaler_long_1
   #scaler_long_2
   #scaler_short_1
   #scaler_short_2

   dataset1_comb = np.array([dataset1[x1],dataset1[x2]]).T
   dataset2_comb = np.array([dataset2[x1],dataset2[x2]]).T

   metric = np.zeros_like(dataset1_comb[:,0])

   for i in range(len(dataset1_comb[:,0])):
      dxi = 100
      dj = 0
      for j in range(len(dataset2_comb[:,0])):
         test = np.linalg.norm(dataset1_comb[i,:]-dataset2_comb[j,:])
         if test < dxi:
            dxi = test
            dj = j
      c_point_short = np.array([dataset1["c0"][i],dataset1["c2"][i]])
      c_point_long = np.array([dataset2["c0"][dj],dataset2["c2"][dj]])

      dci = np.linalg.norm(c_point_long-c_point_short)

      metric[i] = (dci*(dxi**2))
   return metric

def find_best_x(dataset1,dataset2,variables):
   i = 0
   Typelist = []
   Goodness_list = np.zeros([len(variables)**2,1])
   for variable1 in variables:
      for variable2 in variables:
         # TODO : Change to max?
         Goodness_list[i] = (np.linalg.norm(delta_metric(dataset1,dataset2,variable1,variable2))+\
                             np.linalg.norm(delta_metric(dataset2,dataset1,variable1,variable2)))/2     
         Typelist.append(variable1+variable2)
         i += 1
   Best = np.argmax(Goodness_list)
   return Typelist[Best],Typelist,Goodness_list

variables = ["minmax_dudy_squared_scaled","minmax_dudy_inv_scaled","minmax_yplus","minmax_tau","minmax_L","minmax_quotient","minmax_k","minmax_dudy","minmax_u"]

#for variable in variables:
   #print(len(dict_train_full[variable]))
#print(len(dict_train_full["minmax_yplus"]),len(dict_val["minmax_yplus"]))
Best, Types, Goodness = find_best_x(dict_train_full,dict_val,variables)
#print(Goodness)
print(f'Best combination: {Best}') 
print(f'Worst combination: {Types[np.argmin(Goodness)]}')

plt.figure()
plt.plot(dict_train_full["minmax_tau"],dict_train_full["minmax_tau"],label = 'Training data')
plt.plot(dict_val["minmax_tau"],dict_val["minmax_tau"],label = 'Validation data')
plt.legend()
plt.savefig(f'{savedir}worst_x.png')

plt.figure()
plt.plot(dict_train_full["minmax_yplus"],dict_train_full["minmax_k"],label = 'Training data')
plt.plot(dict_val["minmax_yplus"],dict_val["minmax_k"],label = 'Validation data')
plt.legend()
plt.savefig(f'{savedir}best_x.png')






'''
plt.figure()
plt.plot(dict_train_full["c0"],dict_train_full["yplus"],label = "Training y")
plt.plot(dict_val["c0"],dict_val["yplus"],label = "Validation y")
plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend()
plt.savefig(f'{savedir}c0-comparison.png')

plt.figure()
plt.plot(dict_train_full["c2"],dict_train_full["yplus"],label = "Training y")
plt.plot(dict_val["c2"],dict_val["yplus"],label = "Validation y")
plt.xlabel("$c_2$")
plt.ylabel("$y^+$")
plt.legend()
plt.savefig(f'{savedir}c2-comparison.png')
exit()'''



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
ax1.plot(dict_train_full["prod"],dict_train_full["yplus"], 'b-', label="$-\overline{u'v'} \partial U/\partial y$ Cropped Original")
ax1.plot(dict_train_full["eps"],dict_train_full["yplus"],'r--', label="dissipation")
#plt.axis([0,200,0,0.3])
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

dict_test = {}
dict_train = {}
for key in dict_train_full:
   if key not in [ "X", "y", "scaler_dudy", "scaler_dudy2","scaler_yplus","scaler_tau","scaler_L","scaler_quotient","scaler_k","scaler_dudy_pure", "scaler_u"]:
      dict_test[key] = dict_train_full[key][index_test]
      dict_train[key] = dict_train_full[key][index_train]

dict_test["X"] = X_test
dict_test["y"] = y_test
dict_train["X"] = X_train
dict_train["y"] = y_train

dict_test["scaler_dudy"] = dict_train_full["scaler_dudy"]
dict_test["scaler_dudy2"] = dict_train_full["scaler_dudy2"]
dict_test["scaler_yplus"] = dict_train_full["scaler_yplus"]
dict_test["scaler_tau"] = dict_train_full["scaler_tau"]
dict_test["scaler_L"] = dict_train_full["scaler_L"]
dict_test["scaler_quotient"] = dict_train_full["scaler_quotient"]
dict_test["scaler_k"] = dict_train_full["scaler_k"]
dict_test["scaler_dudu_pure"] = dict_train_full["scaler_dudy_pure"]
dict_test["scaler_u"] = dict_train_full["scaler_u"]

dict_train["scaler_dudy"] = dict_train_full["scaler_dudy"]
dict_train["scaler_dudy2"] = dict_train_full["scaler_dudy2"]
dict_train["scaler_yplus"] = dict_train_full["scaler_yplus"]
dict_train["scaler_tau"] = dict_train_full["scaler_tau"]
dict_train["scaler_L"] = dict_train_full["scaler_L"]
dict_train["scaler_quotient"] = dict_train_full["scaler_quotient"]
dict_train["scaler_k"] = dict_train_full["scaler_k"]
dict_train["scaler_dudu_pure"] = dict_train_full["scaler_dudy_pure"]
dict_train["scaler_u"] = dict_train_full["scaler_u"]

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

########################## check


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

Nx=len(X_train)

for k in range(0,Nx):
  train_k = train_dataset[k]
  yplus = dict_train["yplus"][k]
  print ('k,k_train,c_0_train,yplus',k,train_dataset[k][1][0],yplus)
  if k == 0: 
     plt.plot(train_dataset[k][1][0],yplus, 'b+',label='Train dataset')
  else:
     plt.plot(train_dataset[k][1][0],yplus, 'b+')
Mx=len(X_test)
for k in range(0,Mx):
   yplus = dict_test["yplus"][k]
   if k == 0:
      plt.plot(test_dataset[k][1][0],yplus, 'ro',label='Test dataset')
   else:
      plt.plot(test_dataset[k][1][0],yplus, 'ro')

plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}c0-and-cNN-train-and-test-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-ustar-and-nu-BL.png')

# %%
############################################## START VSU CODE ##################################
def delta_metric_vsu(dataset1:dict,dataset2:dict,x1:str,x2:str):

   dataset1_comb = np.array([dataset1[x1],dataset1[x2]]).T
   dataset2_comb = np.array([dataset2[x1],dataset2[x2]]).T

   #print(len(dataset1_comb[:,0]),dataset1["c0"].shape)

   dd_arr = np.zeros_like(dataset1_comb[:,0])
   dc_arr = np.zeros_like(dataset1_comb[:,0])
   dx_arr = np.zeros_like(dataset1_comb[:,0])

   for i in range(len(dataset1_comb[:,0])):
      dxi = 10000
      dj = 0
      # Only look at every fifth to find distance, save computational time
      # Find closest point in X2-space 
      for j in range(len(dataset2_comb[:,0])):
         test = np.linalg.norm(dataset1_comb[i,:]-dataset2_comb[j,:])
         if test < dxi:
            dxi = test
            dj = j
      c_point_short = np.array([dataset1["c0"][i],dataset1["c2"][i]])
      c_point_long = np.array([dataset2["c0"][dj],dataset2["c2"][dj]])

      dci = np.linalg.norm(c_point_long-c_point_short)

      dd_arr[i] = (dci*(dxi**2))
      dc_arr[i] = dci
      dx_arr[i] = dxi

   #print(dc_arr[:5])

   # RMS
   dd = np.sqrt(np.mean(np.square(dd_arr)))
   dc = np.sqrt(np.mean(np.square(dc_arr)))
   dx = np.sqrt(np.mean(np.square(dx_arr)))

   return dd, dc, dx


exponents = [-2,-1,1,2]
variables = ["dudy","yplus","k","eps","u"]

varcombs = []

# Format : X1 exponent, X2 exponent ..., dX, dc, dd
space_dict = {

}


orthogonal_combinations = list(itertools.combinations(variables, 2))

for comb in orthogonal_combinations:
   for X1_exp in exponents:
      for X2_exp in exponents:
         colname = f"{comb[0]}{X1_exp}{comb[1]}{X2_exp}"
         temp_col_test = np.array((dict_train_full[comb[0]]**X1_exp) * (dict_train_full[comb[1]]**X2_exp))
         temp_col_val = np.array((dict_val[comb[0]]**X1_exp) * (dict_val[comb[1]]**X2_exp))

         scaler_test = MinMaxScaler()
         scaler_val = MinMaxScaler()

         dict_train_full[colname] = scaler_test.fit_transform(temp_col_test.reshape(-1,1))[:,0]
         dict_val[colname] = scaler_val.fit_transform(temp_col_val.reshape(-1,1))[:,0]

         varcombs += [colname]

         #print(temp_col_test[50],temp_col_val[50],dict_train_full[colname][50],dict_val[colname][50])


test_combinations = list(itertools.combinations(varcombs, 2))

print(varcombs)

for comb in test_combinations[:1000]:
   print(comb,delta_metric_vsu(dict_train_full,dict_val,comb[0],comb[1]))
















############################################## END VSU CODE ##################################
# %%
1/0
















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
   
   dict_temp["a_11_NN"] = 1/12*dict_temp["tau"]**2*dict_temp["dudy"]**2*(dict_temp["c0_NN"] + 6*dict_temp["c2_NN"])
   dict_temp["uu_NN"] = (dict_temp["a_11_NN"]+0.6666)*dict_temp["k"]

   #a_{22} = \frac{1}{12} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 - 6c_2 + c_3)
   dict_temp["a_22_NN"] = 1/12*dict_temp["tau"]**2*dict_temp["dudy"]**2*(dict_temp["c0_NN"] - 6*dict_temp["c2_NN"])
   dict_temp["vv_NN"] = (dict_temp["a_22_NN"]+0.6666)*dict_temp["k"]

   # a_{33} = -\frac{1}{6} \tau^2 \left(\frac{\D \Vb_1}{\dx_2}\right)^2(c_1 + c_3)
   dict_temp["a_33_NN"] = -1/6*dict_temp["tau"]**2*dict_temp["dudy"]**2*dict_temp["c0_NN"]
   dict_temp["ww_NN"] = (dict_temp["a_33_NN"]+0.6666)*dict_temp["k"]

   # Compare NN values to original data
   dict_temp["c0_std"] = np.std(dict_temp["c0_NN"]-dict_temp["c0"])/(np.mean(dict_temp["c0"].flatten()**2))**0.5
   dict_temp["c2_std"] = np.std(dict_temp["c2_NN"]-dict_temp["c2"])/(np.mean(dict_temp["c2"].flatten()**2))**0.5

   print('\nc0_error_std',dict_temp["c0_std"])
   print('\nc2_error_std',dict_temp["c2_std"])

   np.savetxt(f'{savedir}{typelabel}error-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.txt', [test_loss,dict_temp["c0_std"],dict_temp["c2_std"]] )
   #return dict_temp

# %%


calc_dict(dict_test,X_test_tensor,neural_net,"Test_")
calc_dict(dict_val,X_VAL_tensor,neural_net,"val_")

########################## du/dy vs k / epsilon
'''
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.plot(dict_val["k"]/dict_val["eps"],dict_val["dudy"],'b', label="Validation")
ax1.plot(dict_train_full["k"]/dict_train_full["eps"],dict_train_full["dudy"],'r--', label="Train")
plt.ylabel("$\partial_y u$")
plt.xlabel("$k/\epsilon$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}kepsdudy.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.plot(dict_val["k"],dict_val["dudy_squared_scaled"],'b', label="Validation")
ax1.plot(dict_train_full["k"],dict_train_full["dudy_squared_scaled"],'r--', label="Train")
plt.ylabel("$(\partial_y u \\tau)^2$")
plt.xlabel("$k$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}kdudy.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.plot(dict_val["yplus"],dict_val["dudy_squared_scaled"],'b', label="Validation")
ax1.plot(dict_train_full["yplus"],dict_train_full["dudy_squared_scaled"],'r--', label="Train")
plt.ylabel("$(\partial_y u \\tau)^2$")
plt.xlabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(f'{savedir}ydudy.png')
'''



def plot_dict(dict_temp,X_tens,typelabel):

   filename = f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.pth'
   torch.save(neural_net, filename)
   dump(dict_temp["scaler_dudy2"],f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2_scaler-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')
   dump(dict_temp["scaler_dudy"],f'{savedir}{typelabel}model-channel-DNS-dudy-and-dudy2_scaler-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.bin')


   dudy2_max = np.max(dict_temp["dudy_squared"])
   dudy2_min = np.min(dict_temp["dudy_squared"])
   dudy_min = np.min(dict_temp["dudy"])
   dudy_max = np.max(dict_temp["dudy"])
   c0_min = np.min(dict_temp["c0"])
   c0_max = np.max(dict_temp["c0"])
   c2_min = np.min(dict_temp["c2"])
   c2_max = np.max(dict_temp["c2"])

   np.savetxt(f'{savedir}{typelabel}min-max-model-channel-DNS-dudy-and-dudy2-2-hidden-9-yplus-2200-dudy-min-eq.6-scale-with-k-eps-units-BL.txt', [dudy2_min, dudy2_max, dudy_min, dudy_max, c0_min, c0_max, c2_min, c2_max] )



   ########################## c0
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   plt.scatter(dict_temp["c0"],dict_temp["yplus"], c ='b', marker ='o',label='target')
   plt.scatter(dict_temp["c0_NN"],dict_temp["yplus"], c='r', marker='+', label='NN')
   plt.xlabel("$c_0$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}c0-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units.png')

   ########################## c0 v dudy**2
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)


   #dudy2_inverted=dict_temp["scaler_dudy2"].inverse_transform(X_tens)
   plt.scatter(dict_temp["c0"],dict_temp["dudy"]**2, c ='b', marker ='o',label='target')
   plt.scatter(dict_temp["c0_NN"],dict_temp["dudy"]**2,  c='r', marker='+',label='NN')   
   plt.xlabel("$c_0$")
   plt.ylabel(r"$\left(\partial U/\partial y\right)^2$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}c0-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units.png')

   ########################## c2 v dudy**2
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)

   plt.scatter(dict_temp["c2"],dict_temp["dudy"]**2,c ='b', marker ='o',label='target')
   plt.scatter(dict_temp["c2_NN"],dict_temp["dudy"]**2,  c='r', marker='+',label='NN')
   plt.xlabel("$c_2$")
   plt.ylabel(r"$\left(\partial U/\partial y\right)^2$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}c2-dudu2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## c2
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)

   plt.scatter(dict_temp["c2"],dict_temp["yplus"], c ='b', marker ='o',label='target')
   plt.scatter(dict_temp["c2_NN"],dict_temp["yplus"],  c='r', marker='+',label='NN')

   # ax4.axis([-2000, 0, 0,5000])
   # ax5.axis([-2000, 0, 0,5000])
   plt.xlabel("$c_2$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}c2-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')




   ########################## uu
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(dict_temp["uu"],dict_temp["yplus"],c='b', marker = 'o', label="Target")
   ax1.scatter(dict_temp["uu_NN"],dict_temp["yplus"], marker="+", c="red", label="NN")
   plt.xlabel("$\overline{u'u'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}uu-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')


   ########################## vv
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(dict_temp["vv"],dict_temp["yplus"],c='b', marker='o', label="Target")
   ax1.scatter(dict_temp["vv_NN"],dict_temp["yplus"], marker="+", c="red", label="NN")
   plt.xlabel("$\overline{v'v'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}vv-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## ww
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(dict_temp["ww"],dict_temp["yplus"], marker = 'o', c='b', label="Target")
   ax1.scatter(dict_temp["ww_NN"],dict_temp["yplus"], marker="+", c="red", label="Neural Network")
   plt.xlabel("$\overline{w'w'}^+$")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}ww-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## time scales
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(dict_temp["dudy_org"],dict_temp["yplus"],c='r', label=r"$dudy$")
   ax1.scatter(1/dict_temp["dudy_org"],dict_temp["yplus"],c='b', label=r"$\left(\partial U/\partial y\right)^{-1}$")
   plt.xlabel("time scsles")
   plt.ylabel("$y^+$")
   plt.legend(loc="best",fontsize=12)
   plt.savefig(f'{savedir}{typelabel}time-scales-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

   ########################## time scales
   fig1,ax1 = plt.subplots()
   plt.subplots_adjust(left=0.20,bottom=0.20)
   ax1.scatter(dict_temp["dudy_org"]*dict_temp["dudy_org"],dict_temp["yplus"],c='b')
   plt.xlabel(r"$\left(\partial U/\partial y\right)^{-1} dudy$")
   plt.ylabel("$y^+$")
   plt.savefig(f'{savedir}{typelabel}dudy-times-dudy-dudy-and-dudy-squared-dudy2-and-dudy-2-hidden-9-yplus-2200-dudy-min-eq.4e-4-scale-with-k-eps-units-BL.png')

plot_dict(dict_val,X_VAL_tensor,"val_")
plot_dict(dict_test,X_test_tensor,"test_")

print(f"{'total time: '}{time.time()-init_time:.2e}")