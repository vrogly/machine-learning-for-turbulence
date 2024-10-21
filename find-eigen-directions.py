# %%
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load
import itertools
import pandas as pd

trainset = 'CF5200' #CF5200
valset = 'BL' #BL

yplusmin_train = 5
yplusmax_train = 1000

yplusmin_val = 5
yplusmax_val = 1000

# Add unique run ID here
savedir = "renders/investigate-distances/"
os.makedirs(os.path.dirname(savedir), exist_ok=True)

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

   # Calculate ny_t
   dict_temp["viscous_t"] = dict_temp["k"]**2/dict_temp["eps"] 
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
   
   return dict_temp

dict_train_full = loaddict(trainset,yplusmin_train,yplusmax_train)
dict_val = loaddict(valset,yplusmin_val,yplusmax_val)


# %%
def delta_metric(dataset1:dict,dataset2:dict,x1:str,x2:str):
   accuracy = 5 # 1 is full accuracy for finding nearest neighbor
   dataset1_comb = np.array([dataset1[x1],dataset1[x2]]).T
   dataset2_comb = np.array([dataset2[x1],dataset2[x2]]).T

   dd_arr = np.zeros_like(dataset1_comb[:,0])
   dc_arr = np.zeros_like(dataset1_comb[:,0])
   dx_arr = np.zeros_like(dataset1_comb[:,0])


   for i in range(0,len(dataset1_comb[:,0])):
      dxi = 10000
      dj = 0
      # Only look at every *accuracy* to find distance, save computational time
      # Find closest point in X2-space 
      for j in range(0,len(dataset2_comb[:,0]),accuracy):
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

   # RMS
   dd = np.sqrt(np.mean(np.square(dd_arr)))
   dc = np.sqrt(np.mean(np.square(dc_arr)))
   dx = np.sqrt(np.mean(np.square(dx_arr)))

   return dd, dc, dx


exponents = [-2,-1,1,2]
variables = ["dudy","yplus","k","eps","u"]

varcombs = []
orthogonal_combinations = list(itertools.combinations(variables, 2))

# 1 Variable
for comb in variables:
   for X1_exp in exponents:
      colname = f"{comb}^{X1_exp}_0^0"
      temp_col_test = np.array((dict_train_full[comb]**X1_exp))
      temp_col_val = np.array((dict_val[comb]**X1_exp))

      scaler_test = MinMaxScaler()
      scaler_val = MinMaxScaler()

      dict_train_full[colname] = scaler_test.fit_transform(temp_col_test.reshape(-1,1))[:,0]
      dict_val[colname] = scaler_val.fit_transform(temp_col_val.reshape(-1,1))[:,0]

      varcombs += [colname]


# 2 Variables
for comb in orthogonal_combinations:
   for X1_exp in exponents:
      for X2_exp in exponents:
         colname = f"{comb[0]}^{X1_exp}_{comb[1]}^{X2_exp}"
         temp_col_test = np.array((dict_train_full[comb[0]]**X1_exp) * (dict_train_full[comb[1]]**X2_exp))
         temp_col_val = np.array((dict_val[comb[0]]**X1_exp) * (dict_val[comb[1]]**X2_exp))

         scaler_test = MinMaxScaler()
         scaler_val = MinMaxScaler()

         dict_train_full[colname] = scaler_test.fit_transform(temp_col_test.reshape(-1,1))[:,0]
         dict_val[colname] = scaler_val.fit_transform(temp_col_val.reshape(-1,1))[:,0]

         varcombs += [colname]

print(varcombs)
test_combinations = list(itertools.combinations(varcombs, 2))

df = pd.DataFrame()

progress_index = 0
for comb in test_combinations:
   if progress_index % 100 == 0:
      print(f"At {progress_index} of {len(test_combinations)}, current {comb}")
   progress_index += 1

   dd, dc, dx = delta_metric(dict_train_full,dict_val,comb[0],comb[1])
   exp1_1 = (comb[0].split("_"))[0].split("^")[1]
   X_1_1 = (comb[0].split("_"))[0].split("^")[0]
   exp1_2 = (comb[0].split("_"))[1].split("^")[1]
   X_1_2 = (comb[0].split("_"))[1].split("^")[0]

   exp2_1 = (comb[1].split("_"))[0].split("^")[1]
   X_2_1 = (comb[1].split("_"))[0].split("^")[0]
   exp2_2 = (comb[1].split("_"))[1].split("^")[1]
   X_2_2 = (comb[1].split("_"))[1].split("^")[0]

   temp ={'Name': [comb], 'X_1_1': [X_1_1], 'exponent_1_1': [exp1_1], 'X_1_2': [X_1_2], 'exponent_1_2': [exp1_2], \
          'X_2_1': [X_2_1], 'exponent_2_1': [exp2_1], 'X_2_2': [X_2_2], 'exponent_2_2': [exp2_2], \
            'dx' : [dx], 'dc' : [dc], 'dd' : [dd]}
   df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)

df.to_csv(f'{savedir}rankings.csv', sep='\t')
print(df)

# %%

# Compare dudy2k2eps-2, dudy-1k1eps-1 manually

colname_default_A = "dudy^2_k2^_eps^-2"
temp_col_test_default_A = np.array((dict_train_full["dudy"]**2) * (dict_train_full["k"]**2) * (dict_train_full["eps"]**(-2)))
temp_col_val_default_A = np.array((dict_val["dudy"]**2) * (dict_val["k"]**2) * (dict_val["eps"]**(-2)))

scaler_test_default_A = MinMaxScaler()
scaler_val_default_A = MinMaxScaler()

dict_train_full[colname_default_A] = scaler_test_default_A.fit_transform(temp_col_test_default_A.reshape(-1,1))[:,0]
dict_val[colname_default_A] = scaler_val_default_A.fit_transform(temp_col_val_default_A.reshape(-1,1))[:,0]

###

colname_default_B = "dudy^-1_k^1_eps-1"
temp_col_test_default_B = np.array((dict_train_full["dudy"]**(-1)) * (dict_train_full["k"]**1) * (dict_train_full["eps"]**(-1)))
temp_col_val_default_B = np.array((dict_val["dudy"]**(-1)) * (dict_val["k"]**1) * (dict_val["eps"]**(-1)))

scaler_test_default_B = MinMaxScaler()
scaler_val_default_B = MinMaxScaler()

dict_train_full[colname_default_B] = scaler_test_default_B.fit_transform(temp_col_test_default_B.reshape(-1,1))[:,0]
dict_val[colname_default_B] = scaler_val_default_B.fit_transform(temp_col_val_default_B.reshape(-1,1))[:,0]

print(delta_metric(dict_train_full,dict_val,colname_default_A,colname_default_B))