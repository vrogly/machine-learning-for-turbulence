import numpy as np
import torch 
import sys 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.discriminant_analysis import StandardScaler
from random import randrange
from joblib import dump, load
from matplotlib import ticker


plt.rcParams.update({'font.size': 12})
plt.interactive(True)
plt.close('all')


# load DNS data

#vel_DNS=np.genfromtxt("/chalmers/users/lada/kth-data-bound-layer/vel_11000_DNS.dat", dtype=None,comments="%")
vel_DNS=np.genfromtxt("vel_11000_DNS.dat", comments="%")

# % Wall-normal profiles:
# y/\delta_{99}       y+          U+          urms+       vrms+       wrms+       uv+         prms+       pu+         pv+         S(u)        F(u)        dU+/dy+     V+



y_DNS=vel_DNS[:,0]
yplus_DNS=vel_DNS[:,1]
u_DNS=vel_DNS[:,2]
uu_DNS=vel_DNS[:,3]**2
vv_DNS=vel_DNS[:,4]**2
ww_DNS=vel_DNS[:,5]**2
uv_DNS=vel_DNS[:,6]



# %Wall-normal profiles:
#y/\delta_{99}       y+          conv+          prod+          diss+          t-diff+        velp+          vis-diff+      residual+



DNS_RSTE = np.genfromtxt("bud_11000.prof",comments="%")


eps_DNS = -DNS_RSTE[:,4]

