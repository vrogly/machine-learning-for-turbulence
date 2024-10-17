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

#plt.rcParams.update({'font.size': 22})
#plt.interactive(True)'



   # y/d99           y+              Produc.         Advect.         Tur. flux       Pres. flux      Dissip
   #DNS_RSTE = np.genfromtxt("/chalmers/users/lada/DNS-boundary-layers-jimenez/balances_6500_Re_theta.6500.bal.uu.txt",comments="%")
   #
   #prod_DNS = -DNS_RSTE[:,2]*3/2 #multiply by 3/2 to get P^k from P_11
   #eps_DNS = -DNS_RSTE[:,6]*3/2  #multiply by 3/2 to get eps from eps_11
   #yplus_DNS_uu = DNS_RSTE[:,1]
      #yplus_DNS_uu = yplus_DNS


   # make dudy non-dimensional
   #dudy_DNS = dudy_DNS*tau_DNS
   #tau_DNS = np.ones(len(dudy_DNS))

#scaler_dudy2 = StandardScaler()
   #scaler_tau = StandardScaler()
   
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

'''
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
'''

# OBS
# Suggestions : 5, len(X_train)
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

#ax1.plot(yplus_DNS_uu,prod_DNS, 'b-', label="prod")
#ax1.plot(dict_train_full["prod"],dict_train_full["yplus"], 'b-', label="$-\overline{u'v'} \partial U/\partial y$ Cropped Original")

#ax1.plot(yplus_DNS_uu,prod_DNS, 'b-', label="prod")
#ax1.plot(dict_train_full["prod"],dict_train_full["yplus"], 'b-', label="$-\overline{u'v'} \partial U/\partial y$ Cropped Original")

#ax1.plot(yplus_DNS_uu,prod_DNS, 'b-', label="prod")
#ax1.plot(dict_train_full["prod"],dict_train_full["yplus"], 'b-', label="$-\overline{u'v'} \partial U/\partial y$ Cropped Original")