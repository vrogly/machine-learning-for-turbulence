from NN_train import *

# Suggestions : 1e-1, 2e-1, 5e-1, 9e-1, 1e1 
learning_rate = 1e-1

# Suggestions : 3, 5, 30
my_batch_size = 5

# Suggestions : 3e1, 1e4, 4e4
epochs = 5000

run1 = {}
run1["trainset"] = 'BL'
run1["valset"] = 'CF5200'
run1["yplusmin_train"] = 5
run1["yplusmax_train"] = 1000
run1["yplusmin_val"] = 5
run1["yplusmax_val"] = 1000
run1["X"] = 'dudy'
run1["concatenate"] = False
run1["search"] = False
run1["dir"] = "renders/xdudy_BL-CF/"

run2 = {}
run2["trainset"] = 'CF5200'
run2["valset"] = 'BL'
run2["yplusmin_train"] = 5
run2["yplusmax_train"] = 1000
run2["yplusmin_val"] = 5
run2["yplusmax_val"] = 1000
run2["X"] = 'dudy'
run2["concatenate"] = False
run2["search"] = False
run2["dir"] = "renders/xdudy_CF-BL/"

run3 = {}
run3["trainset"] = 'CF5200'
run3["valset"] = 'BL'
run3["yplusmin_train"] = 5
run3["yplusmax_train"] = 1000
run3["yplusmin_val"] = 5
run3["yplusmax_val"] = 1000
run3["X"] = 'yplusk'
run3["concatenate"] = False
run3["search"] = False
run3["dir"] = "renders/xyplusk_CF-BL/"

run4 = {}
run4["trainset"] = 'BL'
run4["valset"] = 'CF5200'
run4["yplusmin_train"] = 5
run4["yplusmax_train"] = 1000
run4["yplusmin_val"] = 5
run4["yplusmax_val"] = 1000
run4["X"] = 'yplusk'
run4["concatenate"] = False
run4["search"] = False
run4["dir"] = "renders/xyplusk_BL-CF/"

run5 = {}
run5["trainset"] = 'BL'
run5["valset"] = 'CF5200'
run5["yplusmin_train"] = 5
run5["yplusmax_train"] = 1000
run5["yplusmin_val"] = 5
run5["yplusmax_val"] = 1000
run5["X"] = 'yplusk'
run5["concatenate"] = True
run5["search"] = False
run5["dir"] = "renders/xyplusk_Both/"

run6 = {}
run6["trainset"] = 'BL'
run6["valset"] = 'CF5200'
run6["yplusmin_train"] = 5
run6["yplusmax_train"] = 1000
run6["yplusmin_val"] = 5
run6["yplusmax_val"] = 1000
run6["X"] = 'dudy'
run6["concatenate"] = True
run6["search"] = False
run6["dir"] = "renders/xdudy_Both/"

# Viktor's ideas
run7 = {}
run7["trainset"] = 'BL'
run7["valset"] = 'CF5200'
run7["yplusmin_train"] = 5
run7["yplusmax_train"] = 1000
run7["yplusmin_val"] = 5
run7["yplusmax_val"] = 1000
run7["X"] = 'dudy-2yplus-2yplus2k-2'
run7["concatenate"] = True
run7["search"] = False
run7["dir"] = "renders/bra_enligt_viktor_Both/"

run8 = {}
run8["trainset"] = 'BL'
run8["valset"] = 'CF5200'
run8["yplusmin_train"] = 5
run8["yplusmax_train"] = 1000
run8["yplusmin_val"] = 5
run8["yplusmax_val"] = 1000
run8["X"] = 'yplus2eps2k1eps-2'
run8["concatenate"] = True
run8["search"] = False
run8["dir"] = "renders/bästa_utan_dudy_enligt_viktor_Both/"

run9 = {}
run9["trainset"] = 'BL'
run9["valset"] = 'CF5200'
run9["yplusmin_train"] = 5
run9["yplusmax_train"] = 1000
run9["yplusmin_val"] = 5
run9["yplusmax_val"] = 1000
run9["X"] = 'dudy-2yplus-2yplus2k-2'
run9["concatenate"] = False
run9["search"] = False
run9["dir"] = "renders/bra_enligt_viktor_BL-CF/"

run10 = {}
run10["trainset"] = 'CF5200'
run10["valset"] = 'BL'
run10["yplusmin_train"] = 5
run10["yplusmax_train"] = 1000
run10["yplusmin_val"] = 5
run10["yplusmax_val"] = 1000
run10["X"] = 'dudy-2yplus-2yplus2k-2'
run10["concatenate"] = False
run10["search"] = False
run10["dir"] = "renders/bra_enligt_viktor_CF-BL/"

run13 = {}
run13["trainset"] = 'CF5200'
run13["valset"] = 'BL'
run13["yplusmin_train"] = 5
run13["yplusmax_train"] = 1000
run13["yplusmin_val"] = 5
run13["yplusmax_val"] = 1000
run13["X"] = 'yplusk2'
run13["concatenate"] = False
run13["search"] = False
run13["dir"] = "renders/xyplusk2_CF-BL/"

run14 = {}
run14["trainset"] = 'BL'
run14["valset"] = 'CF5200'
run14["yplusmin_train"] = 5
run14["yplusmax_train"] = 1000
run14["yplusmin_val"] = 5
run14["yplusmax_val"] = 1000
run14["X"] = 'yplusk2'
run14["concatenate"] = False
run14["search"] = False
run14["dir"] = "renders/xyplusk2_BL-CF/"

run15 = {}
run15["trainset"] = 'BL'
run15["valset"] = 'CF5200'
run15["yplusmin_train"] = 5
run15["yplusmax_train"] = 1000
run15["yplusmin_val"] = 5
run15["yplusmax_val"] = 1000
run15["X"] = 'yplusk2'
run15["concatenate"] = True
run15["search"] = False
run15["dir"] = "renders/xyplusk2_Both/"

#runs = [run1,run2,run3,run4,run5,run6]
#runs = [run7,run8]
#runs = [run9,run10]
runs = [run13,run14,run15]

for run in runs:
    main(learning_rate,my_batch_size,epochs,run["trainset"],run["valset"],run["yplusmin_train"],
         run["yplusmax_train"],run["yplusmin_val"],run["yplusmax_val"],run["dir"],run["X"],run["concatenate"],run["search"])



