import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tb = pd.read_table('..//results//mini_MSR_B.table', header=7)

#%%
tb_arr = tb.to_numpy()
tb_arr = tb_arr[::4]
ax = plt.figure().add_subplot(projection='3d')

for i in range(len(tb_arr)):
    lst_of_val = tb_arr[i].item().split()
    x = float(lst_of_val[0])
    y = float(lst_of_val[1])
    z = float(lst_of_val[2])
    Bx = float(lst_of_val[3])
    By = float(lst_of_val[4])
    Bz = float(lst_of_val[5])
    
    norm = np.sqrt(Bx**2+By**2+Bz**2)
    
    ax.quiver(x, y, z, Bx, By, Bz, length=norm*3000, normalize=True, cmap=plt.cm.jet)

plt.show()

