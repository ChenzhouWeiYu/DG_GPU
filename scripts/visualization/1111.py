import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt


data = np.loadtxt('./Sedov/Order_1/rho_T_%d_N_20.txt'%(1))
x,y,z,rh,uh,vh,wh,eh = data.T

plt.figure(figsize=(7,5),dpi=400)
# plt.tripcolor(x,y,uh,cmap='jet',vmin=1-1e-5,vmax=1+1e-5)
plt.tripcolor(x,y,(rh),cmap='jet',)
plt.colorbar()
# plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
plt.xticks(np.linspace(0,1,5),minor=False)
plt.xticks(np.linspace(0,1,21),minor=True)
plt.yticks(np.linspace(0,1,5),minor=False)
plt.yticks(np.linspace(0,1,21),minor=True)
plt.grid(which='both')
plt.xlabel('x')
plt.ylabel('y')
plt.show()