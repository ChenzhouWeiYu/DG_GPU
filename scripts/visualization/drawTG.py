import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt


data = np.loadtxt('./Sedov/Order_1/rho_T_%d_N_40.txt'%(2))
if (data.shape[1]==3+5*2):
    x,y,z,rh,rs,uh,us,vh,vs,wh,ws,eh,es = data.T
elif (data.shape[1]==3+5*1):
    x,y,z,rh,uh,vh,wh,eh = data.T

plt.figure(figsize=(8,7.2),dpi=200)
# plt.tripcolor(x,y,uh,cmap='jet',vmin=1-1e-5,vmax=1+1e-5)
plt.subplot(2,2,1)
# plt.tripcolor(x,y,(uh),cmap='jet',vmin=-1,vmax=1)
# plt.tripcolor(x,y,(uh),cmap='jet',vmin=-0.157914,vmax=0.157914)
# plt.tripcolor(x,y,(uh),cmap='jet',)
plt.tripcolor(x,y,(uh-us),cmap='jet',)
plt.colorbar()
# plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
plt.xticks(np.linspace(0,1,5),minor=False)
plt.xticks(np.linspace(0,1,21),minor=True)
plt.yticks(np.linspace(0,1,5),minor=False)
plt.yticks(np.linspace(0,1,21),minor=True)
plt.grid(which='both')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,2)
# plt.tripcolor(x,y,(vh),cmap='jet',vmin=-1,vmax=1)
# plt.tripcolor(x,y,(vh),cmap='jet',vmin=-0.157914,vmax=0.157914)
# plt.tripcolor(x,y,(vh),cmap='jet',)
plt.tripcolor(x,y,(vh-vs),cmap='jet',)
plt.colorbar()
# plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
plt.xticks(np.linspace(0,1,5),minor=False)
plt.xticks(np.linspace(0,1,21),minor=True)
plt.yticks(np.linspace(0,1,5),minor=False)
plt.yticks(np.linspace(0,1,21),minor=True)
plt.grid(which='both')
plt.xlabel('x')
plt.ylabel('y')


plt.subplot(2,2,3)
# plt.tripcolor(x,y,(wh),cmap='jet',vmin=-1,vmax=1)
plt.tripcolor(x,y,(wh),cmap='jet',)
plt.colorbar()
# plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
plt.xticks(np.linspace(0,1,5),minor=False)
plt.xticks(np.linspace(0,1,21),minor=True)
plt.yticks(np.linspace(0,1,5),minor=False)
plt.yticks(np.linspace(0,1,21),minor=True)
plt.grid(which='both')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,4)
# plt.tripcolor(x,y,(eh),cmap='jet',vmin=-1.25,vmax=1.25)
plt.tripcolor(x,y,(eh-es),cmap='jet',)
plt.colorbar()
# plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
plt.xticks(np.linspace(0,1,5),minor=False)
plt.xticks(np.linspace(0,1,21),minor=True)
plt.yticks(np.linspace(0,1,5),minor=False)
plt.yticks(np.linspace(0,1,21),minor=True)
plt.grid(which='both')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()