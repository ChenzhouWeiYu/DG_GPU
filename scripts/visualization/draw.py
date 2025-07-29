import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.use('Agg')
plt.figure(figsize=(6,3),dpi=200)
for p in [0,1,2,3,4,5]:
    NN = []
    VV = []
    for N in [5,10,15,20,40,80,160]:
        if (p*N) >= 400 : continue
        NN.append(N)
        VV.append(np.loadtxt('./logging_%d_%d'%(p,N))[-1,1])
    print(NN,VV)
    plt.loglog(NN,VV,'o-',label='P%d'%(p))
NN = np.array([10,20,40,80,160])
# plt.loglog(NN,0.4*(2/NN)**1,'k--',label=r'slope=1')
plt.loglog(NN,0.4*(2/NN)**2,'k--',label=r'slope=2')
# plt.loglog(NN,0.4*(2/NN)**3,'k:',label=r'slope=3')
NN = np.array([5,10,20,])
plt.loglog(NN,0.2*(2/NN)**4,'k-.',label=r'slope=4')
plt.loglog(NN,0.4*(2/NN)**5,'k:',label=r'slope=5')
# plt.loglog(NN,0.8*(2/NN)**6,'k-.',label=r'slope=6')
plt.grid()
plt.legend()
plt.xlabel('N')
plt.ylabel('error    $\|\\rho_h-\\rho\|/\|\\rho\|$')
NN = np.array([5,10,20,40,80,160,])
plt.xticks(NN,NN)
plt.xticks([],minor=True)
plt.xlim(4,640)
plt.savefig('rho_order.png', format='png', bbox_inches='tight')
plt.savefig('rho_order.eps', format='eps', bbox_inches='tight')
plt.show()





plt.figure(figsize=(7,6),dpi=200)
for TT in range(1,1+4):
    data = np.loadtxt('./Order_2/rho_T_%d_NT_200_N_40.txt'%(TT*200))
    x,y,z,rho = data.T


    plt.subplot(2,2,TT)
    plt.tripcolor(x,y,rho,cmap='jet',vmin=0.36,vmax=1, rasterized=True)
    plt.colorbar()
    plt.title('$\\rho(t)$,  at t = %.2f'%(TT))
    plt.xticks(np.linspace(0,10,6),minor=False)
    plt.xticks(np.linspace(0,10,11),minor=True)
    plt.yticks(np.linspace(0,10,6),minor=False)
    plt.yticks(np.linspace(0,10,11),minor=True)
    plt.grid(which='both')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig('rho_solution.png', format='png', bbox_inches='tight')
plt.savefig('rho_solution.eps', format='eps', bbox_inches='tight')
plt.show()