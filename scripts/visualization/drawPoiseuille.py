import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt


# data = np.loadtxt('./Poiseuille/Order_2/rho_T_%d_N_8.txt'%(50))

# if (data.shape[1]==3+5*2):
#     x,y,z,rh,rs,uh,us,vh,vs,wh,ws,eh,es = data.T
# elif (data.shape[1]==3+5*1):
#     x,y,z,rh,uh,vh,wh,eh = data.T

# plt.figure(figsize=(7,5),dpi=200)
# # # plt.tripcolor(x,y,uh,cmap='jet',vmin=1-1e-5,vmax=1+1e-5)
# # plt.subplot(2,2,1)
# # plt.tripcolor(x,y,(uh),cmap='jet',)
# # # plt.tripcolor(x,y,(uh),cmap='jet',vmin=-0.157914,vmax=0.157914)
# # # plt.tripcolor(x,y,(uh),cmap='jet',)
# # # plt.tripcolor(x,y,(uh-us),cmap='jet',)
# # plt.colorbar()
# # # plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
# # plt.xticks(np.linspace(0,1,5),minor=False)
# # plt.xticks(np.linspace(0,1,21),minor=True)
# # plt.yticks(np.linspace(0,1,5),minor=False)
# # plt.yticks(np.linspace(0,1,21),minor=True)
# # plt.grid(which='both')
# # plt.xlabel('x')
# # plt.ylabel('y')

# # plt.subplot(2,2,2)
# # plt.tripcolor(x,y,(vh),cmap='jet',)
# # # plt.tripcolor(x,y,(vh),cmap='jet',vmin=-0.157914,vmax=0.157914)
# # # plt.tripcolor(x,y,(vh),cmap='jet',)
# # # plt.tripcolor(x,y,(vh-vs),cmap='jet',)
# # plt.colorbar()
# # # plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
# # plt.xticks(np.linspace(0,1,5),minor=False)
# # plt.xticks(np.linspace(0,1,21),minor=True)
# # plt.yticks(np.linspace(0,1,5),minor=False)
# # plt.yticks(np.linspace(0,1,21),minor=True)
# # plt.grid(which='both')
# # plt.xlabel('x')
# # plt.ylabel('y')


# # plt.subplot(2,2,3)
# # # plt.tripcolor(x,y,(wh),cmap='jet',vmin=-1,vmax=1)
# # plt.tripcolor(x,y,(wh),cmap='jet',)
# # plt.colorbar()
# # # plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
# # plt.xticks(np.linspace(0,1,5),minor=False)
# # plt.xticks(np.linspace(0,1,21),minor=True)
# # plt.yticks(np.linspace(0,1,5),minor=False)
# # plt.yticks(np.linspace(0,1,21),minor=True)
# # plt.grid(which='both')
# # plt.xlabel('x')
# # plt.ylabel('y')

# # plt.subplot(2,2,4)
# # # plt.tripcolor(x,y,(eh),cmap='jet',vmin=-1.25,vmax=1.25)
# # plt.tripcolor(x,y,eh,cmap='jet',)
# # plt.colorbar()
# # # plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
# # plt.xticks(np.linspace(0,1,5),minor=False)
# # plt.xticks(np.linspace(0,1,21),minor=True)
# # plt.yticks(np.linspace(0,1,5),minor=False)
# # plt.yticks(np.linspace(0,1,21),minor=True)
# # plt.grid(which='both')
# # plt.xlabel('x')
# # plt.ylabel('y')



# ticks=178.24+0.2*np.arange(5)
# # h1=plt.tripcolor(x,y,es,cmap='jet',)
# h1=plt.tripcolor(x,y,rh*eh,cmap='jet',vmin=ticks.min(),vmax=ticks.max())
# h2=plt.tricontour(x,y,rh*eh,levels=ticks,colors='k',)

# # h1=plt.tripcolor(x,y,rh*eh-es,cmap='jet')
# # h2=plt.tricontour(x,y,rh*eh-es,colors='k',)
# plt.colorbar(h1)
# plt.clabel(h2,fontsize=10,colors='k',fmt='%.2f')
# # plt.title('$\\lg{\\left|\\rho_h(t) - \\rho(t)\\right|}$,  at t = %.2f'%(1.5))
# plt.xticks(np.linspace(0,1,5),minor=False)
# plt.xticks(np.linspace(0,1,21),minor=True)
# plt.yticks(np.linspace(0,1,5),minor=False)
# plt.yticks(np.linspace(0,1,21),minor=True)
# plt.grid(which='both')
# plt.xlabel('x')
# plt.ylabel('y')




# plt.tight_layout()
# plt.show()





# plt.figure(figsize=(6,3),dpi=200)
# for p in [1,]:
#     NN = []
#     VV = []
#     for N in [3,4,5,12,20]:
#         if (p*N) >= 400 : continue
#         NN.append(N)
#         VV.append(np.loadtxt('./Poiseuille/logging_%d_%d'%(p,N))[-1,6])
#     NN = np.array(NN)
#     VV = np.array(VV)
#     h = 1/NN
#     print(np.diff(np.log(VV))/np.diff(np.log(h)))
#     print(NN,VV)
#     plt.loglog(h,VV,'o-',label='P%d'%(p))
# NN = np.array([3,4,5,12])
# h = 1/NN
# # plt.loglog(NN,0.4*(2/NN)**1,'k--',label=r'slope=1')
# plt.loglog(h,0.004*(2/NN)**2,'k--',label=r'slope=2')
# # # plt.loglog(NN,0.4*(2/NN)**3,'k:',label=r'slope=3')
# # NN = np.array([5,10,20,])
# # plt.loglog(NN,0.2*(2/NN)**4,'k-.',label=r'slope=4')
# # plt.loglog(NN,0.4*(2/NN)**5,'k:',label=r'slope=5')
# # plt.loglog(NN,0.8*(2/NN)**6,'k-.',label=r'slope=6')
# plt.grid()
# plt.legend()
# plt.xlabel('h')
# plt.ylabel('error    $\|\\rho_h-\\rho\|/\|\\rho\|$')
# NN = np.array([3,4,5,12,20])
# h = 1/NN
# plt.xticks(h,h)
# plt.xticks([],minor=True)
# # plt.xlim()
# # plt.savefig('rho_order.png', format='png', bbox_inches='tight')
# # plt.savefig('rho_order.eps', format='eps', bbox_inches='tight')
# plt.show()








plt.figure(figsize=(6,3),dpi=200)
for p in [1,]:
    NN = []
    VV = []
    for N in [3,4,5,6,8,10,12]:
        if (p*N) >= 400 : continue
        data = np.loadtxt('./Poiseuille/logging_%d_%d'%(p,N))
        NN.append(N)
        VV.append(data[-1,5])
    NN = np.array(NN)
    VV = np.array(VV)
    h = 1/NN
    print(np.diff(np.log(VV))/np.diff(np.log(h)))
    # print(NN,VV)
    plt.loglog(h,VV,'o-',label='P%d'%(p))
    for xi,yi,si in zip(h[:-1]+0.8*(h[1:]-h[:-1]),(VV[1:]*VV[:-1])**0.5,np.diff(np.log(VV))/np.diff(np.log(h))):
        plt.text(xi,yi,'%.2f'%si)
for p in [2,]:
    NN = []
    VV = []
    for N in [3,4,5,6,8,10]:
        if (p*N) >= 400 : continue
        data = np.loadtxt('./Poiseuille/logging_%d_%d'%(p,N))
        NN.append(N)
        VV.append(data[-1,5])
    NN = np.array(NN)
    VV = np.array(VV)
    h = 1/NN
    print(np.diff(np.log(VV))/np.diff(np.log(h)))
    # print(NN,VV)
    plt.loglog(h,VV,'o-',label='P%d'%(p))
    for xi,yi,si in zip(h[:-1]+0.8*(h[1:]-h[:-1]),(VV[1:]*VV[:-1])**0.5,np.diff(np.log(VV))/np.diff(np.log(h))):
        plt.text(xi,yi,'%.2f'%si)
        
for p in [3,]:
    NN = []
    VV = []
    for N in [3,4,]:
        if (p*N) >= 400 : continue
        data = np.loadtxt('./Poiseuille/logging_%d_%d'%(p,N))
        NN.append(N)
        VV.append(data[-1,5])
    NN = np.array(NN)
    VV = np.array(VV)
    h = 1/NN
    print(np.diff(np.log(VV))/np.diff(np.log(h)))
    # print(NN,VV)
    plt.loglog(h,VV,'o-',label='P%d'%(p))
    for xi,yi,si in zip(h[:-1]+0.8*(h[1:]-h[:-1]),(VV[1:]*VV[:-1])**0.5,np.diff(np.log(VV))/np.diff(np.log(h))):
        plt.text(xi,yi,'%.2f'%si)
        
for p in [4,]:
    NN = []
    VV = []
    for N in [3,4,]:
        if (p*N) >= 400 : continue
        data = np.loadtxt('./Poiseuille/logging_%d_%d'%(p,N))
        NN.append(N)
        VV.append(data[-1,5])
    NN = np.array(NN)
    VV = np.array(VV)
    h = 1/NN
    print(np.diff(np.log(VV))/np.diff(np.log(h)))
    # print(NN,VV)
    plt.loglog(h,VV,'o-',label='P%d'%(p))
    for xi,yi,si in zip(h[:-1]+0.8*(h[1:]-h[:-1]),(VV[1:]*VV[:-1])**0.5,np.diff(np.log(VV))/np.diff(np.log(h))):
        plt.text(xi,yi,'%.2f'%si)
        
        
NN = np.array([3,4,5,12])
h = 1/NN
# plt.loglog(NN,0.4*(2/NN)**1,'k--',label=r'slope=1')
# plt.loglog(h,0.004*(2/NN)**2,'k--',label=r'slope=2')
# # plt.loglog(NN,0.4*(2/NN)**3,'k:',label=r'slope=3')
# NN = np.array([5,10,20,])
# plt.loglog(NN,0.2*(2/NN)**4,'k-.',label=r'slope=4')
# plt.loglog(NN,0.4*(2/NN)**5,'k:',label=r'slope=5')
# plt.loglog(NN,0.8*(2/NN)**6,'k-.',label=r'slope=6')
plt.grid()
plt.legend()
plt.xlabel('h')
plt.ylabel('error    $\|E_h-E\|/\|E\|$')
NN = np.array([2,3,4,5,8,12,16,20])
h = 1/NN
plt.xticks(h,['1/%d'%d for d in NN])
plt.xticks([],minor=True)
# plt.xlim()
# plt.savefig('rho_order.png', format='png', bbox_inches='tight')
# plt.savefig('rho_order.eps', format='eps', bbox_inches='tight')
plt.show()













# plt.figure(figsize=(6,3),dpi=200)
# for p in [2,]:
#     for N in [3,4,5,6,8,10]:
#         data = np.loadtxt('./Cprogram/FVE_DG/Poiseuille/logging_%d_%d'%(p,N))
#         plt.semilogy(data[:,0],data[:,5],label='N=%d'%(N))
        
        
# plt.legend()
# plt.show()