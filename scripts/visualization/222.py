import numpy as np
import matplotlib.pyplot as plt

init_x0 = 3
init_y0 = 3
param_phi = 5
param_gamma = 1.4
param_mu = 2e-3
velocity_u = 1.0
velocity_v = 0.5

def fe_xyz(x, y, z, t):
    cos2x = np.cos(2 * np.pi * x)
    cos4x = np.cos(4 * np.pi * x)
    cos6x = np.cos(6 * np.pi * x)
    cos2y = np.cos(2 * np.pi * y)
    cos4y = np.cos(4 * np.pi * y)
    cos6y = np.cos(6 * np.pi * y)
    
    exp24 = np.exp(-24 * np.pi * np.pi * t * param_mu) * np.pi / (param_gamma - 1)
    Cos24 = cos2x * cos2y * (cos4x - cos4y)
    
    exp16 = -4 * np.exp(-16 * np.pi * np.pi * t * param_mu) * np.pi * np.pi * param_mu
    Cos16 = 1 + cos4x * cos4y
    
    return exp24 * Cos24 + exp16 * Cos16



x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(x, y,indexing='ij')

plt.pcolor(X,Y,fe_xyz(X,Y,0.0,0.0),cmap='jet')
plt.colorbar()
plt.show()