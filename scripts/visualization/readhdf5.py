import numpy as np
import scipy 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt 
import h5py
import numpy as np

# mpl.style.use('fast')

def read_DG_hdf5_solution(filename):
    with h5py.File(filename, 'r') as f:
        # 元信息
        time = f.attrs['time']
        step = f.attrs['step']
        
        # 加载数据
        basis_table = f['basis_table'][...]     # [N_plot, N_basis]
        plot_points = f['plot_points'][...]     # [N_plot, 3]
        vertices = f['vertices'][...]           # [N_cell, 4, 3]
        coeffs = f['coeffs'][...]               # [N_cell, 5, N_basis]

    N_cell = vertices.shape[0]
    N_plot = plot_points.shape[0]
    N_var = coeffs.shape[1]
    N_basis = coeffs.shape[2]

    # ========== 坐标变换（标准点 → 物理点） ==========
    # 形函数线性插值 [N_plot, 4]
    l1 = 1.0 - plot_points[:, 0] - plot_points[:, 1] - plot_points[:, 2]
    l2 = plot_points[:, 0]
    l3 = plot_points[:, 1]
    l4 = plot_points[:, 2]
    lambda_table = np.stack([l1, l2, l3, l4], axis=1)  # [N_plot, 4]

    # 重建物理坐标：broadcast → [N_cell, N_plot, 3]
    x_phys = np.einsum('pq,cqk->cpk', lambda_table, vertices)

    # ========== 解的重建（basis_table × coeffs） ==========
    # coeffs: [N_cell, 5, N_basis]
    # basis_table.T: [N_basis, N_plot]
    U = np.einsum('cvb,bp->cvp', coeffs, basis_table.T)  # [N_cell, 5, N_plot]

    # ========== 输出整理 ==========
    # [N_cell * N_plot]
    x = x_phys[:, :, 0]
    y = x_phys[:, :, 1]
    z = x_phys[:, :, 2]

    rho = U[:, 0, :]
    u   = U[:, 1, :] / rho
    v   = U[:, 2, :] / rho
    w   = U[:, 3, :] / rho
    E   = U[:, 4, :]
    p   = (E - 0.5 * rho * (u**2 + v**2 + w**2)) * (1.4 - 1.0)  # 假设 γ = 1.4

    return {
        'time': time,
        'step': step,
        'x': x, 'y': y, 'z': z,
        'rho': rho, 'u': u, 'v': v, 'w': w, 'E': E, 'p': p
    }

def read(stderr_file):
    with open(stderr_file, "r") as err:
        stderr_content = err.read()
        # 在这里对 stderr_content 进行处理
        sss = stderr_content.split('\n')[3:-2]
        results = {'time': [],'iteration': [],'value': []}
        for line in sss:
            parts = line.strip().split()
            time = float(parts[1])
            iteration = int(parts[-2])
            value = float(parts[-1])
            results['time'].append(time)
            results['iteration'].append(1+iteration)
            results['value'].append(value)
    return results

def get_solution(p,N,T=1,attr=None):
    if not attr:
        result_file = f'./Order_{p}_Mesh_{N}/solution/T_{T}_N_{N}'
    else:
        result_file = f'./Order_{p}_Mesh_{N}_' + attr + f'/solution/T_{T}_N_{N}'
    if os.path.exists(result_file + '.h5'):
        result_file += '.h5'
    elif os.path.exists(result_file + '.txt'):
        result_file += '.txt'
    else:
        return None
    #     #raise FileNotFoundError(f"Solution file {result_file} does not exist.")
    # data = np.loadtxt(result_file)
    # x,y,z, rh,rs, uh,us, vh,vs, wh,ws, eh,es = data.T
    # uh,vh,wh,eh = uh/rh, vh/rh, wh/rh, eh/rh
    # ph = 0.4*rh*(eh-0.5*uh**2-0.5*vh**2-0.5*wh**2)
    # ps = 0.4*rs*(es-0.5*us**2-0.5*vs**2-0.5*ws**2)
    solution = read_DG_hdf5_solution(result_file)
    x = solution['x'][:,:5].ravel()
    y = solution['y'][:,:5].ravel()
    z = solution['z'][:,:5].ravel()
    rho = solution['rho'][:,:5].ravel()
    u = solution['u'][:,:5].ravel()
    v = solution['v'][:,:5].ravel()
    w = solution['w'][:,:5].ravel()
    E = solution['E'][:,:5].ravel()
    p = solution['p'][:,:5].ravel()
    return x,y,z, rho, u, v, w, E, p, solution['time']