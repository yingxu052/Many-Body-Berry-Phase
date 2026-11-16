import numpy as np

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_0 = np.eye(2)

def Bloch_Ham(kx, ky, m, t, Delta):
    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = (- m - 2 * (np.cos(kx) + np.cos(ky))) * sigma_z - t * (np.sin(kx) * sigma_x + np.sin(ky) * sigma_y)
    H[2:, 2:] = (m + 2 * (np.cos(kx) + np.cos(ky))) * sigma_z - t * (np.sin(kx) * sigma_x - np.sin(ky) * sigma_y)
    H[:2, 2:] = - Delta * (np.sin(kx) - 1j * np.sin(ky)) * sigma_z - 1j * 0.5 * Delta * sigma_y
    H[2:, :2] = np.conj(H[:2, 2:].T)
    return H

def chern_number_two_lowest(m, t, Delta, Nk=100):
    """
    使用 Fukui-Hatsugai-Suzuki 离散化方法计算最低两能级的总 Chern number。
    
    Parameters
    ----------
    m, t, Delta : float
        Bloch_Ham 的参数。
    Nk : int
        k 空间每个方向的离散网格数（默认 100）。
    
    Returns
    -------
    int
        最低两能级的 Chern number 之和（应为整数）。
    """
    kx = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    ky = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    
    chern = 0.0
    for i in range(Nk):
        for j in range(Nk):
            # plaquette 的四个角点
            k_pts = [
                (kx[i], ky[j]),
                (kx[(i + 1) % Nk], ky[j]),
                (kx[(i + 1) % Nk], ky[(j + 1) % Nk]),
                (kx[i], ky[(j + 1) % Nk]),
            ]
            
            # 获取每个角点上的最低两能级本征态
            states = []
            for kx_k, ky_k in k_pts:
                _, v = np.linalg.eigh(Bloch_Ham(kx_k, ky_k, m, t, Delta))
                states.append(v[:, :2])  # 4x2 矩阵，两列分别为基态和第一激发态
            
            # 计算 plaquette 上的 U(2) Berry phase
            U_plaq = np.eye(2, dtype=complex)
            for s in range(4):
                sp = (s + 1) % 4
                M = np.conj(states[s].T) @ states[sp]  # 2x2 重叠矩阵
                # SVD unitarization：V @ Wh 是 M 的最佳幺正近似
                V, _, Wh = np.linalg.svd(M)
                U_plaq = U_plaq @ (V @ Wh)
            
            # 累加该 plaquette 的 Berry phase（取行列式的幅角）
            chern += np.angle(np.linalg.det(U_plaq))
    
    return round(chern / (2 * np.pi))
