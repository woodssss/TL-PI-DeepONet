################################ Dta generation rte 2d ##################################
# consider (f^{n+1} - f^n) / \Delta t = <f^{n+1}> - f^{n+1}
# with inflow BC
# f(t,0,y,v_x>0)=1/2-(y-1/2)^2, others = 1/2
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse


if __name__ == "__main__":
    # parameter
    Nb = 50
    Nte = 30
    Nst = Nb + Nte

    eps = 1

    dt = 0.01
    st = 200
    evo_step = 20

    ## smaller dt
    dt_sm = 0.001
    st_sm = int(dt/dt_sm)

    # define mesh size for x, y
    lx, ly = 1, 1
    NN = 24
    Nx, Ny = NN, NN
    dx, dy = lx / (Nx+1), ly / (Ny+1)

    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]
    points_y = np.linspace(dy, ly - dy, Ny).T
    y = points_y[:, None]

    xx, yy = np.meshgrid(x, y)

    # print(x[::2], x[::2].shape, x_c, x_c.shape)
    # zxc

    # define mesh size for v
    Nv = 16

    points_v, weights = np.polynomial.legendre.leggauss(Nv)
    points_v = (points_v + 1) * np.pi
    weights = weights * np.pi
    v, w = np.float32(points_v[:, None]), np.float32(weights[:, None])

    xy = np.kron(x, np.ones((Ny, 1)))
    yx = np.tile(y, (Nx, 1))

    eq = (1-xy)*(1/4-(yx-1/2)**2) + 1/4

    x_train = np.kron(x, np.ones((Ny * Nv, 1)))
    y_train = np.tile(np.kron(y, np.ones((Nv, 1))), (Nx, 1))
    v_train = np.tile(v, (Nx * Ny, 1))

    eq_vec = (1 - x_train.T) * (1 / 4 - (y_train.T - 1 / 2) ** 2) + 0.25

    ################ generate rho #########################

    l1 = 1
    l2 = 1

    def kernal(xs, ys, l):
        dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
        return np.exp(-((dx / l) ** 2) / 2)

    Corv_x = kernal(points_x, points_x, l1)
    Corv_y = kernal(points_y, points_y, l1)
    Corv_v = kernal(points_v, points_v, l2)

    mean_x = np.zeros_like(points_x)
    mean_y = np.zeros_like(points_y)
    mean_v = np.zeros_like(points_v)

    rho_mat = np.zeros((Nst, Nx * Ny))

    CC = 3

    for i in range(Nst):
        tmp_rho_x = np.random.multivariate_normal(mean_x, Corv_x)[:, None]
        tmp_rho_y = np.random.multivariate_normal(mean_y, Corv_y)[:, None]
        tmp_rho = np.kron(tmp_rho_x, tmp_rho_y)
        while np.min(CC * tmp_rho * xy *(1-xy) *yx*(1-yx) + eq) < 0:
            tmp_rho_x = np.random.multivariate_normal(mean_x, Corv_x)[:, None]
            tmp_rho_y = np.random.multivariate_normal(mean_y, Corv_y)[:, None]
            tmp_rho = np.kron(tmp_rho_x, tmp_rho_y)

        rho_mat[[i], :] = (CC * tmp_rho * xy *(1-xy) *yx*(1-yx) + eq).T

    rho_vec_mat = np.kron(rho_mat, np.ones((1, Nv)))

    fig = plt.figure(1)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_mat[0, :].reshape(Nx, Ny).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho')

    fig = plt.figure(2)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_mat[2, :].reshape(Nx, Ny).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho')

    fig = plt.figure(3)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_mat[Nb, :].reshape(Nx, Ny).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho')

    plt.show()

    ################### generate g #######################
    # define AVG
    w_mat = np.zeros((Nv, Nv))
    for i in range(Nv):
        w_mat[i, :] = 1 / 2 / np.pi * w.T

    AVG = sparse.kron(np.eye(Nx * Ny), w_mat)

    g_mat = np.zeros((Nst, Nx * Ny * Nv))
    f_mat = np.zeros((Nst, Nx * Ny * Nv))
    mean = np.zeros((Nx * Ny * Nv,))

    for i in range(Nst):
        tmp_g_x = np.random.multivariate_normal(mean_x, Corv_x)[:, None]
        tmp_g_y = np.random.multivariate_normal(mean_y, Corv_y)[:, None]
        tmp_g_v = np.random.multivariate_normal(mean_v, Corv_v)[:, None]

        tmp_g = np.kron(tmp_g_x, np.kron(tmp_g_y, tmp_g_v)) * x_train*(1-x_train)*y_train*(1-y_train)
        tmp_g = tmp_g - AVG.dot(tmp_g)
        tmp_f = rho_vec_mat[i, :][:, None] + tmp_g

        while np.min(tmp_f) < 0:
            tmp_g_x = np.random.multivariate_normal(mean_x, Corv_x)[:, None]
            tmp_g_y = np.random.multivariate_normal(mean_y, Corv_y)[:, None]
            tmp_g_v = np.random.multivariate_normal(mean_v, Corv_v)[:, None]

            tmp_g = np.kron(tmp_g_x, np.kron(tmp_g_y, tmp_g_v)) * x_train * (1 - x_train) * y_train * (1 - y_train)
            tmp_g = tmp_g - AVG.dot(tmp_g)
            tmp_f = rho_vec_mat[i, :][:, None] + tmp_g

        f_mat[[i], :] = tmp_f.T

    # use for transfer learning
    x_train = np.kron(x, np.ones((Ny * Nv, 1)))
    y_train = np.tile(np.kron(y, np.ones((Nv, 1))), (Nx, 1))
    v_train = np.tile(v, (Nx * Ny, 1))

    xxv, vv = np.meshgrid(x, v)

    xx, yy = np.meshgrid(x, y)

    f1 = f_mat[Nb, :].reshape(Nx, Ny, Nv)


    ### f 2 rho
    def f2rho_mat(f):
        rho = np.zeros((Nx, Ny))
        f_mat = f.reshape(Nx, Ny, Nv)
        for j in range(Nx):
            for k in range(Ny):
                rho[j, k] = np.sum(f_mat[[j], [k], :] * w.T) / 2 / np.pi
        return rho

    def f2rho(f):
        rho = np.zeros((Nx, Ny))
        f_mat = f.reshape(Nx, Ny, Nv)
        for j in range(Nx):
            for k in range(Ny):
                rho[j, k] = np.sum(f_mat[[j], [k], :] * w.T) / 2 / np.pi
        rho = rho.reshape(1, Nx*Ny)
        return rho


    ### define a test sample
    rho1 = f2rho_mat(f1)

    fig = plt.figure(1)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f1[:, int(Nx / 2), :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_xv')

    fig = plt.figure(2)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f1[:, 10, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_xv')

    fig = plt.figure(3)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f1[int(Nx / 2), :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_yv')

    fig = plt.figure(4)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f1[10, :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_yv')

    fig = plt.figure(5)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, f1[:, :, -1].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_xy')

    fig = plt.figure(6)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho1.T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho')

    plt.show()

    ########### define evo matrix
    N1, N2, N3 = np.where(v > np.pi / 2), np.where(v > np.pi), np.where(v > 3 * np.pi / 2)
    N1, N2, N3 = N1[0][0], N2[0][0], N3[0][0]

    # define T_x
    D_xp = np.zeros((Nx, Nx))
    D_xm = np.zeros((Nx, Nx))

    for i in range(Nx):
        D_xp[i, i] = 1
        D_xm[i, i] = -1

    for i in range(Nx - 1):
        D_xm[i, i + 1] = 1
        D_xp[i + 1, i] = -1

    I_y = np.eye(Ny)

    R_xm = np.kron(D_xm, I_y)
    R_xp = np.kron(D_xp, I_y)

    V_xm = np.zeros((Nv, Nv))
    V_xp = np.zeros((Nv, Nv))

    for i in range(N1, N3):
        V_xm[i, i] = np.cos(v[i]) / dx

    for i in range(N1):
        V_xp[i, i] = np.cos(v[i]) / dx

    for i in range(N3, Nv):
        V_xp[i, i] = np.cos(v[i]) / dx

    Tx = sparse.kron(R_xm, V_xm) + sparse.kron(R_xp, V_xp)

    # define Ty
    D_yp = np.zeros((Ny, Ny))
    D_ym = np.zeros((Ny, Ny))

    for i in range(Ny):
        D_yp[i, i] = 1
        D_ym[i, i] = -1

    for i in range(Ny - 1):
        D_ym[i, i + 1] = 1
        D_yp[i + 1, i] = -1

    I_x = np.eye(Nx)

    R_ym = np.kron(I_x, D_ym)
    R_yp = np.kron(I_x, D_yp)

    V_ym = np.zeros((Nv, Nv))
    V_yp = np.zeros((Nv, Nv))

    for i in range(N2):
        V_yp[i, i] = np.sin(v[i]) / dy

    for i in range(N2, Nv):
        V_ym[i, i] = np.sin(v[i]) / dy

    Ty = sparse.kron(R_ym, V_ym) + sparse.kron(R_yp, V_yp)

    T = Tx + Ty

    # define L
    w_mat = np.zeros((Nv, Nv))
    for i in range(Nv):
        w_mat[i, :] = 1 / 2 / np.pi * w.T

    L = sparse.kron(np.eye(Nx * Ny), w_mat)


    # deine BC
    CC = 0.25

    def get_bc(y):
        return (1/4-(y - 1/2)**2) + CC


    BCx = np.zeros((Nx * Ny * Nv, 1))
    BCx_tensor = np.zeros((Nx, Ny, Nv))

    # x = 0
    for j in range(Ny):
        for m in range(N1):
            BCx_tensor[0, j, m] = get_bc(y[j]) * np.cos(v[m]) / dx

        for m in range(N3, Nv):
            BCx_tensor[0, j, m] = get_bc(y[j]) * np.cos(v[m]) / dx

    # x = 1
    for j in range(Ny):
        for m in range(N1, N3):
            BCx_tensor[-1, j, m] = -CC * np.cos(v[m]) / dx

    BCy = np.zeros((Nx * Ny * Nv, 1))
    BCy_tensor = np.zeros((Nx, Ny, Nv))

    # y = 0
    for i in range(Nx):
        for m in range(N2):
            BCy_tensor[i, 0, m] = CC * np.sin(v[m]) / dx

    # y = 1
    for i in range(Nx):
        for m in range(N2, Nv):
            BCy_tensor[i, -1, m] = -CC * np.sin(v[m]) / dx

    BC_tensor = (BCx_tensor + BCy_tensor)
    BC = (BCx_tensor + BCy_tensor).reshape(Nx * Ny * Nv, 1)

    P_sm = sparse.eye(Nx * Ny * Nv) * (1-dt_sm) - dt_sm * T + dt_sm * L


    # print(P_sm.shape, f_mat.shape)
    # zxc

    ###### get ref evo function with smaller step ##################
    def get_ref_f_one_step(f):
        for i in range(st_sm):
            f = P_sm.dot(f) + dt_sm * BC
        return f


    def get_ref_f_evo(st, f):
        f_evo_mat = np.zeros((st, Nx * Ny * Nv))
        rho_evo_mat = np.zeros((st, Nx, Ny))
        f_tmp = f
        for i in range(st):
            f_next = get_ref_f_one_step(f_tmp)

            f_evo_mat[[i], :] = f_next.T

            f_next_mat = f_next.reshape(Nx, Ny, Nv)

            for j in range(Nx):
                for k in range(Ny):
                    rho_evo_mat[i, j, k] = np.sum(f_next_mat[[j], [k], :] * w.T) / 2 / np.pi

            f_tmp = f_next

        return rho_evo_mat, f_evo_mat



    def get_rho_g(NN, Nx, f_mat):
        rho_mat = np.zeros((NN, Nx* Ny))
        rho_vec_mat = np.zeros((NN, Nx * Ny * Nv))
        g_mat = np.zeros((NN, Nx * Ny * Nv))

        for i in range(NN):
            tmp_f = f_mat[[i], :]
            tmp_rho = f2rho(tmp_f)

            tmp_rho_vec = np.kron(tmp_rho, np.ones((1, Nv)))

            tmp_g = (tmp_f - tmp_rho_vec) / eps

            rho_mat[[i], :] = tmp_rho
            rho_vec_mat[[i], :] = tmp_rho_vec
            g_mat[[i], :] = tmp_g


        return rho_mat, rho_vec_mat, g_mat


    def get_rho_g_st(NN, st, Nx, f_mat):
        rho_mat = np.zeros((NN, st, Nx* Ny))
        rho_vec_mat = np.zeros((NN, st, Nx* Ny * Nv))
        g_mat = np.zeros((NN, st, Nx* Ny * Nv))

        for i in range(NN):
            for j in range(st):
                tmp_f = f_mat[[i], [j], :]
                tmp_rho = f2rho(tmp_f)

                tmp_rho_vec = np.kron(tmp_rho, np.ones((1, Nv)))

                tmp_g = (tmp_f - tmp_rho_vec) / eps

                rho_mat[[i], [j], :] = tmp_rho
                rho_vec_mat[[i], [j], :] = tmp_rho_vec
                g_mat[[i], [j], :] = tmp_g

        return rho_mat, rho_vec_mat, g_mat


    rho_mat, rho_vec_mat, g_mat = get_rho_g(Nst, Nx, f_mat)

    Ns = Nb * evo_step

    Train_B_ori = f_mat[:Nb, :]
    Train_B = np.zeros((Ns, Nx * Nx * Nv))

    #################### define the test set ############################
    # no random
    Test_B = f_mat[Nb:, :]

    for i in range(Nb):
        _, Train_B[i * evo_step:(i + 1) * evo_step, :] = get_ref_f_evo(evo_step, Train_B_ori[i, :][:, None])

    Train_B_f = Train_B
    Test_B_f = Test_B

    Train_B_rho, Train_B_rho_vec, Train_B_g = get_rho_g(Ns, Nx, Train_B_f)
    Test_B_rho, Test_B_rho_vec, Test_B_g = get_rho_g(Nte, Nx, Test_B_f)

    all_test_f = Test_B_f
    all_test_rho, all_test_rho_vec, all_test_g = get_rho_g(Nte, Nx, all_test_f)

    all_test_f_evo_mat = np.zeros((Nte, st, Nx * Ny * Nv))

    all_test_rho_evo_mat = np.zeros((Nte, st, Nx, Ny))

    for i in range(Nte):
        tmp_f = f_mat[Nb + i, :][:, None]

        all_test_rho[i, :] = f2rho(tmp_f)

        rho_tmp_evo_mat, f_tmp_evo_mat = get_ref_f_evo(st, tmp_f)

        all_test_f_evo_mat[i, :, :] = f_tmp_evo_mat
        all_test_rho_evo_mat[i, :, :, :] = rho_tmp_evo_mat

    all_test_rho_evo_mat, all_test_rho_vec_evo_mat, all_test_g_evo_mat = get_rho_g_st(Nte, st, Nx, all_test_f_evo_mat)

    f_test = f_mat[[Nb], :]

    rho_test = f2rho(f_test)

    rho_test_evo, f_test_evo = get_ref_f_evo(st, f_test.T)

    f_test_mat = f_test.reshape(Nx, Ny, Nv)
    f_test_end = f_test_evo[-1, :].reshape(Nx, Ny, Nv)

    fig = plt.figure(1)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_test.reshape(Nx, Ny).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho1')

    fig = plt.figure(2)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_test_evo[int(st / 2), :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho mid')

    fig = plt.figure(3)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, rho_test_evo[-1, :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('rho end')

    fig = plt.figure(4)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f_test_end[:, int(Nx / 2), :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_xv')

    fig = plt.figure(5)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f_test_end[:, 10, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_xv')

    fig = plt.figure(6)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f_test_end[int(Nx / 2), :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_yv')

    fig = plt.figure(7)
    ax = fig.add_subplot()
    cp = ax.contourf(xxv, vv, f_test_end[10, :, :].T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.title('f_yv')

    plt.show()

    ####################################################################
    # save model
    def num2str_deciaml(x):
        s = str(x)
        c = ''
        for i in range(len(s)):
            if s[i] == '0':
                c = c + 'z'
            elif s[i] == '.':
                c = c + 'p'
            elif s[i] == '-':
                c = c + 'n'
            else:
                c = c + s[i]

        return c


    filename = 'A_sample_RTE_2d_ib' + '_Nx_' + num2str_deciaml(Nx) + '_Nv_' + num2str_deciaml(
        Nv) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(Nte) + '_l1_' + num2str_deciaml(l1) + '_l2_' + num2str_deciaml(l2) + '_eps_' + num2str_deciaml(eps)
    npy_name = filename + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, Train_B_rho)
        np.save(ss, Train_B_g)
        np.save(ss, Train_B_f)

        np.save(ss, Test_B_rho)
        np.save(ss, Test_B_g)
        np.save(ss, Test_B_f)

        np.save(ss, all_test_rho)
        np.save(ss, all_test_g)
        np.save(ss, all_test_f)

        np.save(ss, all_test_rho_evo_mat)
        np.save(ss, all_test_g_evo_mat)
        np.save(ss, all_test_f_evo_mat)