################################ Dta generation rte 1d ##################################
# consider (f^{n+1} - f^n) / \Delta t = <f^{n+1}> - f^{n+1}
# with inflow BC
# f(t,0,v<0)=1, f(t,1,v>0)=1/2
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":

    # define parameter
    Nb = 50
    Nte = 30
    Nst = Nb + Nte

    evo_step = 2

    dt = 0.01

    eps = 0.0001

    ## compute reference with smaller dt
    dt_sm = 1e-6
    st_sm = int(dt / dt_sm)

    st = 20

    # define mesh size for x
    ## finer grid
    lx = 1
    Nx = 65
    dx = lx / (Nx + 1)
    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]

    ## coarser grid
    Nx_c = 32
    dx_c = lx / (Nx_c + 1)

    points_x_c = np.linspace(dx_c, lx - dx_c, Nx_c).T
    x_c = points_x_c[:, None]

    # define mesh size for v
    lv = 1
    Nv = 16

    points_v, weights = np.polynomial.legendre.leggauss(Nv)
    points_v = lv * points_v
    weights = lv * weights
    v, w = np.float32(points_v[:, None]), np.float32(weights[:, None])

    xx, vv = np.meshgrid(x, v)

    Corv = np.zeros((Nx * Nv, Nx * Nv))

    l = 0.5

    for i in range(Nx * Nv):
        for j in range(Nx * Nv):
            nx1 = i // Nv
            lv1 = i - nx1 * Nv

            nx2 = j // Nv
            lv2 = j - nx2 * Nv

            Corv[i, j] = np.exp(-((points_x[nx1] - points_x[nx2]) ** 2 + (points_v[lv1] - points_v[lv2]) ** 2) / l ** 2)

    g_mat = np.zeros((Nst, Nx * Nv))
    mean = np.zeros((Nx * Nv,))

    for i in range(Nst):
        g_mat[[i], :] = (np.random.multivariate_normal(mean, Corv))

    # f = g * (relu(v)x + relu(-v)(1-x)) + (1-x)
    xv = np.kron(x, np.ones((Nv, 1)))
    vx = np.tile(v, (Nx, 1))

    eq = np.tile(1-0.5*xv.T, (Nst, 1))
    eq_tmp = 1-0.5*xv.T

    def relu(x):
        return (np.abs(x) + x)/2

    delta_mat = np.zeros((Nst, Nx*Nv))
    delta_mat_2 = np.zeros((Nst, Nx * Nv))
    var_mat = np.zeros((Nst, Nx * Nv))
    f_xv_mat = np.zeros((Nst, Nx * Nv))

    for i in range(Nst):
        if np.min(delta_mat[[i], :]) < -1:
            delta_mat[[i], :] = delta_mat[[i], :]/(-np.min(delta_mat[[i], :]))

    f_xv_mat = eq + 0.5 * delta_mat

    c2 = 1.5

    for i in range(Nst):
        var_tmp = np.random.multivariate_normal(mean, Corv)[None, :]* (1/4-(xv.T-1/2)**2)
        while np.min(f_xv_mat[[i], :] + c2*var_tmp) <0:
            var_tmp = np.random.multivariate_normal(mean, Corv)[None, :]* (1/4-(xv.T-1/2)**2)   # ensure the positivity

        f_xv_mat[[i], :] = f_xv_mat[[i], :] + c2* var_tmp


    print(np.min(f_xv_mat), np.max(f_xv_mat))


    ####### get rho and g

    ### f 2 rho
    def f2rho(Nx, f):
        rho = np.zeros((1, Nx))
        f_mat = f.reshape(Nx, Nv)
        for i in range(Nx):
            rho[0, i] = np.sum(f_mat[[i], :] * w.T) / 2

        return rho


    # f_mat 2 rho_mat and g_mat
    def get_rho_g(NN, Nx, f_mat):
        rho_mat = np.zeros((NN, Nx))
        rho_vec_mat = np.zeros((NN, Nx * Nv))
        g_mat = np.zeros((NN, Nx * Nv))

        for i in range(NN):
            tmp_f = f_mat[[i], :]
            tmp_rho = f2rho(Nx, tmp_f)

            tmp_rho_vec = np.kron(tmp_rho, np.ones((1, Nv)))

            tmp_g = (tmp_f - tmp_rho_vec) / eps

            rho_mat[[i], :] = tmp_rho
            rho_vec_mat[[i], :] = tmp_rho_vec
            g_mat[[i], :] = tmp_g

        return rho_mat, rho_vec_mat, g_mat


    def get_rho_g_st(NN, st, Nx, f_mat):
        rho_mat = np.zeros((NN, st, Nx))
        rho_vec_mat = np.zeros((NN, st, Nx * Nv))
        g_mat = np.zeros((NN, st, Nx * Nv))

        for i in range(NN):
            for j in range(st):
                tmp_f = f_mat[[i], [j], :]
                tmp_rho = f2rho(Nx, tmp_f)

                tmp_rho_vec = np.kron(tmp_rho, np.ones((1, Nv)))

                tmp_g = (tmp_f - tmp_rho_vec) / eps

                rho_mat[[i], [j], :] = tmp_rho
                rho_vec_mat[[i], [j], :] = tmp_rho_vec
                g_mat[[i], [j], :] = tmp_g

        return rho_mat, rho_vec_mat, g_mat



    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, vv, f_xv_mat[2, :].reshape(Nx, Nv).T, cmap=cm.coolwarm)
    plt.title('f_xv')

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, vv, f_xv_mat[1, :].reshape(Nx, Nv).T, cmap=cm.coolwarm)
    plt.title('f_xv')
    plt.show()

    xx, vv = np.meshgrid(x, v)
    xx_c, vv_c = np.meshgrid(x_c, v)

    ######## prepare training/test set #################
    f_mat = f_xv_mat


    ### reference for f
    ### define matrix
    Dp, Dm = np.zeros((Nx, Nx)), np.zeros((Nx, Nx))

    Vp, Vm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))

    for i in range(Nx):
        Dp[i][i] = 1
        Dm[i][i] = -1

    for i in range(Nx - 1):
        Dm[i][i + 1] = 1
        Dp[i + 1][i] = -1

    for i in range(int(Nv / 2)):
        Vp[i + int(Nv / 2)][i + int(Nv / 2)] = v[i + int(Nv / 2)]
        Vm[i][i] = v[i]

    Tp = np.kron(Dp, Vp)
    Tm = np.kron(Dm, Vm)

    T = (Tp + Tm) / dx

    w_mat = np.tile(w.T, (Nv, 1))/2

    L = np.kron(np.eye(Nx), w_mat) - np.eye(Nx * Nv)

    BC = np.zeros((Nx * Nv, 1))

    # two different boundary condition
    for i in range(int(Nv / 2)):
        BC[i + int(Nv / 2)] = v[i + int(Nv / 2)] / dx
        BC[(Nx-1)*Nv + i] = -0.5 * v[i] / dx

    P_sm = np.eye(Nx * Nv) - dt_sm/eps * T + dt_sm/eps**2 * L

    ###### get ref evo function with smaller step ##################
    def get_ref_f_one_step(f):
        for i in range(st_sm):
            f = np.matmul(P_sm, f) + dt_sm/eps * BC
        return f

    # def get_ref_f_evo(st, f):
    #     f_evo_mat = np.zeros((st, Nx * Nv))
    #     rho_evo_mat = np.zeros((st, Nx))
    #     f_tmp = f
    #     for i in range(st):
    #         f_next = get_ref_f_one_step(f_tmp)
    #
    #         f_evo_mat[[i], :] = f_next.T
    #
    #         f_next_mat = f_next.reshape(Nx, Nv)
    #
    #         for j in range(Nx):
    #             rho_evo_mat[i, j] = np.sum(f_next_mat[[j], :] * w.T) / 2
    #
    #         f_tmp = f_next
    #
    #     return rho_evo_mat, f_evo_mat

    def get_ref_f_evo(st, f):
        f_evo_mat = np.zeros((st, Nx * Nv))
        rho_evo_mat = np.zeros((st, Nx))

        f_tmp_mat = f.reshape(Nx, Nv)
        f_evo_mat[[0], :] = f.T
        for j in range(Nx):
            rho_evo_mat[0, j] = np.sum(f_tmp_mat[[j], :] * w.T) / 2

        f_tmp = f
        for i in range(st-1):
            f_next = get_ref_f_one_step(f_tmp)

            f_evo_mat[[i+1], :] = f_next.T

            f_next_mat = f_next.reshape(Nx, Nv)

            for j in range(Nx):
                rho_evo_mat[i+1, j] = np.sum(f_next_mat[[j], :] * w.T) / 2

            f_tmp = f_next

        return rho_evo_mat, f_evo_mat

    ############### compute reference on finer grid and smaller step #########

    Ns = Nb * evo_step

    Train_B_ori = f_mat[:Nb, :]
    Train_B = np.zeros((Ns, Nx * Nv))

    for i in range(Nb):
        _, Train_B[i * evo_step:(i + 1) * evo_step, :] = get_ref_f_evo(evo_step, Train_B_ori[i, :][:, None])


    Test_B = f_mat[Nb:, :]

    all_test_f_evo_mat = np.zeros((Nte, st, Nx * Nv))

    all_test_rho_evo_mat = np.zeros((Nte, st, Nx))

    all_test_f = f_mat[Nb:, :]

    all_test_rho = np.zeros((Nte, Nx))

    for i in range(Nte):
        tmp_f = f_mat[Nb + i, :][:, None]

        all_test_rho[i, :] = f2rho(Nx, tmp_f)

        rho_tmp_evo_mat, f_tmp_evo_mat = get_ref_f_evo(st, tmp_f)

        all_test_f_evo_mat[i, :, :] = f_tmp_evo_mat
        all_test_rho_evo_mat[i, :, :] = rho_tmp_evo_mat

    ###### come back to coaser grid
    Train_B_f = Train_B.reshape(Ns, Nx, Nv)[:, 1::2, :].reshape(Ns, Nx_c*Nv)
    Test_B_f = Test_B.reshape(Nte, Nx, Nv)[:, 1::2, :].reshape(Nte, Nx_c * Nv)

    Train_B_rho, Train_B_rho_vec, Train_B_g = get_rho_g(Ns, Nx_c, Train_B_f)
    Test_B_rho, Test_B_rho_vec, Test_B_g = get_rho_g(Nte, Nx_c, Test_B_f)

    all_test_rho = all_test_rho[:, 1::2]
    all_test_f = Test_B_f
    all_test_rho, all_test_rho_vec, all_test_g = get_rho_g(Nte, Nx_c, all_test_f)

    all_test_rho_evo_mat = all_test_rho_evo_mat[:, :, 1::2]
    all_test_f_evo_mat = all_test_f_evo_mat.reshape(Nte, st, Nx, Nv)[:, :, 1::2, :].reshape(Nte, st, Nx_c * Nv)
    all_test_rho_evo_mat, all_test_rho_vec_evo_mat, all_test_g_evo_mat = get_rho_g_st(Nte, st, Nx_c, all_test_f_evo_mat)

    ####################################################################

    ### plot

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, vv_c, all_test_f_evo_mat[0, 0, :].reshape(Nx_c, Nv).T, cmap=cm.coolwarm)
    # plt.title('f_xv')
    #
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, vv_c, all_test_f_evo_mat[0, -1, :].reshape(Nx_c, Nv).T, cmap=cm.coolwarm)
    # plt.title('f_xv')
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, vv_c, all_test_f_evo_mat[5, 0, :].reshape(Nx_c, Nv).T, cmap=cm.coolwarm)
    # plt.title('f_xv')
    #
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, vv_c, all_test_f_evo_mat[5, -1, :].reshape(Nx_c, Nv).T, cmap=cm.coolwarm)
    # plt.title('f_xv')
    #
    #
    # plt.show()



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


    filename = 'A_rte_sample_1d_rhog_ib_evo' + '_Nx_' + num2str_deciaml(Nx_c) + '_Nv_' + num2str_deciaml(Nv) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(Ns) + '_Nte_' + num2str_deciaml(Nte) + '_l_' + num2str_deciaml(l) + '_eps_' + num2str_deciaml(eps)
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


