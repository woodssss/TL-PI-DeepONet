################################ Dta generation ac 2d ##################################
# consider (f^{n+1} - f^n) / \Delta t = d \nable f^{n+1} + k f^{n+1}(1-f^{n+1})
# with periodic BC
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm


if __name__ == "__main__":
    np.random.seed(1)
    np.random.RandomState(1)
    ##################### prepare/load training set ##############################################################
    # parameter

    # d = np.float32(sys.argv[1])
    # k = np.float32(sys.argv[2])
    # l = np.float32(sys.argv[3])

    d = 0.001
    k = 0.1
    l = 1

    dt = 0.01
    dt_sm = 5e-4
    st_sm = int(dt / dt_sm)
    mulp = 2
    lx = 1

    # define mesh size for x
    ## coarser grid
    Nx_c = 20
    dx_c = lx / Nx_c

    points_x_c = np.linspace(0, lx - dx_c, Nx_c).T
    x_c = points_x_c[:, None]

    ## finer grid
    Nx = Nx_c*mulp
    dx = lx / Nx
    points_x = np.linspace(0, lx - dx, Nx).T
    x = points_x[:, None]


    ### for y
    Ny, Ny_c = Nx, Nx_c
    points_y = points_x
    y = x

    points_y_c = points_x_c
    y_c = x_c

    xx, yy = np.meshgrid(x, y)
    xx_c, yy_c = np.meshgrid(x_c, y_c)

    st = 1000
    evo_step = 20

    Nb = 50
    Nte = 30
    Nst = Nb + Nte

    Corv=np.zeros((Nx*Ny, Nx*Ny))

    for i in range(Nx * Ny):
        for j in range(Nx * Ny):
            nx1 = i // Nx
            my1 = i - nx1 * Nx
            nx2 = j // Nx
            my2 = j - nx2 * Nx

            Corv[i, j] = np.exp(-(np.sin(np.pi * (points_x[nx1] - points_x[nx2]) / 1) ** 2 + np.sin(
                np.pi * (points_y[my1] - points_y[my2]) / 1) ** 2) / (2*l) ** 2)


    g_mat = np.zeros((Nst, Nx * Ny))
    mean = np.zeros((Nx * Ny,))

    for i in range(Nst):
        g_mat[[i], :] = (np.random.multivariate_normal(mean, Corv))


    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, g_mat[0, :].reshape(Nx, Ny).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, g_mat[2, :].reshape(Nx, Ny).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, g_mat[-1, :].reshape(Nx, Ny).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, g_mat[-1, :].reshape(Nx, Nx)[::mulp, ::mulp].T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # plt.show()

    ## compute reference solution
    Lap = np.zeros((Nx, Nx))
    for i in range(Nx):
        Lap[i][i] = -2

    for i in range(Nx - 1):
        Lap[i + 1][i] = 1
        Lap[i][i + 1] = 1

    Lap[0][1] = 1
    Lap[0][-1] = 1

    Lap[-1][-2] = 1
    Lap[-1][0] = 1

    Lx = Lap / dx ** 2
    Ly = Lap / dx ** 2

    L = np.kron(Lx, np.eye(Ny)) + np.kron(np.eye(Nx), Ly)

    def get_ref_f_one_step(st_sm, Nx, L, d, k, dt_sm, dx, f):
        for i in range(st_sm):
            f = f + dt_sm * d * L.dot(f) + dt_sm * k * f * (1-np.power(f,2))

        return f


    def get_ref_f_evo(st, Nx, L, d, k, dt, dx, f):
        f_evo_mat = np.zeros((st, Nx**2))
        f_tmp = f
        for i in range(st):
            f_new = get_ref_f_one_step(st_sm, Nx, L, d, k, dt_sm, dx, f_tmp)

            f_evo_mat[[i], :] = f_new.T

            f_tmp = f_new

        return f_evo_mat


    ################################## define Training set ######################################
    f_mat = g_mat

    Ns = Nb * evo_step

    Train_B_ori = f_mat[:Nb, :]
    Train_B = np.zeros((Ns, Nx**2))

    for i in range(Nb):
        print(i)
        Train_B[i * evo_step:(i + 1) * evo_step, :] = get_ref_f_evo(evo_step, Nx, L, d, k, dt, dx,
                                                                    Train_B_ori[i, :][:, None])


    Test_B = f_mat[Nb:, :]

    f_test = f_mat[[-1], :]

    f_test_evo_mat = get_ref_f_evo(st, Nx, L, d, k, dt, dx, f_test.T)

    all_test_evo_mat = np.zeros((Nte, st, Nx*Nx))

    all_test_f = f_mat[Nb:, :]

    for i in range(Nte):
        tmp_f = f_mat[Nb + i, :][:, None]

        f_tmp_evo_mat = get_ref_f_evo(st, Nx, L, d, k, dt, dx, tmp_f)

        all_test_evo_mat[i, :, :] = f_tmp_evo_mat

    ### back to coarser grid
    Train_B = Train_B.reshape(Ns, Nx, Nx)[:, ::mulp, ::mulp].reshape(Ns, Nx_c**2)
    np.random.shuffle(Train_B)

    Test_B = Test_B.reshape(Nte, Nx, Nx)[:, ::mulp, ::mulp].reshape(Nte, Nx_c**2)

    all_test_f = all_test_f.reshape(Nte, Nx, Nx)[:, ::mulp, ::mulp].reshape(Nte, Nx_c**2)
    all_test_evo_mat = all_test_evo_mat.reshape(Nte, st, Nx, Nx)[:, :, ::mulp, ::mulp].reshape(Nte, st, Nx_c**2)

    f_test = f_test.reshape(Nx, Nx)[::mulp, ::mulp].reshape(1, Nx_c**2)
    f_test_evo_mat = f_test_evo_mat.reshape(st, Nx, Nx)[:, ::mulp, ::mulp].reshape(st, Nx_c**2)

    ########### plot evolution  of sample ##############
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, f_test[0, :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, f_test_evo_mat[int(st / 2), :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, f_test_evo_mat[-1, :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, Train_B[evo_step - 1, :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(5)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, Train_B[evo_step * 2 - 1, :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')
    #
    # fig = plt.figure(6)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_c, yy_c, Train_B[evo_step * 3 - 1, :].reshape(Nx_c, Ny_c).T, cmap=cm.coolwarm)
    # plt.title('f')

    #plt.show()

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

    filename = 'A_sample_ac_2d_pb' + '_k_' + num2str_deciaml(k) + '_d_' + num2str_deciaml(d) + '_Nx_' + num2str_deciaml(
        Nx_c) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(Ns) + '_Nte_' + num2str_deciaml(
        Nte) + '_l_' + num2str_deciaml(l)
    npy_name = filename + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_c)
        np.save(ss, Train_B)
        np.save(ss, Test_B)

        np.save(ss, all_test_f)
        np.save(ss, all_test_evo_mat)