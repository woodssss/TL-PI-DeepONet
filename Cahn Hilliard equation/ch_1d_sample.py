################################ Dta generation ac 1d   ##################################
# consider (f^{n+1} - f^n) / \Delta t = \partial_xx g^{n+1}
# g^{n+1} = -d_1 \partial_xx f^{n+1} + d_2 f^{n+1}((f^{n+1})^2-1)
# with periodic BC
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    np.random.seed(1)
    np.random.RandomState(1)

    # d1 = np.float32(sys.argv[1])
    # d2 = np.float32(sys.argv[2])
    # l = np.float32(sys.argv[3])

    d1 = 0.000001
    d2 = 0.001
    l = 0.5

    # define parameter
    Nb = 50
    Nte = 30
    Nst = Nb + Nte

    st = 1000
    evo_step = 20

    # parameter
    dt = 0.05

    ## compute reference with smaller dt
    dt_sm = 1e-4
    st_sm = int(dt / dt_sm)

    # define mesh size for x
    ## finer grid
    lx = 1
    Nx = 128
    dx = lx / Nx
    points_x = np.linspace(0, lx - dx, Nx).T
    x = points_x[:, None]

    ## coarser grid
    Nx_c = 64
    dx_c = lx / Nx_c

    points_x_c = np.linspace(0, lx - dx_c, Nx_c).T
    x_c = points_x_c[:, None]


    def kernal(xs, ys, l):
        dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
        return np.exp(-((np.sin(np.pi * dx) / l) ** 2) / 2)


    Corv = kernal(points_x, points_x, l)
    g_mat = np.zeros((Nst, Nx))
    mean = np.zeros_like(points_x)

    for i in range(Nst):
        g_mat[[i], :] = np.random.multivariate_normal(mean, Corv)

    for i in range(Nst):
        if np.max(np.abs(g_mat[[i], :])) > 2:
            g_mat[[i], :] = g_mat[[i], :] / np.max(np.abs(g_mat[[i], :]))


    f_mat = g_mat

    ## for nonlinear case
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

    L = Lap / dx ** 2

    def get_ref_f_one_step(st_sm, Nx, L, d1, d2, dt_sm, dx, f):
        for i in range(st_sm):
            g = -d1 * np.matmul(L, f) + d2 * (np.power(f, 3) - f)

            f = f + dt_sm * np.matmul(L, g)

        return f


    def get_ref_f_evo(st, Nx, L, d1, d2, dt, dx, f):
        f_evo_mat = np.zeros((st, Nx))
        f_tmp = f
        for i in range(st):
            f_new = get_ref_f_one_step(st_sm, Nx, L, d1, d2, dt_sm, dx, f_tmp)

            f_evo_mat[[i], :] = f_new.T

            f_tmp = f_new

        return f_evo_mat

    ################################## define Training set ######################################
    f_mat = g_mat

    Ns = Nb * evo_step

    Train_B_ori = f_mat[:Nb, :]
    Train_B = np.zeros((Ns, Nx))

    for i in range(Nb):
        Train_B[i * evo_step:(i + 1) * evo_step, :] = get_ref_f_evo(evo_step, Nx, L, d1, d2, dt, dx,
                                                                    Train_B_ori[i, :][:, None])

    Test_B = f_mat[Nb:, :]

    # define the test set
    f_test = f_mat[[-1], :]

    f_test_evo_mat = get_ref_f_evo(st, Nx, L, d1, d2, dt, dx, f_test.T)

    all_test_evo_mat = np.zeros((Nte, st, Nx))

    all_test_f = f_mat[Nb:, :]

    for i in range(Nte):
        tmp_f = f_mat[Nb + i, :][:, None]

        f_tmp_evo_mat = get_ref_f_evo(st, Nx, L, d1, d2, dt, dx, tmp_f)

        all_test_evo_mat[i, :, :] = f_tmp_evo_mat

    ### back to coarser grid
    Train_B = Train_B[:, ::2]
    np.random.shuffle(Train_B)

    Test_B = Test_B[:, ::2]

    all_test_f = all_test_f[:, ::2]
    all_test_evo_mat = all_test_evo_mat[:, :, ::2]

    f_test = f_test[:, ::2]
    f_test_evo_mat = f_test_evo_mat[:, ::2]


    ########### plot evolution  of sample ##############
    # plt.figure(1)
    # for i in range(Nb):
    #     plt.plot(x_c, Train_B[(i + 1) * evo_step - 1, :])
    #
    # plt.figure(2)
    #
    # plt.plot(x_c, f_test.T, 'r*')
    #
    # for i in range(st):
    #     plt.plot(x_c, f_test_evo_mat[i, :])
    #
    # plt.show()


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


    filename = 'A_ch_sample_1d_pb_evo' + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(
        d2) + '_Nx_' + num2str_deciaml(
        Nx_c) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(
        Nte) + '_l_' + num2str_deciaml(l)
    npy_name = filename + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x)
        np.save(ss, Train_B)
        np.save(ss, Test_B)

        np.save(ss, all_test_f)
        np.save(ss, all_test_evo_mat)
