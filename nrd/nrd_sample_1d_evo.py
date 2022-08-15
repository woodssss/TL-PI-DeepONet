import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize

if __name__ == "__main__":

    ##################### Data generation for 1d nonlinear reaction diffusion equation ##########################

    ## parameter

    d = np.float32(sys.argv[1])
    k = np.float32(sys.argv[2])
    l = np.float32(sys.argv[3])

    # d = 0.001
    # k = 0.001
    # l = 0.2

    ##################### prepare/load training set ##############################################################
    ## time step size
    dt = 0.05

    ## compute reference with smaller dt
    dt_sm = 5e-4
    st_sm = int(dt / dt_sm)

    ## define mesh size for x
    ## finer grid
    lx = 1
    Nx = 259
    dx = lx / (Nx + 1)
    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]

    ## coarser grid
    Nx_c = 64
    dx_c = lx / (Nx_c + 1)

    points_x_c = np.linspace(dx_c, lx - dx_c, Nx_c).T
    x_c = points_x_c[:, None]

    ################# data generation function ##############################
    # Guassian random field with

    st = 1000
    evo_step = 20

    Nb = 100
    Nte = 30
    Nst = Nb + Nte


    def kernal(xs, ys, l):
        dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
        return np.exp(-((dx / l) ** 2) / 2)


    Corv = kernal(points_x, points_x, l)
    g_mat = np.zeros((Nst, Nx))
    mean = np.zeros_like(points_x)
    for i in range(Nst):
        g_mat[[i], :] = (np.random.multivariate_normal(mean, Corv) * points_x * (1 - points_x)) # enforce zero B.C.

    for i in range(Nst):
        plt.plot(x, g_mat[i, :])

    plt.show()

    ## compute reference solution by FD method
    Lap = np.zeros((Nx, Nx))
    for i in range(Nx):
        Lap[i][i] = -2

    for i in range(Nx - 1):
        Lap[i + 1][i] = 1
        Lap[i][i + 1] = 1

    Lap[0][1] = 1
    Lap[-1][-2] = 1

    L = Lap / dx ** 2

    def get_ref_f_one_step(st_sm, Nx, L, d, k, dt_sm, dx, f):
        for i in range(st_sm):
            f = f + dt_sm * d * np.matmul(L, f) + dt_sm * k * f * f

        return f

    def get_ref_f_evo(st, Nx, L, d, k, dt, dx, f):
        f_evo_mat = np.zeros((st, Nx))
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
    Train_B = np.zeros((Ns, Nx))

    for i in range(Nb):
        Train_B[i * evo_step:(i + 1) * evo_step, :] = get_ref_f_evo(evo_step, Nx, L, d, k, dt, dx,
                                                                    Train_B_ori[i, :][:, None])


    Test_B = f_mat[Nb:, :]

    # define the test set
    f_test = f_mat[[-1], :]

    f_test_evo_mat = get_ref_f_evo(st, Nx, L, d, k, dt, dx, f_test.T)

    all_test_evo_mat = np.zeros((Nte, st, Nx))

    all_test_f = f_mat[Nb:, :]

    for i in range(Nte):
        tmp_f = f_mat[Nb + i, :][:, None]

        f_tmp_evo_mat = get_ref_f_evo(st, Nx, L, d, k, dt, dx, tmp_f)

        all_test_evo_mat[i, :, :] = f_tmp_evo_mat

    ### back to coarser grid
    Train_B = Train_B[:, 3::4]
    Test_B = Test_B[:, 3::4]

    all_test_f = all_test_f[:, 3::4]
    all_test_evo_mat = all_test_evo_mat[:, :, 3::4]

    f_test = f_test[:, 3::4]
    f_test_evo_mat = f_test_evo_mat[:, 3::4]

    plt.figure(1)
    for i in range(Nb):
        plt.plot(x_c, Train_B[(i + 1) * evo_step - 1, :])

    plt.figure(2)

    plt.plot(x_c, f_test.T, 'r*')

    for i in range(st):
        plt.plot(x_c, f_test_evo_mat[i, :])

    plt.show()


    ####################################################################
    # save training data set
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

    filename = 'A_nrd_sample_1d_one_evo' + '_k_' + num2str_deciaml(k) + '_d_' + num2str_deciaml(
        d) + '_Nx_' + num2str_deciaml(
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