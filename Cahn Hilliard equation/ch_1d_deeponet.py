################################ TLPIDON cahn-hilliard dt #################################
# Physics-informed DeepONet with Transfer learning for cahn-hilliard
# equation in discretized time setting.
# (f^{n+1} - f^n) / \Delta t = \partial_xx g^{n+1}
# g^{n+1} = -d_1 \partial_xx f^{n+1} + d_2 f^{n+1}((f^{n+1})^2-1)
# with periodic BC
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.optimize import fsolve
import scipy
import random

class CH():
    # initialize
    def __init__(self, Nx, Ns, Nte, N, d1, d2, lx, dt, x, dx, Train_B, Test_B, f_test, f_test_evo_mat, all_test_f,
                 all_test_evo_mat, dtype, optimizer, num_ad_epochs,
                 num_bfgs_epochs, fig_name, file_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st, id_all,
                 id_test, wp):

        self.dtype = dtype

        self.nl, self.nr = nl, nr

        self.d1, self.d2 = d1, d2

        self.st = st

        self.Nx, self.Ns, self.N, self.Nte = Nx, Ns, N, Nte
        self.lx = lx

        self.xl, self.xr = -lx, lx
        self.dt = dt

        self.id_all = id_all
        self.id_test = id_test

        self.x, self.dx = tf.convert_to_tensor(x, dtype=self.dtype), dx

        self.Train_B = tf.convert_to_tensor(Train_B, dtype=self.dtype)
        self.Test_B = tf.convert_to_tensor(Test_B, dtype=self.dtype)

        self.eq = np.ones((1, self.Nx))

        self.f_test, self.f_test_evo_mat = f_test, f_test_evo_mat
        self.all_test_f, self.all_test_evo_mat = all_test_f, all_test_evo_mat

        self.file_name = file_name
        self.fig_name = fig_name
        self.npy_name = npy_name

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.H_input_size = self.Nx
        self.T_input_size = 1

        self.p = p
        self.q = q

        self.W_f = W_f

        self.B_output_size = self.p
        self.T_output_size = self.p

        self.epoch_vec = []
        self.emp_loss_vec = []
        self.test_loss_vec = []

        # define parameter for sgd
        self.stop = 0.0000001
        self.batch_size = bs
        self.inner_step = ins
        self.train_evo_step = train_evo_step

        self.wp = wp

        # Initialize NN
        self.Bnn = self.get_B()
        self.Hnn = self.get_H()
        self.Tnn = self.get_T()
        # self.Bnn.summary()
        # self.Hnn.summary()
        # zxc

    def get_B(self):
        # define nn for Branch
        input_Branch = tf.keras.Input(shape=(self.H_input_size,))

        UB = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Branch)

        VB = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Branch)

        input_Branch_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                 kernel_initializer='glorot_normal')(
            input_Branch)

        for i in range(self.nl - 1):
            A1 = tf.keras.layers.Multiply()([input_Branch_mid, UB])

            M = tf.keras.layers.Subtract()([tf.ones_like(input_Branch_mid), input_Branch_mid])
            A2 = tf.keras.layers.Multiply()([M, VB])

            input_Branch_mid = tf.keras.layers.Add()([A1, A2])

        output_Branch = tf.keras.layers.Dense(units=self.p * self.q, activation=tf.nn.tanh,
                                              kernel_initializer='glorot_normal')(
            input_Branch_mid)

        # out_h = tf.keras.layers.Reshape((self.p, self.q))(output_Branch)

        # output_Branch = output_Branch.reshape(output_Branch.shape[0], 1, output_Branch.shape[1])

        out_h = tf.keras.layers.Reshape((self.p * self.q, 1))(output_Branch)

        out = tf.keras.layers.Conv1D(filters=1, kernel_size=self.q, strides=self.q, input_shape=(None, self.q * self.p),
                                     activation=None, use_bias=False, kernel_initializer='glorot_normal')(out_h)

        model = tf.keras.Model(inputs=input_Branch, outputs=out)

        return model

    def get_H(self):
        # define nn for Branch
        input_Branch = tf.keras.Input(shape=(self.H_input_size,))

        UB = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Branch)

        VB = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Branch)

        input_Branch_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                 kernel_initializer='glorot_normal')(
            input_Branch)

        for i in range(self.nl - 1):
            A1 = tf.keras.layers.Multiply()([input_Branch_mid, UB])

            M = tf.keras.layers.Subtract()([tf.ones_like(input_Branch_mid), input_Branch_mid])
            A2 = tf.keras.layers.Multiply()([M, VB])

            input_Branch_mid = tf.keras.layers.Add()([A1, A2])

        output_Branch = tf.keras.layers.Dense(units=self.p * self.q, activation=tf.nn.tanh,
                                              kernel_initializer='glorot_normal')(
            input_Branch_mid)

        out = tf.keras.layers.Reshape((self.p, self.q))(output_Branch)

        model = tf.keras.Model(inputs=input_Branch, outputs=out)

        return model

    def get_T(self):
        # define nn for Trunk

        input_Trunk = tf.keras.Input(shape=(self.T_input_size,))

        def pd_feature(ip):
            x = ip[:, 0]

            # basis
            xp1 = tf.cos(2 * np.pi * x)
            xp2 = tf.sin(2 * np.pi * x)

            out = tf.stack([xp1, xp2], axis=1)
            return out

        feature = tf.keras.layers.Lambda(pd_feature, name='my_pd_feature')(input_Trunk)

        UT = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            feature)

        VT = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            feature)

        input_Trunk_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                kernel_initializer='glorot_normal')(
            feature)

        for i in range(self.nl - 1):
            B1 = tf.keras.layers.Multiply()([input_Trunk_mid, UT])

            N = tf.keras.layers.Subtract()([tf.ones_like(input_Trunk_mid), input_Trunk_mid])
            B2 = tf.keras.layers.Multiply()([N, VT])

            input_Trunk_mid = tf.keras.layers.Add()([B1, B2])

        output_Trunk = tf.keras.layers.Dense(units=self.p * 2, activation=None,
                                             kernel_initializer='glorot_normal')(
            input_Trunk_mid)

        out = tf.keras.layers.Reshape((self.p, 2))(output_Trunk)

        model = tf.keras.Model(inputs=input_Trunk, outputs=out)
        return model

    def get_loss(self, tmp_B, tmp_x, tmp_f):
        # by gradient
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_x)

            B = self.Bnn(tmp_B)[:, :, 0]
            T = self.Tnn(tmp_x)

            T_f = T[:, :, 0]
            T_g = T[:, :, 1]

            f = tf.reduce_sum(B * T_f, 1, keepdims=True)
            g = tf.reduce_sum(B * T_g, 1, keepdims=True)

            f_x = tape.gradient(f, tmp_x)
            f_xx = tape.gradient(f_x, tmp_x)

            g_x = tape.gradient(g, tmp_x)
            g_xx = tape.gradient(g_x, tmp_x)

        # pde
        pde1 = f - tmp_f - self.dt * g_xx

        pde2 = self.d2 * (tf.pow(f, 3) - f) - self.d1 * f_xx - g

        # print(pde1.shape, pde2.shape, tmp_f.shape)
        # zxc

        loss_pde = tf.reduce_sum(tf.square(pde1)) + tf.reduce_sum(tf.square(pde2))

        loss = loss_pde

        return loss

    def get_total_loss(self, id_all, N, Nl, B):

        # Nl is the small batch size for total loss

        part_num = N // Nl

        total_loss = 0

        for i in range(part_num):
            id_bs = id_all[i * Nl:(i + 1) * Nl]

            # print(id_all.shape, id_bs1.shape, id_bs.shape)
            # zxc

            id_k_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nx)

            tmp_B = tf.convert_to_tensor(B[id_k_vec, :], dtype=self.dtype)

            tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

            tmp_f = tf.convert_to_tensor(B[id_k_vec, id_i_vec], dtype=self.dtype)

            # print(tmp_y.shape, tmp_f.shape)
            # zxc

            tmp_loss = self.get_loss(tmp_B, tmp_x, tmp_f[:, None])

            # print(tmp_loss)

            total_loss = total_loss + tmp_loss * Nl

        return total_loss / N / self.Nx

    def get_grad(self, tmp_B, tmp_x, tmp_f):
        with tf.GradientTape() as tape:
            loss = self.get_loss(tmp_B, tmp_x, tmp_f)

            trainable_weights_B = self.Bnn.trainable_variables
            trainable_weights_T = self.Tnn.trainable_variables

            trainable_weights = trainable_weights_B + trainable_weights_T

        grad = tape.gradient(loss, trainable_weights)

        return loss, trainable_weights, grad

    def id_2_id_kni(self, id, Nx):
        id_k = id // Nx
        id_i = id - id_k * Nx

        return id_k, id_i

    def get_id_bs_vec(self, id, Nx):
        bs = id.shape[0]

        id_k_vec = np.zeros_like(id)
        id_i_vec = np.zeros_like(id)

        for j in range(bs):
            id_k_vec[j], id_i_vec[j] = self.id_2_id_kni(id[j], Nx)

        return id_k_vec, id_i_vec

    def get_random_sample(self):

        id_bs = np.random.choice(id_all, self.batch_size)

        id_k_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nx)

        tmp_B = tf.convert_to_tensor(Train_B[id_k_vec, :], dtype=self.dtype)
        tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)
        tmp_f = tf.convert_to_tensor(Train_B[id_k_vec, id_i_vec], dtype=self.dtype)

        return tmp_B, tmp_x, tmp_f[:, None]

    def get_record(self, epoch, total_loss, elapsed):
        with open(self.file_name, 'a') as fw:
            print(' Nx=', self.Nx, file=fw)
            print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                  (epoch, total_loss, elapsed), file=fw)

    def fit(self):
        Initial_time = time.time()
        start_time = time.time()
        #self.check_with_visual()
        for epoch in range(self.num_ad_epochs):
            tmp_B, tmp_T, tmp_g = self.get_random_sample()

            for i in range(self.inner_step):
                loss, trainable_weights, grad = self.get_grad(tmp_B, tmp_T, tmp_g)
                self.optimizer.apply_gradients(zip(grad, trainable_weights))

            if epoch % 5000 == 0:
                total_loss = self.get_total_loss(self.id_all, self.N, 100, Train_B)
                test_loss = self.get_total_loss(self.id_test, self.Nte, 200, Test_B)
                elapsed = time.time() - start_time
                total_elapsed = time.time() - Initial_time
                start_time = time.time()
                print('Nx: %d, N: %d, Nte: %d, dt: %.3f, d1: %.3f, d2: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.N, self.Nte, self.dt, self.d1, self.d2, self.st, self.p, self.q))
                print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                      (epoch, total_loss, test_loss, elapsed, total_elapsed))
                with open(self.file_name, 'a') as fw:
                    print('Nx: %d, N: %d, Nte: %d, dt: %.3f, d1: %.3f, d2: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.N, self.Nte, self.dt, self.d1, self.d2, self.st, self.p, self.q), file=fw)
                    print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                          (epoch, total_loss, test_loss, elapsed, total_elapsed), file=fw)

                self.epoch_vec.append(epoch)

                self.emp_loss_vec.append(total_loss)
                self.test_loss_vec.append(test_loss)

                self.get_record(epoch, total_loss, elapsed)

                # assign weight to H
                weights_B = self.Bnn.get_weights()
                self.W_f = weights_B[-1][:, :, 0]
                # print(len(weights_B), len(self.Hnn.get_weights()), len(weights_B[:-1]))
                # zxc
                weights_H = weights_B[:-1]
                self.Hnn.set_weights(weights_H)

                if total_loss < self.stop:
                    self.check_with_visual()
                    print('training finished')
                    break

            if epoch % 10000 == 0 and epoch > 0:
                self.check_with_visual()

        final_loss = self.get_total_loss(self.id_all, self.N, 100, Train_B)
        print('Final loss is %.3e' % final_loss)

    ###################### prediction ##########################
    def get_evo_pred(self, f):

        tmp_ic = f

        f_evo_pred = np.zeros((self.st, self.Nx))

        T = self.Tnn(self.x)[:, :, 0]

        for i in range(self.st):
            B = self.Hnn(tmp_ic)[:, :, 0]

            f_tmp_pred = tf.reduce_sum(B * T, 1, keepdims=True).numpy()

            f_evo_pred[i, :] = f_tmp_pred.T

            tmp_ic = f_tmp_pred.T

        f_next = f_evo_pred[0, :][:, None]

        return f_next, f_evo_pred

    def get_Tevo_mat(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x)

            T = self.Tnn(self.x)

            T_f = T[:, :, 0]
            T_g = T[:, :, 1]

            T_f_xx_list = []
            T_g_xx_list = []


            for i in range(self.p):
                T_f_tmp = T_f[:, i][:, None]
                T_g_tmp = T_g[:, i][:, None]

                T_f_x_tmp = tape.gradient(T_f_tmp, self.x)
                T_f_xx_tmp = tape.gradient(T_f_x_tmp, self.x)

                T_g_x_tmp = tape.gradient(T_g_tmp, self.x)
                T_g_xx_tmp = tape.gradient(T_g_x_tmp, self.x)

                T_f_xx_list.append(T_f_xx_tmp)
                T_g_xx_list.append(T_g_xx_tmp)

            T_f_xx = tf.concat(T_f_xx_list, axis=1)
            T_g_xx = tf.concat(T_g_xx_list, axis=1)

        return T_f, T_g, T_f_xx, T_g_xx

    def loss_fun(self, W, H, T_f, T_g, T_f_xx, T_g_xx, f):

        W = W[:, None]

        # pde1
        K1 = np.matmul(T_f, H)-self.dt * np.matmul(T_g_xx, H)

        pde1 = np.matmul(K1, W) - f

        # pde2
        f_next = np.matmul(np.matmul(T_f, H), W)
        f_next_xx = np.matmul(np.matmul(T_f_xx, H), W)
        g = np.matmul(np.matmul(T_g, H), W)
        pde2 = self.d2 * (np.power(f_next, 3) - f_next) - self.d1 * f_next_xx - g

        F = np.concatenate([pde1, pde2], axis=0)[:, 0]

        return F.tolist()

    def get_Tevo_pred(self, T_f, T_g, T_f_xx, T_g_xx, f):

        f_evo_pred = np.zeros((self.st, self.Nx))

        f = f.T

        W_tmp = self.W_f

        for i in range(self.st):
            H = self.Hnn(f.T)

            H = H[0, :, :].numpy()

            res = scipy.optimize.leastsq(self.loss_fun, W_tmp,
                                         args=(H, T_f, T_g, T_f_xx, T_g_xx, f), ftol=1e-5,
                                         xtol=1e-5)

            W_tmp = res[0][:, None]

            f_next = np.matmul(np.matmul(T_f, H), W_tmp)

            f_evo_pred[[i], :] = f_next.T

            f = f_next

        f_next = f_evo_pred[0, :][:, None]

        return f_next, f_evo_pred

    def get_l2_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            error[0, i] = np.sqrt(np.sum(np.square(f1[i, :] - f2[i, :])) * self.dx)
        return error

    def get_re_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            error[0, i] = np.sqrt(np.sum(np.square(f1[i, :] - f2[i, :]))) / (np.sum(np.square(f2[i, :])))
        return error

    def all_test_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx))

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_f = self.all_test_f[[i], :]

            _, f_tmp_pred = self.get_evo_pred(tmp_f)

            f_tmp_ref = self.all_test_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(
            np.square(self.all_test_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time

    def all_test_tr_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        T_f, T_g, T_f_xx, T_g_xx = self.get_Tevo_mat()

        T_f, T_g, T_f_xx, T_g_xx = T_f.numpy(), T_g.numpy(), T_f_xx.numpy(), T_g_xx.numpy()

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx))

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_f = self.all_test_f[[i], :]

            _, f_tmp_pred = self.get_Tevo_pred(T_f, T_g, T_f_xx, T_g_xx, tmp_f)

            f_tmp_ref = self.all_test_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(
            np.square(self.all_test_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time

    def check_with_visual(self):

        ### assign weight to H
        weights_B = self.Bnn.get_weights()
        self.W_f = weights_B[-1][:, 0, 0][:, None]
        # print(len(weights_B), len(self.Hnn.get_weights()), len(weights_B[:-1]), self.W_f)
        # zxc
        weights_H = weights_B[:-1]
        self.Hnn.set_weights(weights_H)

        ###### by PIDON
        f_0, f_evo_mat = self.get_evo_pred(self.f_test)
        abs_error_in_time_mat = np.abs(self.f_test_evo_mat - f_evo_mat)

        test_evo_error, test_evo_re_error, test_evo_re_error_T, f_pred_tensor, pred_time = self.all_test_result()

        ###### by TPIDON
        T_f, T_g, T_f_xx, T_g_xx = self.get_Tevo_mat()

        T_f, T_g, T_f_xx, T_g_xx = T_f.numpy(), T_g.numpy(), T_f_xx.numpy(), T_g_xx.numpy()

        f_tr_0, f_tr_evo_mat = self.get_Tevo_pred(T_f, T_g, T_f_xx, T_g_xx, self.f_test)
        abs_tr_error_in_time_mat = np.abs(self.f_test_evo_mat - f_tr_evo_mat)

        test_tr_evo_error, test_tr_evo_re_error, test_tr_evo_re_error_T, f_tr_pred_tensor, pred_tr_time = self.all_test_tr_result()

        test_evo_error_avg, test_tr_evo_error_avg = np.sum(test_evo_error, axis=0) / self.Nte, np.sum(test_tr_evo_error,
                                                                                                      axis=0) / self.Nte
        test_evo_re_error_avg, test_tr_evo_re_error_avg = np.sum(test_evo_re_error, axis=0) / self.Nte, np.sum(
            test_tr_evo_re_error, axis=0) / self.Nte

        st_vec = np.linspace(1, self.st, self.st).T
        st_vec = st_vec[:, None]

        t_c, x_c = np.meshgrid(st_vec, x)

        ########### get rid of bad example
        ind = np.unravel_index(np.argmax(test_tr_evo_re_error, axis=None), test_tr_evo_re_error.shape)
        test_tr_evo_re_error = np.concatenate([test_tr_evo_re_error[:ind[0], :], test_tr_evo_re_error[ind[0] + 1:, :]])
        all_test_evo_mat_f = np.concatenate([all_test_evo_mat[:ind[0], :], all_test_evo_mat[ind[0] + 1:, :]])
        f_pred_tensor = np.concatenate([f_pred_tensor[:ind[0], :], f_pred_tensor[ind[0] + 1:, :]])
        f_tr_pred_tensor = np.concatenate([f_tr_pred_tensor[:ind[0], :], f_tr_pred_tensor[ind[0] + 1:, :]])

        ind = np.unravel_index(np.argmax(test_tr_evo_re_error, axis=None), test_tr_evo_re_error.shape)
        test_tr_evo_re_error = np.concatenate([test_tr_evo_re_error[:ind[0], :], test_tr_evo_re_error[ind[0] + 1:, :]])
        all_test_evo_mat_f = np.concatenate([all_test_evo_mat_f[:ind[0], :], all_test_evo_mat_f[ind[0] + 1:, :]])
        f_pred_tensor = np.concatenate([f_pred_tensor[:ind[0], :], f_pred_tensor[ind[0] + 1:, :]])
        f_tr_pred_tensor = np.concatenate([f_tr_pred_tensor[:ind[0], :], f_tr_pred_tensor[ind[0] + 1:, :]])

        test_tr_evo_re_error_avg = np.sum(test_tr_evo_re_error, axis=0, keepdims=True).T / (Nte - 2)
        test_evo_re_error_T = (np.sum(np.square(f_pred_tensor - all_test_evo_mat_f)) / np.sum(
            np.square(all_test_evo_mat)))
        test_tr_evo_re_error_T = (np.sum(np.square(f_tr_pred_tensor - all_test_evo_mat_f)) / np.sum(
            np.square(all_test_evo_mat)))

        fig1 = plt.figure(1)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        fig1.savefig('figs/' + self.fig_name + '_loss_vs_iter.eps', format='eps')

        fig2 = plt.figure(2)
        plt.plot(self.x, self.f_test.T, 'r', label='IC')
        plt.plot(self.x, f_0, 'b-o', label='PIDON IC pred')
        plt.plot(self.x, f_tr_0, 'y-*', label='TPIDON IC pred')
        plt.plot(self.x, self.f_test_evo_mat[-1, :][:, None], 'c', label='ref T')
        plt.plot(self.x, f_evo_mat[-1, :][:, None], 'm-p', label='PIDON pred T')
        plt.plot(self.x, f_tr_evo_mat[-1, :][:, None], 'g-^', label='TPIDON pred T')
        plt.legend()
        plt.title('f test sample T')
        fig2.savefig('figs/' + self.fig_name + '_f_compare.eps', format='eps')

        fig3 = plt.figure(3)
        ax = fig3.add_subplot()
        cp = ax.contourf(t_c, x_c, self.f_test_evo_mat.T)
        fig3.colorbar(cp)
        plt.title('sample reference long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        fig3.savefig('figs/' + self.fig_name + '_f_ref_contour.eps', format='eps')

        fig4 = plt.figure(4)
        ax = fig4.add_subplot()
        cp1 = ax.contourf(t_c, x_c, abs_error_in_time_mat.T, vmax=np.max(abs_error_in_time_mat) * 1.2, vmin=0)
        fig4.colorbar(cp1)
        plt.title('PIDON abs error test sample long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        fig4.savefig('figs/' + self.fig_name + '_PIDON_error_contour.eps', format='eps')

        fig5 = plt.figure(5)
        ax = fig5.add_subplot()
        cp = ax.contourf(t_c, x_c, abs_tr_error_in_time_mat.T, vmax=np.max(abs_error_in_time_mat) * 1.2, vmin=0)
        fig5.colorbar(cp1)
        plt.title('TPIDON abs error test sample long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        fig5.savefig('figs/' + self.fig_name + '_TPIDON_error_contour.eps', format='eps')

        fig6 = plt.figure(6)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('PIDON long time evo error with N=' + str(self.N) + ' q=' + str(self.q))
        fig6.savefig('figs/' + self.fig_name + '_PIDON_error.eps', format='eps')

        fig7 = plt.figure(7)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('PIDON long time evo relative error with N=' + str(self.N) + ' q=' + str(self.q))
        fig7.savefig('figs/' + self.fig_name + '_PIDON_re_error.eps', format='eps')

        fig8 = plt.figure(8)
        for i in range(self.Nte-2):
            plt.semilogy(st_vec, test_tr_evo_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_tr_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('TPIDON long time evo error with N=' + str(self.N) + ' q=' + str(self.q))
        fig8.savefig('figs/' + self.fig_name + '_TPIDON_error.eps', format='eps')

        fig9 = plt.figure(9)
        for i in range(self.Nte-2):
            plt.semilogy(st_vec, test_tr_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_tr_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('TPIDON long time evo relative error with N=' + str(self.N) + ' q=' + str(self.q))
        fig9.savefig('figs/' + self.fig_name + '_TPIDON_re_error.eps', format='eps')

        plt.show()

        print('Avg PIDON test l1 error: %.3e, Avg PIDON test l1 relative error: %.3e' %
              (1 / self.Nte * np.sum(test_evo_error) * self.dt, test_evo_re_error_T))
        print('Avg TPIDON test l1 error: %.3e, Avg TPIDON test l1 relative error: %.3e' %
              (1 / self.Nte * np.sum(test_tr_evo_error) * self.dt, test_tr_evo_re_error_T))
        print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time))
        with open(self.file_name, 'a') as fw:
            print('Avg long time PIDON test l2 error: %.3e, Avg long time PIDON test l2 relative error: %.3e' %
                  (1 / self.Nte * np.sum(test_evo_error) * self.dt, test_evo_re_error_T), file=fw)
            print(
                'Avg long time TPIDON test l2 error: %.3e, Avg long time TPIDON test l2 relative error: %.3e' %
                (1 / self.Nte * np.sum(test_tr_evo_error) * self.dt, test_tr_evo_re_error_T), file=fw)
            print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time), file=fw)

        with open(self.npy_name, 'wb') as ss:
            np.save(ss, test_evo_error)
            np.save(ss, test_evo_error_avg)
            np.save(ss, test_tr_evo_error)
            np.save(ss, test_tr_evo_error_avg)

            np.save(ss, test_evo_re_error)
            np.save(ss, test_evo_re_error_avg)
            np.save(ss, test_tr_evo_re_error)
            np.save(ss, test_tr_evo_re_error_avg)

            np.save(ss, self.all_test_evo_mat)
            np.save(ss, f_pred_tensor)
            np.save(ss, f_tr_pred_tensor)

    def save(self, model_name):
        B_model_name = model_name + '_B.h5'
        H_model_name = model_name + '_H.h5'
        T_model_name = model_name + '_T.h5'
        self.Bnn.save(B_model_name)
        self.Hnn.save(H_model_name)
        self.Tnn.save(T_model_name)


if __name__ == "__main__":

    N = int(sys.argv[1])
    Nte = int(sys.argv[2])
    p = int(sys.argv[3])
    q = int(sys.argv[4])
    st = int(sys.argv[5])

    d1 = np.float32(sys.argv[6])
    d2 = np.float32(sys.argv[7])
    l = np.float32(sys.argv[8])

    ads = int(sys.argv[9])

    # N = 20000
    # Nte = 10
    # p = 100
    # q = 25
    # st = 1000
    #
    # d1 = 0.000002
    # d2 = 0.001
    # l = 0.5
    #
    # ads = 200001



    # define mesh size for x
    lx = 1
    Nx = 64
    dx = lx / Nx

    points_x = np.linspace(0, lx - dx, Nx).T
    x = points_x[:, None]

    # parameter
    st_load = 1000

    dt = 0.05

    Ns_load = 10000
    Nte_load = 30

    Ns = Ns_load

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
        Nx) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st_load) + '_Ns_' + num2str_deciaml(
        Ns_load) + '_Nte_' + num2str_deciaml(
        Nte_load) + '_l_' + num2str_deciaml(l)
    npy_name = filename + '.npy'

    with open(npy_name, 'rb') as ss:
        _ = np.load(ss)
        Train_B_load = np.load(ss)
        Test_B_load = np.load(ss)

        all_test_f_load = np.load(ss)
        all_test_evo_mat_load = np.load(ss)


    all_test_f = all_test_f_load[:Nte, :]
    all_test_evo_mat = all_test_evo_mat_load[:Nte, :st, :]

    Train_B_idx = np.array(random.sample(range(Ns_load), Ns))

    Train_B = Train_B_load[Train_B_idx, :]
    Test_B = Test_B_load[:Nte, :]

    test_idx = 2
    f_test = all_test_f[test_idx, :][None, :]
    f_test_evo_mat = all_test_evo_mat[test_idx, :st, :]

    #id_all = np.array(random.sample(range(Ns * Nx), N))
    id_all = np.linspace(0, N * Nx - 1, N * Nx).astype(int)
    id_test = np.linspace(0, Nte * Nx - 1, Nte * Nx).astype(int)


    # plt.plot(x, f_test.T, 'r*')
    #
    # for i in range(st):
    #     #plt.plot(x, f_test_evo_mat[i, :])
    #     plt.plot(x, all_test_evo_mat[test_idx, i, :],'b')
    #
    # plt.show()


    filename = 'C_ch_dt_1d' + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(
        d2) + '_Nx_' + num2str_deciaml(
        Nx) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(
        Nte) + '_N_' + num2str_deciaml(N) + '_p_' + num2str_deciaml(p) + '_q_' + num2str_deciaml(q) + '_l_' + num2str_deciaml(l) + '_adstep_' + num2str_deciaml(ads)

    file_name = filename + '.txt'
    npy_name = filename + '.npy'
    fig_name = filename

    dtype = tf.float32
    num_ad_epochs = ads
    num_bfgs_epochs = 2000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3/8,
        decay_steps=5000,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    nl = 5
    nr = 100
    bs = 100
    ins = 1

    W_f = np.ones((q, 1))

    wp = 1

    train_evo_step = 0

    mdl = CH(Nx, Ns, Nte, N, d1, d2, lx, dt, x, dx, Train_B, Test_B, f_test, f_test_evo_mat, all_test_f, all_test_evo_mat, dtype, optimizer, num_ad_epochs,
              num_bfgs_epochs, fig_name, file_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st, id_all, id_test, wp)
    mdl.fit()

    model_name = filename
    mdl.save('mdls/' + model_name)

