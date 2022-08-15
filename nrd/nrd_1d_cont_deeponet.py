################################ TLPIDON nrd ct #################################
# Physics-informed DeepONet with Transfer learning for nonlinear reaction
# diffusion equation in continuous time setting.
# consider (f^{n+1} - f^n) / \Delta t = d \partial_xx f^{n+1} + k (f^{n+1})^2
# with zero BC
# Date: 07/27/2022
########################################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.optimize import fsolve
import random
import scipy


class TPIDON():
    # initialize
    def __init__(self, Nt, Nx, Ns, Nte, N, k, d, lx, t, dt, x, dx, t_train, x_train, Train_B, Test_B, f_test,
                 f_test_evo_mat, all_test_f, all_test_evo_mat, dtype, optimizer, num_ad_epochs,
                 num_bfgs_epochs, file_name, fig_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, w_bc, st,
                 cl, id_all):

        self.dtype = dtype

        self.nl, self.nr = nl, nr

        self.k, self.d = k, d

        self.st = st
        self.cl = cl

        self.Nt, self.Nx, self.Ns, self.Nte, self.N = Nt, Nx, Ns, Nte, N
        self.lx = lx

        self.xl, self.xr = -lx, lx

        self.dt = dt

        self.f_test_real = np.ones_like(x)

        self.x, self.dx = tf.convert_to_tensor(x, dtype=self.dtype), dx
        self.t, self.dt = tf.convert_to_tensor(t, dtype=self.dtype), dt

        self.t_train, self.x_train = tf.convert_to_tensor(t_train, dtype=self.dtype), tf.convert_to_tensor(x_train,
                                                                                                           dtype=self.dtype)

        self.id_all = id_all
        self.id_test = np.linspace(0, Nte * Nt * Nx - 1, Nte * Nt * Nx).astype(int)

        self.Train_B = tf.convert_to_tensor(Train_B, dtype=self.dtype)
        self.Test_B = tf.convert_to_tensor(Test_B, dtype=self.dtype)

        self.eq = np.ones((1, self.Nx))

        self.f_test, self.f_test_evo_mat, self.all_test_f, self.all_test_evo_mat = f_test, f_test_evo_mat, all_test_f, all_test_evo_mat

        self.file_name = file_name
        self.fig_name = fig_name
        self.npy_name = npy_name

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.H_input_size = self.Nx
        self.T_input_size = 2

        self.p = p
        self.q = q

        self.W_f = W_f

        self.B_output_size = self.p
        self.T_output_size = self.p

        self.epoch_vec = []
        self.emp_loss_vec = []
        self.test_loss_vec = []

        # define parameter for sgd
        self.stop = 0.00000001
        self.batch_size = bs
        self.inner_step = ins
        self.train_evo_step = train_evo_step

        self.w_bc = w_bc

        #### coarser grid
        # random idex
        self.id_c = np.array(random.sample(range(Nx * Nt), 400))

        # ## every other grid
        # self.id_c = np.zeros(int(Nx / 2), )
        # for i in range(int(Nx / 2)):
        #     self.id_c[i] = int(2 * i)
        #
        # self.id_c = self.id_c.astype(int)

        # Initialize NN
        self.Bnn = self.get_B()
        self.Hnn = self.get_H()
        self.Tnn = self.get_T()

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

        UT = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Trunk)

        VT = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal')(
            input_Trunk)

        input_Trunk_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                kernel_initializer='glorot_normal')(
            input_Trunk)

        for i in range(self.nl - 1):
            B1 = tf.keras.layers.Multiply()([input_Trunk_mid, UT])

            N = tf.keras.layers.Subtract()([tf.ones_like(input_Trunk_mid), input_Trunk_mid])
            B2 = tf.keras.layers.Multiply()([N, VT])

            input_Trunk_mid = tf.keras.layers.Add()([B1, B2])

        output_Trunk = tf.keras.layers.Dense(units=self.T_output_size, activation=tf.nn.tanh,
                                             kernel_initializer='glorot_normal')(
            input_Trunk_mid)

        def my_bc(ip):
            x = ip[:, 1]
            return (tf.ones_like(x) - x) * x

        zero_bc = tf.keras.layers.Lambda(my_bc, name='zero_bc')(input_Trunk)

        out = tf.keras.layers.Multiply()([output_Trunk, zero_bc])

        model = tf.keras.Model(inputs=input_Trunk, outputs=out)

        return model

    def get_loss(self, tmp_B, tmp_t, tmp_x, tmp_IC):
        bs = tmp_B.shape[0]
        # by gradient
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_t)
            tape.watch(tmp_x)

            Train_T = tf.concat([tmp_t, tmp_x], axis=1)
            T = self.Tnn(Train_T)

            B = self.Bnn(tmp_B)[:, :, 0]
            # print(B.shape, T.shape)
            # zxc

            f = tf.reduce_sum(B * T, 1, keepdims=True)

            f_t = tape.gradient(f, tmp_t)
            f_x = tape.gradient(f, tmp_x)

        f_xx = tape.gradient(f_x, tmp_x)

        pde = f_t - self.d * f_xx - self.k * tf.square(f)

        loss_pde = tf.reduce_sum(tf.square(pde))

        T0 = self.Tnn(tf.concat([tf.zeros([bs, 1]), tmp_x], axis=1))

        ic = tf.reduce_sum(B * T0, 1, keepdims=True)

        loss_ic = tf.reduce_mean(tf.square(ic - tmp_IC))

        loss = loss_pde + loss_ic

        return loss

    def get_total_loss(self, id_all, N, Nl, B):

        # Nl is the small batch size for total loss

        part_num = N // Nl

        total_loss = 0

        for i in range(part_num):
            id_bs = id_all[i * Nl:(i + 1) * Nl]

            id_k_vec, id_n_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nt, self.Nx)

            tmp_B = tf.convert_to_tensor(B[id_k_vec, :], dtype=self.dtype)

            tmp_t = tf.convert_to_tensor(t[id_n_vec, :], dtype=self.dtype)

            tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

            tmp_IC = tf.convert_to_tensor(B[id_k_vec, id_i_vec],
                                          dtype=self.dtype)


            tmp_loss = self.get_loss(tmp_B, tmp_t, tmp_x, tmp_IC[:, None])

            # print(tmp_loss)

            total_loss = total_loss + tmp_loss * Nl

        return total_loss / N

    def get_grad(self, tmp_B, tmp_t, tmp_x, tmp_IC):
        with tf.GradientTape() as tape:
            loss = self.get_loss(tmp_B, tmp_t, tmp_x, tmp_IC)
            trainable_weights_B = self.Bnn.trainable_variables
            trainable_weights_T = self.Tnn.trainable_variables

            trainable_weights = trainable_weights_B + trainable_weights_T

        grad = tape.gradient(loss, trainable_weights)

        return loss, trainable_weights, grad

    def id_2_id_kni(self, id, Nt, Nx):
        id_k = id // (Nt * Nx)
        id_n = (id - id_k * Nt * Nx) // Nx
        id_i = id - id_k * Nt * Nx - id_n * Nx

        return id_k, id_n, id_i

    def get_id_bs_vec(self, id, Nt, Nx):
        bs = id.shape[0]

        id_k_vec = np.zeros_like(id)
        id_n_vec = np.zeros_like(id)
        id_i_vec = np.zeros_like(id)

        for j in range(bs):
            id_k_vec[j], id_n_vec[j], id_i_vec[j] = self.id_2_id_kni(id[j], Nt, Nx)

        return id_k_vec, id_n_vec, id_i_vec

    def get_random_sample(self):

        id_bs = np.random.choice(id_all, self.batch_size)

        id_k_vec, id_n_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nt, self.Nx)

        tmp_B = tf.convert_to_tensor(Train_B[id_k_vec, :], dtype=self.dtype)
        tmp_t = tf.convert_to_tensor(t[id_n_vec, :], dtype=self.dtype)
        tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)
        tmp_IC = tf.convert_to_tensor(Train_B[id_k_vec, id_i_vec], dtype=self.dtype)

        return tmp_B, tmp_t, tmp_x, tmp_IC[:, None]

    def get_record(self, epoch, total_loss, test_loss, elapsed):
        with open(self.file_name, 'a') as fw:
            print('Nx: %d, N: %d, Nte: %d, k: %.3f, d: %.3f, st: %d, p: %d, q: %d' % (
                self.Nx, self.N, self.Nte, self.k, self.d, self.st, self.p, self.q), file=fw)

            print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f' %
                  (epoch, total_loss, test_loss, elapsed), file=fw)

    def fit(self):
        start_time = time.time()
        Initial_time = time.time()
        for epoch in range(self.num_ad_epochs):
            tmp_B, tmp_t, tmp_x, tmp_IC = self.get_random_sample()

            for i in range(self.inner_step):
                loss, trainable_weights, grad = self.get_grad(tmp_B, tmp_t, tmp_x, tmp_IC)
                self.optimizer.apply_gradients(zip(grad, trainable_weights))

            if epoch % 5000 == 0:
                total_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B)
                test_loss = self.get_total_loss(self.id_test, self.Nte * self.Nt * self.Nx, 1000, Test_B)
                elapsed = time.time() - start_time
                total_elapsed = time.time() - Initial_time
                start_time = time.time()
                print('Nx: %d, N: %d, Nte: %d, dt: %.3f, k: %.3f, d: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.N, self.Nte, self.dt, self.k, self.d, self.st, self.p, self.q))
                print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                      (epoch, total_loss, test_loss, elapsed, total_elapsed))
                with open(self.file_name, 'a') as fw:
                    print('Nx: %d, N: %d, Nte: %d, dt: %.3f, k: %.3f, d: %.3f, st: %d, p: %d, q: %d' % (
                        self.Nx, self.N, self.Nte, self.dt, self.k, self.d, self.st, self.p, self.q), file=fw)
                    print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                          (epoch, total_loss, test_loss, elapsed, total_elapsed), file=fw)

                self.epoch_vec.append(epoch)

                self.emp_loss_vec.append(total_loss)
                self.test_loss_vec.append(test_loss)

                self.get_record(epoch, total_loss, test_loss, elapsed)

                # assign weight to H
                weights_B = self.Bnn.get_weights()
                # print(len(weights_B), len(self.Hnn.get_weights()), len(weights_B[:-1]))
                # zxc
                weights_H = weights_B[:-1]
                self.Hnn.set_weights(weights_H)

                self.W_f = weights_B[-1][:, :, 0]

                if total_loss < self.stop:
                    print('training finished')
                    break

            #if epoch % (self.num_ad_epochs - 1) == 0 and epoch > 0:
            if epoch % 50000 == 0 and epoch > 0:
                self.check_with_visual()

        final_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B)
        print('Final loss is %.3e' % final_loss)

    def get_evo_pred(self, f, tmp_cl):

        tmp_ic = f

        f_evo_pred = np.zeros((tmp_cl, self.Nt * self.Nx))

        Train_T = tf.concat([self.t_train, self.x_train], axis=1)

        T = self.Tnn(Train_T)

        for i in range(tmp_cl):
            B = self.Bnn(tmp_ic)[:, :, 0]

            f_tmp_pred = tf.reduce_sum(B * T, 1, keepdims=True).numpy()

            f_evo_pred[i, :] = f_tmp_pred.T

            tmp_ic = f_tmp_pred.reshape(self.Nt, self.Nx)[-1, :][None, :]

        return f_evo_pred

    def get_Tevo_mat(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t_train)
            tape.watch(self.x_train)

            Train_T = tf.concat([self.t_train, self.x_train], axis=1)

            T = self.Tnn(Train_T)

            T_xx_list = []
            T_t_list = []

            for i in range(self.p):
                T_tmp = T[:, i][:, None]
                T_x_tmp = tape.gradient(T_tmp, self.x_train)
                T_xx_tmp = tape.gradient(T_x_tmp, self.x_train)

                T_xx_list.append(T_xx_tmp)

                T_t_tmp = tape.gradient(T_tmp, self.t_train)
                T_t_list.append(T_t_tmp)

        T_xx = tf.concat(T_xx_list, axis=1)
        T_t = tf.concat(T_t_list, axis=1)

        return T, T_t, T_xx

    def loss_fun_vec(self, W, H, T, T_t, T_xx, Tl, Tr, T_ic, f_ic):
        W = W[:, None]

        K1 = np.matmul(T_t, H) - self.d * np.matmul(T_xx, H)

        K_ic = np.matmul(T_ic, H)

        pde = np.matmul(K1, W) - self.k * np.square(np.matmul(np.matmul(T, H), W))

        ic = np.matmul(K_ic, W) - f_ic

        F = np.concatenate([pde, ic], axis=0)[:, 0]

        return F.tolist()

    def get_Tevo_pred_c(self, T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic, f, tmp_cl):

        f_evo_pred = np.zeros((tmp_cl, self.Nt * self.Nx))

        tmp_ic = f

        W_tmp = self.W_f

        for i in range(tmp_cl):
            H = self.Hnn(tmp_ic)[0, :, :].numpy()

            res = scipy.optimize.leastsq(self.loss_fun_vec, W_tmp, args=(H, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic, tmp_ic.T),
                                         ftol=1e-5,
                                         xtol=1e-5)

            W_tmp = res[0][:, None]

            f_evo_pred_tmp = np.matmul(np.matmul(T, H), W_tmp)

            f_evo_pred[[i], :] = f_evo_pred_tmp.T

            tmp_ic = f_evo_pred_tmp.reshape(self.Nt, self.Nx)[-1, :][None, :]

        return f_evo_pred

    def get_l2_error_f_evo(self, f1, f2, tmp_cl):
        error = np.zeros((1, tmp_cl * self.st))
        f1_mat = f1.reshape(tmp_cl, self.Nt, self.Nx)
        f2_mat = f2.reshape(tmp_cl, self.Nt, self.Nx)
        for i in range(tmp_cl):
            for j in range(self.st):
                error[0, i * self.st + j] = np.sum(np.square(f1_mat[i, j, :] - f2_mat[i, j, :])) * self.dx
        return np.sqrt(error)

    def get_l2_re_error_f_evo(self, f1, f2, tmp_cl):
        error = np.zeros((1, tmp_cl * self.st))
        f1_mat = f1.reshape(tmp_cl, self.Nt, self.Nx)
        f2_mat = f2.reshape(tmp_cl, self.Nt, self.Nx)
        for i in range(tmp_cl):
            for j in range(self.st):
                error[0, i * self.st + j] = np.sum(np.square(f1_mat[i, j, :] - f2_mat[i, j, :])) / np.sum(
                    np.square(f2_mat[i, j, :]))
        return np.sqrt(error)

    def all_test_result(self, tmp_cl, tmp_st):

        error_mat, re_error_mat = np.zeros((self.Nte, tmp_cl * self.st)), np.zeros((self.Nte, tmp_cl * self.st))

        f_pred_tensor = np.zeros((self.Nte, tmp_cl, self.Nt * self.Nx))

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_f = self.all_test_f[[i], :]

            f_tmp_pred = self.get_evo_pred(tmp_f, tmp_cl)

            f_tmp_ref = self.all_test_evo_mat[i, :tmp_cl, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            # print(f_tmp_ref.shape, f_tmp_pred.shape)
            # zxc

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref, tmp_cl)

            re_error_mat[i, :] = self.get_l2_re_error_f_evo(f_tmp_pred, f_tmp_ref, tmp_cl)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(
            np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(np.square(self.all_test_evo_mat)))

        re_error_T_sm = np.sqrt(
            np.sum(np.square(f_pred_tensor.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st,
                             :] - self.all_test_evo_mat.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st,
                                  :])) / np.sum(
                np.square(self.all_test_evo_mat.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st, :])))

        return error_mat, re_error_mat, re_error_T, re_error_T_sm, f_pred_tensor, pred_time


    def get_all_T_c(self):
        T, T_t, T_xx = self.get_Tevo_mat()

        T, T_t, T_xx = T.numpy(), T_t.numpy(), T_xx.numpy()

        Train_l = tf.concat([self.t, -self.lx * tf.ones([self.Nt, 1])], axis=1)
        Train_r = tf.concat([self.t, self.lx * tf.ones([self.Nt, 1])], axis=1)
        Train_0 = tf.concat([tf.zeros_like(self.x), self.x], axis=1)

        Tl, Tr = self.Tnn(Train_l), self.Tnn(Train_r)

        Tl, Tr = Tl.numpy(), Tr.numpy()

        T_c, T_t_c, T_xx_c = T[self.id_c, :], T_t[self.id_c, :], T_xx[self.id_c, :]

        T_ic = self.Tnn(Train_0).numpy()

        return T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic

    def all_test_tr_result(self, tmp_cl, tmp_st):

        error_mat, re_error_mat = np.zeros((self.Nte, tmp_cl * self.st)), np.zeros((self.Nte, tmp_cl * self.st))

        f_pred_tensor = np.zeros((self.Nte, tmp_cl, self.Nt * self.Nx))

        T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic = self.get_all_T_c()

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_f = self.all_test_f[[i], :]

            f_tmp_pred = self.get_Tevo_pred_c(T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic, tmp_f, tmp_cl)

            f_tmp_ref = self.all_test_evo_mat[i, :tmp_cl, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            # print(f_tmp_ref.shape, f_tmp_pred.shape)
            # zxc

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref, tmp_cl)

            re_error_mat[i, :] = self.get_l2_re_error_f_evo(f_tmp_pred, f_tmp_ref, tmp_cl)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(
            np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(np.square(self.all_test_evo_mat)))

        re_error_T_sm = np.sqrt(
            np.sum(np.square(f_pred_tensor.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st, :] - self.all_test_evo_mat.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st, :] )) / np.sum(np.square(self.all_test_evo_mat.reshape(self.Nte, tmp_cl, self.Nt, self.Nx)[:, 0, :tmp_st, :] )))

        return error_mat, re_error_mat, re_error_T, re_error_T_sm, f_pred_tensor, pred_time


    def check_with_visual(self):

        ###### by PIDON
        f_evo_mat = self.get_evo_pred(self.f_test, self.cl)

        abs_error_in_time_mat = np.abs(self.f_test_evo_mat - f_evo_mat.reshape(self.cl * self.st, self.Nx))

        test_evo_error, test_evo_re_error, test_evo_re_error_T, test_evo_re_error_T_sm, f_pred_tensor, pred_time = self.all_test_result(self.cl, 4)

        test_evo_error_avg = np.sum(test_evo_error, axis=0) / self.Nte
        test_evo_re_error_avg = np.sum(test_evo_re_error, axis=0) / self.Nte

        ###### by TPIDON
        T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic = self.get_all_T_c()

        f_tr_evo_mat = self.get_Tevo_pred_c(T, T_c, T_t_c, T_xx_c, Tl, Tr, T_ic, self.f_test, self.cl)
        abs_tr_error_in_time_mat = np.abs(self.f_test_evo_mat - f_tr_evo_mat.reshape(self.cl * self.st, self.Nx))

        test_tr_evo_error, test_tr_evo_re_error, test_tr_evo_re_error_T, test_tr_evo_re_error_T_sm, f_tr_pred_tensor, pred_tr_time = self.all_test_tr_result(
            self.cl, 4)

        test_evo_error_avg, test_tr_evo_error_avg = np.sum(test_evo_error, axis=0) / self.Nte, np.sum(test_tr_evo_error,
                                                                                                      axis=0) / self.Nte
        test_evo_re_error_avg, test_tr_evo_re_error_avg = np.sum(test_evo_re_error, axis=0) / self.Nte, np.sum(
            test_tr_evo_re_error, axis=0) / self.Nte

        clst_vec = np.linspace(1, self.st * self.cl, self.st * self.cl).T
        clst_vec = clst_vec[:, None]

        t_c, x_c = np.meshgrid(clst_vec, x)

        f_evo_tensor = f_evo_mat.reshape(self.cl, self.Nt, self.Nx)
        f_tr_evo_tensor = f_tr_evo_mat.reshape(self.cl, self.Nt, self.Nx)

        fig1 = plt.figure(1)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        fig1.savefig('figs/' + self.fig_name + '_loss_vs_iter.eps', format='eps')

        fig7 = plt.figure(7)
        plt.plot(self.x, self.f_test.T, 'r', label='IC')
        plt.plot(self.x, f_evo_tensor[0, 0, :], 'b-o', label='PIDON IC pred')
        plt.plot(self.x, f_tr_evo_tensor[0, 0, :], 'y-*', label='TPIDON IC pred')
        plt.plot(self.x, self.f_test_evo_mat[-1, :][:, None], 'c', label='ref T')
        plt.plot(self.x, f_evo_tensor[-1, -1, :], 'm-p', label='PIDON pred T')
        plt.plot(self.x, f_tr_evo_tensor[-1, -1, :], 'g-^', label='TPIDON pred T')
        plt.legend()
        plt.title('f test sample T')
        fig7.savefig('figs/' + self.fig_name + '_f_compare.eps', format='eps')

        fig2 = plt.figure(2)
        ax = fig2.add_subplot()
        cp = ax.contourf(t_c, x_c, self.f_test_evo_mat.T)
        fig2.colorbar(cp)
        plt.title('sample reference long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)

        fig3 = plt.figure(3)
        ax = fig3.add_subplot()
        cp = ax.contourf(t_c, x_c, abs_error_in_time_mat.T)
        fig3.colorbar(cp)
        plt.title('PIDON abs error test sample long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)

        fig10 = plt.figure(10)
        ax = fig10.add_subplot()
        cp = ax.contourf(t_c, x_c, abs_tr_error_in_time_mat.T)
        fig10.colorbar(cp)
        plt.title('TPIDON abs error test sample long time')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)

        fig4 = plt.figure(4)
        for i in range(self.Nte):
            plt.semilogy(clst_vec, test_evo_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(clst_vec, test_evo_error_avg, 'r', linewidth=2)
        plt.title('PIDON long time evo error with N=' + str(self.N) + ' q=' + str(self.q))
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        fig4.savefig('figs/' + self.fig_name + '_PIDON_error.eps', format='eps')

        fig5 = plt.figure(5)
        for i in range(self.Nte):
            plt.semilogy(clst_vec, test_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(clst_vec, test_evo_re_error_avg, 'r', linewidth=2)
        plt.title('PIDON long time evo relative error with N=' + str(self.N) + ' q=' + str(self.q))
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        fig5.savefig('figs/' + self.fig_name + '_PIDON_re_error.eps', format='eps')

        fig8 = plt.figure(8)
        for i in range(self.Nte):
            plt.semilogy(clst_vec, test_tr_evo_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(clst_vec, test_tr_evo_error_avg, 'r', linewidth=2)
        plt.title('PIDON long time evo error with N=' + str(self.N) + ' q=' + str(self.q))
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        fig8.savefig('figs/' + self.fig_name + '_TPIDON_error.eps', format='eps')

        fig9 = plt.figure(9)
        for i in range(self.Nte):
            plt.semilogy(clst_vec, test_tr_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(clst_vec, test_tr_evo_re_error_avg, 'r', linewidth=2)
        plt.title('PIDON long time evo relative error with N=' + str(self.N) + ' q=' + str(self.q))
        xticks = np.arange(0, self.st * self.cl + 1, int(self.st * self.cl / 5))
        labels = np.round(np.arange(0, (self.st * self.cl + 1) * self.dt, int(self.st * self.cl / 5) * self.dt),
                          decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        fig9.savefig('figs/' + self.fig_name + '_TPIDON_re_error.eps', format='eps')

        plt.show()

        with open(self.npy_name, 'wb') as ss:
            np.save(ss, test_evo_error)
            np.save(ss, test_evo_error_avg)

            np.save(ss, test_evo_re_error)
            np.save(ss, test_evo_re_error_avg)

            np.save(ss, self.all_test_evo_mat)
            np.save(ss, f_pred_tensor)

        print('Avg PIDON test l2 relative error to 1: %.3e, Avg PIDON test l2 relative error to end: %.3e' %
              (test_evo_re_error_T_sm, test_evo_re_error_T))
        print('Avg TPIDON test l2 error to 1: %.3e, Avg TPIDON test l2 relative error to end: %.3e' %
              (test_tr_evo_re_error_T_sm, test_tr_evo_re_error_T))
        print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time))

        with open(self.file_name, 'a') as fw:
            print(
                'Avg long time PIDON test l2 error to 1: %.3e, Avg long time PIDON test l2 relative error to end: %.3e' %
                (test_evo_re_error_T_sm, test_evo_re_error_T), file=fw)
            print(
                'Avg long time TPIDON test l2 error to 1: %.3e, Avg long time TPIDON test l2 relative error to end: %.3e' %
                (test_tr_evo_re_error_T_sm, test_tr_evo_re_error_T), file=fw)
            print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time), file=fw)

    def save(self, model_name):
        B_model_name = model_name + '_B.h5'
        T_model_name = model_name + '_T.h5'
        self.Bnn.save(B_model_name)
        self.Tnn.save(T_model_name)


if __name__ == "__main__":

    # consider (f^{n+1} - f^n) / \Delta t = d \partial_xx f^{n+1} + k f^{n+1}
    # with zero BC
    #
    ##################### prepare/load training set ##############################################################

    # input parameter

    N = int(sys.argv[1])
    Nte = int(sys.argv[2])
    p = int(sys.argv[3])
    q = int(sys.argv[4])
    cl = int(sys.argv[5])

    d = np.float32(sys.argv[6])
    k = np.float32(sys.argv[7])
    l = np.float32(sys.argv[8])

    ads = int(sys.argv[9])

    # N = 10000
    # Nte = 30
    # p = 100
    # q = 25
    # cl = 50
    #
    # d = 0.001
    # k = 0.001
    # l = 0.2
    #
    # ads = 300001

    # mesh size for t
    T = 1
    Nt = 20
    dt = T / Nt
    points_t = np.linspace(dt, T, Nt).T
    t = points_t[:, None]

    # define mesh size for x
    lx = 1
    Nx = 64
    dx = lx / (Nx + 1)

    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]

    st = Nt

    nl = 5
    nr = 100
    bs = 400
    ins = 1

    Ns_load = 2000
    Ns = Ns_load

    Nte_load = 30
    st_load = 1000


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


    filename = 'A_nrd_sample_1d_one_nevo' + '_k_' + num2str_deciaml(k) + '_d_' + num2str_deciaml(
        d) + '_Nx_' + num2str_deciaml(
        Nx) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st_load) + '_Ns_' + num2str_deciaml(
        Ns_load) + '_Nte_' + num2str_deciaml(
        Nte_load) + '_l_' + num2str_deciaml(l)
    npy_name = filename + '.npy'

    with open(npy_name, 'rb') as ss:
        _ = np.load(ss)
        Train_B = np.load(ss)
        Test_B_load = np.load(ss)

        all_test_f_load = np.load(ss)
        all_test_evo_mat_load = np.load(ss)

    all_test_f = all_test_f_load[:Nte, :]
    all_test_evo_mat = all_test_evo_mat_load[:Nte, :st * cl, :].reshape(Nte, cl, st * Nx)

    Test_B = Test_B_load[:Nte, :]

    f_test = all_test_f[0, :][None, :]
    f_test_evo_mat = all_test_evo_mat_load[0, :st * cl, :]

    id_all = np.array(random.sample(range(Ns * Nt * Nx), N))

    t_train = np.kron(t, np.ones((Nx, 1)))
    x_train = np.tile(x, (Nt, 1))


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


    dtype = tf.float32
    num_ad_epochs = ads
    num_bfgs_epochs = 2000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    W_f = np.ones((q, 1))

    w_bc = 1

    filename = 'C_TPIDON_nrd_ct_1d' + '_k_' + num2str_deciaml(k) + '_d_' + num2str_deciaml(
        d) + '_Nx_' + num2str_deciaml(
        Nx) + '_dt_' + num2str_deciaml(dt) + '_batchN_' + num2str_deciaml(N) + '_Nte_' + num2str_deciaml(
        Nte) + '_p_' + num2str_deciaml(p) + '_q_' + num2str_deciaml(q) + '_l_' + num2str_deciaml(
        l) + '_adstep_' + num2str_deciaml(num_ad_epochs)

    fig_name = filename
    file_name = filename + '.txt'
    npy_name = filename + '.npy'

    train_evo_step = 0

    mdl = TPIDON(Nt, Nx, Ns, Nte, N, k, d, lx, t, dt, x, dx, t_train, x_train, Train_B, Test_B, f_test, f_test_evo_mat,
                 all_test_f, all_test_evo_mat, dtype, optimizer, num_ad_epochs,
                 num_bfgs_epochs, file_name, fig_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, w_bc, st,
                 cl, id_all)
    mdl.fit()

    model_name = filename
    mdl.save('mdls/' + model_name)