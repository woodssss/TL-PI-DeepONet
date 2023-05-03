################################ Data generation ns   ##################################
# Physics-informed DeepONet with Transfer learning for navier stokes equation
# in discretized time setting.
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
from matplotlib import cm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import random
import pickle as pl


class NS2D():
    # initialize
    def __init__(self, Nx, Ny, N_c, Ns, Nte, N, nu, lx, dt, x, y, dx, dy, f_sc, Train_B, Train_psi_x, Train_psi_y,
                 Test_B, Test_psi_x, Test_psi_y, f_test, f_test_evo_mat,
                 all_test_f, all_test_evo_mat, dtype, optimizer, num_ad_epochs,
                 num_bfgs_epochs, fig_name, file_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, w_bc, st,
                 id_all, id_test):

        self.dtype = dtype
        self.id_all = id_all
        self.id_test = np.linspace(0, Nte * Nx * Ny - 1, Nte * Nx * Ny).astype(int)

        self.nl, self.nr = nl, nr

        self.nu = nu

        self.st = st

        self.Nx, self.Ns, self.N, self.Nte = Nx, Ns, N, Nte
        self.Ny = Ny
        self.lx = lx

        self.dt = dt

        self.f_test_real = np.ones_like(x)

        self.x, self.dx = tf.convert_to_tensor(x, dtype=self.dtype), dx
        self.y, self.dy = tf.convert_to_tensor(y, dtype=self.dtype), dy

        self.xx, self.yy = np.meshgrid(x, y)

        self.x_train = tf.convert_to_tensor(np.kron(x, np.ones((Nx, 1))), dtype=self.dtype)
        self.y_train = tf.convert_to_tensor(np.tile(y, (Nx, 1)), dtype=self.dtype)

        self.Train_T = tf.concat([self.x_train, self.y_train], axis=1)

        self.Train_B = tf.convert_to_tensor(Train_B, dtype=self.dtype)
        self.Train_psi_x = tf.convert_to_tensor(Train_psi_x, dtype=self.dtype)
        self.Train_psi_y = tf.convert_to_tensor(Train_psi_y, dtype=self.dtype)

        self.Test_B = tf.convert_to_tensor(Test_B, dtype=self.dtype)
        self.Test_psi_x = tf.convert_to_tensor(Test_psi_x, dtype=self.dtype)
        self.Test_psi_y = tf.convert_to_tensor(Test_psi_y, dtype=self.dtype)

        self.Train_w_tensor = Train_B.reshape(Ns, Nx, Ny)
        self.Train_psi_x_tensor = Train_psi_x.reshape(Ns, Nx, Ny)
        self.Train_psi_y_tensor = Train_psi_y.reshape(Ns, Nx, Ny)

        self.Test_w_tensor = Test_B.reshape(Nte, Nx, Ny)
        self.Test_psi_x_tensor = Test_psi_x.reshape(Nte, Nx, Ny)
        self.Test_psi_y_tensor = Test_psi_y.reshape(Nte, Nx, Ny)

        self.eq = np.ones((1, self.Nx))

        self.f_test, self.f_test_evo_mat = f_test, f_test_evo_mat
        self.all_test_f, self.all_test_evo_mat = all_test_f, all_test_evo_mat

        self.file_name = file_name
        self.fig_name = fig_name
        self.npy_name = npy_name

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.H_input_size = self.Nx ** 2
        self.T_input_size = 2

        self.p = p
        self.q = q

        self.W_f = W_f

        self.B_output_size = self.p
        self.T_output_size = self.p

        self.kernel_size = 12

        self.epoch_vec = []
        self.emp_loss_vec = []
        self.test_loss_vec = []

        # define parameter for sgd
        self.stop = 0.0000001
        self.batch_size = bs
        self.inner_step = ins
        self.train_evo_step = train_evo_step

        self.w_bc = w_bc

        ### define coarser grid
        # N_c = int(self.Nx / 2)
        # N_c = int(self.Nx/2)

        self.id_c = np.array(random.sample(range(Nx ** 2), N_c ** 2))

        self.f_sc = f_sc
        self.f_sc_c = f_sc[self.id_c, :]

        # Initialize NN
        self.Bnn = self.get_B()
        self.Hnn = self.get_H()
        self.T_wvnn = self.get_T_wv()
        self.T_psinn = self.get_T_psi()
        # self.Bnn.summary()
        # self.Hnn.summary()
        # self.T_wvnn.summary()
        # self.T_psinn.summary()

    def get_B(self):
        # define nn for Branch
        # use CNN
        input_Branch_b = tf.keras.Input(shape=(self.H_input_size,))

        # reshape to image
        img = tf.keras.layers.Reshape((self.Nx, self.Nx, 1))(input_Branch_b)

        img = tf.keras.layers.Conv2D(1, self.kernel_size, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            img)

        # img = tf.keras.layers.Conv2D(1, self.kernel_size, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(img)
        # print(img.shape)

        input_Branch = tf.keras.layers.Flatten()(img)

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

        model = tf.keras.Model(inputs=input_Branch_b, outputs=out)

        return model

    def get_H(self):
        # define nn for Branch
        input_Branch_h = tf.keras.Input(shape=(self.H_input_size,))

        # reshape to image
        img = tf.keras.layers.Reshape((self.Nx, self.Nx, 1))(input_Branch_h)

        img = tf.keras.layers.Conv2D(1, self.kernel_size, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            img)

        # img = tf.keras.layers.Conv2D(1, self.kernel_size, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(img)

        input_Branch = tf.keras.layers.Flatten()(img)

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

        model = tf.keras.Model(inputs=input_Branch_h, outputs=out)

        return model

    def get_T_wv(self):
        # define nn for Trunk

        input_Trunk = tf.keras.Input(shape=(self.T_input_size,))

        def pd_feature(ip):
            x = ip[:, 0]
            y = ip[:, 1]
            # basis
            b1 = tf.ones_like(x)
            b2 = tf.cos(2 * np.pi * x)
            b3 = tf.sin(2 * np.pi * x)
            b4 = tf.cos(2 * np.pi * y)
            b5 = tf.sin(2 * np.pi * y)

            # b6 = tf.cos(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b7 = tf.cos(2 * np.pi * x) * tf.sin(2 * np.pi * y)
            # b8 = tf.sin(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b9 = tf.sin(2 * np.pi * x) * tf.sin(2 * np.pi * y)

            #out = tf.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9], axis=1)
            out = tf.stack([b1, b2, b3, b4, b5], axis=1)
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

        output_Trunk = tf.keras.layers.Dense(units=self.p, activation=tf.nn.tanh,
                                             kernel_initializer='glorot_normal')(
            input_Trunk_mid)

        model = tf.keras.Model(inputs=input_Trunk, outputs=output_Trunk)
        return model

    def get_T_psi(self):
        # define nn for Trunk

        input_Trunk = tf.keras.Input(shape=(self.T_input_size,))

        def pd_feature(ip):
            x = ip[:, 0]
            y = ip[:, 1]
            # basis
            b1 = tf.ones_like(x)
            b2 = tf.cos(2 * np.pi * x)
            b3 = tf.sin(2 * np.pi * x)
            b4 = tf.cos(2 * np.pi * y)
            b5 = tf.sin(2 * np.pi * y)

            # b6 = tf.cos(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b7 = tf.cos(2 * np.pi * x) * tf.sin(2 * np.pi * y)
            # b8 = tf.sin(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b9 = tf.sin(2 * np.pi * x) * tf.sin(2 * np.pi * y)

            #out = tf.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9], axis=1)
            out = tf.stack([b1, b2, b3, b4, b5], axis=1)
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

        output_Trunk = tf.keras.layers.Dense(units=self.p, activation=tf.nn.tanh,
                                             kernel_initializer='glorot_normal')(
            input_Trunk_mid)

        model = tf.keras.Model(inputs=input_Trunk, outputs=output_Trunk)
        return model

    def get_source_f(self, x, y):
        f = 0.1 * (tf.sin(2 * np.pi * (x + y)) + tf.cos(2 * np.pi * (x + y)))
        return f

    def get_loss(self, tmp_B, tmp_x, tmp_y, tmp_w, tmp_psi_x, tmp_psi_y):

        w_old = tmp_w

        # by gradient
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_x)
            tape.watch(tmp_y)

            Train_T = tf.concat([tmp_x, tmp_y], axis=1)

            T_wv = self.T_wvnn(Train_T)
            T_psi = self.T_psinn(Train_T)

            B = self.Bnn(tmp_B)[:, :, 0]

            wv = tf.reduce_sum(B * T_wv, 1, keepdims=True)
            psi = tf.reduce_sum(B * T_psi, 1, keepdims=True)

            wv_x = tape.gradient(wv, tmp_x)
            wv_xx = tape.gradient(wv_x, tmp_x)

            wv_y = tape.gradient(wv, tmp_y)
            wv_yy = tape.gradient(wv_y, tmp_y)

            psi_x = tape.gradient(psi, tmp_x)
            psi_xx = tape.gradient(psi_x, tmp_x)

            psi_y = tape.gradient(psi, tmp_y)
            psi_yy = tape.gradient(psi_y, tmp_y)

        # poisson equation for updating psi
        pde1 = psi_xx + psi_yy + wv

        # update wv
        Lap_wv = wv_xx + wv_yy

        U_d_grad_wv = tmp_psi_y * wv_x - tmp_psi_x * wv_y

        f_sc = self.get_source_f(tmp_x, tmp_y)

        pde2 = wv + self.dt * U_d_grad_wv - self.dt * self.nu * Lap_wv - self.dt * f_sc - w_old

        # print(f_sc.shape, w_old.shape, wv.shape, Lap_wv.shape, U_d_grad_wv.shape, pde1.shape, pde2.shape)
        # zxc

        loss_pde = tf.reduce_mean(tf.square(pde1)) + tf.reduce_mean(tf.square(pde2))

        loss = loss_pde

        del tape

        return loss

    def get_total_loss(self, id_all, N, Nl, B, B_f_tensor, B_psi_x_tensor, B_psi_y_tensor):

        # Nl is the small batch size for total loss

        part_num = N // Nl

        total_loss = 0

        for i in range(part_num):
            id_bs = id_all[i * Nl:(i + 1) * Nl]

            # print(id_all.shape, id_bs1.shape, id_bs.shape)
            # zxc

            id_k_vec, id_i_vec, id_j_vec = self.get_id_bs_vec(id_bs, self.Nx, self.Nx)

            tmp_B = tf.convert_to_tensor(B[id_k_vec, :], dtype=self.dtype)

            tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

            tmp_y = tf.convert_to_tensor(y[id_j_vec, :], dtype=self.dtype)

            tmp_w = tf.convert_to_tensor(B_f_tensor[id_k_vec, id_i_vec, id_j_vec],
                                         dtype=self.dtype)

            tmp_psi_x = tf.convert_to_tensor(B_psi_x_tensor[id_k_vec, id_i_vec, id_j_vec],
                                             dtype=self.dtype)

            tmp_psi_y = tf.convert_to_tensor(B_psi_y_tensor[id_k_vec, id_i_vec, id_j_vec],
                                             dtype=self.dtype)

            tmp_loss = self.get_loss(tmp_B, tmp_x, tmp_y, tmp_w[:, None], tmp_psi_x[:, None], tmp_psi_y[:, None])

            total_loss = total_loss + tmp_loss * Nl

        return total_loss / N

    def get_grad(self, tmp_B, tmp_x, tmp_y, tmp_w, tmp_psi_x, tmp_psi_y):
        with tf.GradientTape() as tape:
            loss = self.get_loss(tmp_B, tmp_x, tmp_y, tmp_w, tmp_psi_x, tmp_psi_y)
            trainable_weights_B = self.Bnn.trainable_variables
            trainable_weights_T_wv = self.T_wvnn.trainable_variables
            trainable_weights_T_psi = self.T_psinn.trainable_variables

            trainable_weights = trainable_weights_B + trainable_weights_T_wv + trainable_weights_T_psi

        grad = tape.gradient(loss, trainable_weights)

        return loss, trainable_weights, grad

    def id_2_id_kij(self, id, Nx, Ny):
        id_k = int(id // (Nx * Ny))
        id_i = int((id - id_k * Nx * Ny) // (Ny))
        id_j = int(id - id_k * Nx * Ny - id_i * Ny)

        return id_k, id_i, id_j

    def get_id_bs_vec(self, id, Nx, Ny):
        bs = id.shape[0]

        id_k_vec = np.zeros_like(id)
        id_i_vec = np.zeros_like(id)
        id_j_vec = np.zeros_like(id)

        for j in range(bs):
            id_k_vec[j], id_i_vec[j], id_j_vec[j] = self.id_2_id_kij(id[j], Nx, Ny)

        return id_k_vec, id_i_vec, id_j_vec

    def get_random_sample(self, id_all):

        id_bs = np.random.choice(id_all, self.batch_size)

        id_k_vec, id_i_vec, id_j_vec = self.get_id_bs_vec(id_bs, self.Nx, self.Nx)

        tmp_B = tf.convert_to_tensor(Train_B[id_k_vec, :], dtype=self.dtype)
        tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)
        tmp_y = tf.convert_to_tensor(y[id_j_vec, :], dtype=self.dtype)
        tmp_w = tf.convert_to_tensor(self.Train_w_tensor[id_k_vec, id_i_vec, id_j_vec], dtype=self.dtype)
        tmp_psi_x = tf.convert_to_tensor(self.Train_psi_x_tensor[id_k_vec, id_i_vec, id_j_vec], dtype=self.dtype)
        tmp_psi_y = tf.convert_to_tensor(self.Train_psi_y_tensor[id_k_vec, id_i_vec, id_j_vec], dtype=self.dtype)

        return tmp_B, tmp_x, tmp_y, tmp_w[:, None], tmp_psi_x[:, None], tmp_psi_y[:, None]

    def get_record(self, epoch, total_loss, elapsed):
        with open(self.file_name, 'a') as fw:
            print(' Nx=', self.Nx, file=fw)
            print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                  (epoch, total_loss, elapsed), file=fw)

    def adam_step(self):
        tmp_B, tmp_x, tmp_y, tmp_w, tmp_psi_x, tmp_psi_y = self.get_random_sample(self.id_all)

        for i in range(self.inner_step):
            with tf.device('/device:GPU:0'):
                loss, trainable_weights, grad = self.get_grad(tmp_B, tmp_x, tmp_y, tmp_w, tmp_psi_x, tmp_psi_y)
                self.optimizer.apply_gradients(zip(grad, trainable_weights))

    def fit(self):
        Initial_time = time.time()
        start_time = time.time()
        #self.compare_loss(self.f_test, self.Test_psi_x[0, :][:, None], self.Test_psi_y[0, :][:, None])
        #self.check_with_visual()
        for epoch in range(self.num_ad_epochs):
            self.adam_step()

            if epoch % 20000 == 0:
                total_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B, self.Train_w_tensor,
                                                 self.Train_psi_x_tensor, self.Train_psi_y_tensor)
                test_loss = self.get_total_loss(self.id_test, self.Nte * self.Nx * self.Nx, 1000, Test_B,
                                                self.Test_w_tensor, self.Test_psi_x_tensor, self.Test_psi_y_tensor)
                elapsed = time.time() - start_time
                total_elapsed = time.time() - Initial_time
                start_time = time.time()
                print('Nx: %d, N: %d, Nte: %d, dt: %.3f, nu: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.N, self.Nte, self.dt, self.nu, self.st, self.p, self.q))
                print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                      (epoch, total_loss, test_loss, elapsed, total_elapsed))
                with open(self.file_name, 'a') as fw:
                    print('Nx: %d, N: %d, Nte: %d, dt: %.3f, nu: %.3f, st: %d, p: %d, q: %d' % (
                        self.Nx, self.N, self.Nte, self.dt, self.nu, self.st, self.p, self.q), file=fw)
                    print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                          (epoch, total_loss, test_loss, elapsed, total_elapsed), file=fw)

                self.epoch_vec.append(epoch)
                self.emp_loss_vec.append(total_loss)
                self.test_loss_vec.append(test_loss)

                self.get_record(epoch, total_loss, elapsed)

                ### assign weight to H
                weights_B = self.Bnn.get_weights()
                self.W_f = weights_B[-1][:, 0, 0][:, None]
                # print(len(weights_B), len(self.Hnn.get_weights()), len(weights_B[:-1]), self.W_f)
                # zxc
                weights_H = weights_B[:-1]
                self.Hnn.set_weights(weights_H)

                if total_loss < self.stop:
                    self.check_with_visual()
                    print('training finished')
                    break

            if epoch % (self.num_ad_epochs - 1) == 0 and epoch > 0:
            #if epoch % 40000 == 0 and epoch > 0:
                self.check_with_visual()

        final_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B, self.Train_w_tensor,
                                         self.Train_psi_x_tensor, self.Train_psi_y_tensor)
        print('Final loss is %.3e' % final_loss)

    ###################### prediction ##########################
    def get_evo_pred(self, f):

        tmp_ic = f

        f_evo_pred = np.zeros((self.st, self.Nx ** 2))

        T_wv = self.T_wvnn(self.Train_T)

        for i in range(self.st):
            B = self.Hnn(tmp_ic)[:, :, 0]

            f_tmp_pred = tf.reduce_sum(B * T_wv, 1, keepdims=True).numpy()

            f_evo_pred[i, :] = f_tmp_pred.T

            tmp_ic = f_tmp_pred.T

        f_next = f_evo_pred[0, :][:, None]

        return f_next, f_evo_pred

    def get_Tevo_mat(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_train)
            tape.watch(self.y_train)

            Train_T = tf.concat([self.x_train, self.y_train], axis=1)

            T_wv = self.T_wvnn(Train_T)
            T_psi = self.T_psinn(Train_T)

            T_wv_x_list = []
            T_wv_y_list = []
            T_wv_xx_list = []
            T_wv_yy_list = []

            T_psi_x_list = []
            T_psi_y_list = []
            T_psi_xx_list = []
            T_psi_yy_list = []

            for i in range(self.p):
                T_wv_tmp = T_wv[:, i][:, None]
                T_wv_x_tmp = tape.gradient(T_wv_tmp, self.x_train)
                T_wv_xx_tmp = tape.gradient(T_wv_x_tmp, self.x_train)
                T_wv_y_tmp = tape.gradient(T_wv_tmp, self.y_train)
                T_wv_yy_tmp = tape.gradient(T_wv_y_tmp, self.y_train)

                T_psi_tmp = T_psi[:, i][:, None]
                T_psi_x_tmp = tape.gradient(T_psi_tmp, self.x_train)
                T_psi_xx_tmp = tape.gradient(T_psi_x_tmp, self.x_train)
                T_psi_y_tmp = tape.gradient(T_psi_tmp, self.y_train)
                T_psi_yy_tmp = tape.gradient(T_psi_y_tmp, self.y_train)

                T_wv_x_list.append(T_wv_x_tmp)
                T_wv_y_list.append(T_wv_y_tmp)
                T_wv_xx_list.append(T_wv_xx_tmp)
                T_wv_yy_list.append(T_wv_yy_tmp)

                T_psi_x_list.append(T_psi_x_tmp)
                T_psi_y_list.append(T_psi_y_tmp)
                T_psi_xx_list.append(T_psi_xx_tmp)
                T_psi_yy_list.append(T_psi_yy_tmp)

            T_wv_x = tf.concat(T_wv_x_list, axis=1)
            T_wv_y = tf.concat(T_wv_y_list, axis=1)
            T_wv_xx = tf.concat(T_wv_xx_list, axis=1)
            T_wv_yy = tf.concat(T_wv_yy_list, axis=1)

            T_psi_x = tf.concat(T_psi_x_list, axis=1)
            T_psi_y = tf.concat(T_psi_y_list, axis=1)
            T_psi_xx = tf.concat(T_psi_xx_list, axis=1)
            T_psi_yy = tf.concat(T_psi_yy_list, axis=1)

        return T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy

    def get_Tevo_mat_sm(self, tmp_x, tmp_y):
        # T
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_x)
            tape.watch(tmp_y)

            Train_T = tf.concat([tmp_x, tmp_y], axis=1)

            T_wv = self.T_wvnn(Train_T)
            T_psi = self.T_psinn(Train_T)

            T_wv_x_list = []
            T_wv_y_list = []
            T_wv_xx_list = []
            T_wv_yy_list = []

            T_psi_x_list = []
            T_psi_y_list = []
            T_psi_xx_list = []
            T_psi_yy_list = []

            for i in range(self.p):
                T_wv_tmp = T_wv[:, i][:, None]

                T_wv_x_tmp = tape.gradient(T_wv_tmp, tmp_x)
                T_wv_xx_tmp = tape.gradient(T_wv_x_tmp, tmp_x)
                T_wv_y_tmp = tape.gradient(T_wv_tmp, tmp_y)
                T_wv_yy_tmp = tape.gradient(T_wv_y_tmp, tmp_y)

                T_psi_tmp = T_psi[:, i][:, None]
                T_psi_x_tmp = tape.gradient(T_psi_tmp, tmp_x)
                T_psi_xx_tmp = tape.gradient(T_psi_x_tmp, tmp_x)
                T_psi_y_tmp = tape.gradient(T_psi_tmp, tmp_y)
                T_psi_yy_tmp = tape.gradient(T_psi_y_tmp, tmp_y)

                T_wv_x_list.append(T_wv_x_tmp)
                T_wv_y_list.append(T_wv_y_tmp)
                T_wv_xx_list.append(T_wv_xx_tmp)
                T_wv_yy_list.append(T_wv_yy_tmp)

                T_psi_x_list.append(T_psi_x_tmp)
                T_psi_y_list.append(T_psi_y_tmp)
                T_psi_xx_list.append(T_psi_xx_tmp)
                T_psi_yy_list.append(T_psi_yy_tmp)

            T_wv_x = tf.concat(T_wv_x_list, axis=1)
            T_wv_y = tf.concat(T_wv_y_list, axis=1)
            T_wv_xx = tf.concat(T_wv_xx_list, axis=1)
            T_wv_yy = tf.concat(T_wv_yy_list, axis=1)

            T_psi_x = tf.concat(T_psi_x_list, axis=1)
            T_psi_y = tf.concat(T_psi_y_list, axis=1)
            T_psi_xx = tf.concat(T_psi_xx_list, axis=1)
            T_psi_yy = tf.concat(T_psi_yy_list, axis=1)

        return T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy

    def get_Tevo_mat_cat(self, Nl):
        num_part = self.Nx * self.Ny // Nl

        T_wv_list = []

        T_wv_x_list = []
        T_wv_y_list = []
        T_wv_xx_list = []
        T_wv_yy_list = []

        T_psi_x_list = []
        T_psi_y_list = []
        T_psi_xx_list = []
        T_psi_yy_list = []

        for i in range(num_part):

            tmp_x, tmp_y = self.x_train[i * Nl:(i + 1) * Nl, 0][:, None], self.y_train[i * Nl:(i + 1) * Nl, 0][:, None]

            T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy = self.get_Tevo_mat_sm(tmp_x, tmp_y)

            T_wv_list.append(T_wv.numpy())
            T_wv_x_list.append(T_wv_x.numpy())
            T_wv_y_list.append(T_wv_y.numpy())
            T_wv_xx_list.append(T_wv_xx.numpy())
            T_wv_yy_list.append(T_wv_yy.numpy())

            T_psi_x_list.append(T_psi_x.numpy())
            T_psi_y_list.append(T_psi_y.numpy())
            T_psi_xx_list.append(T_psi_xx.numpy())
            T_psi_yy_list.append(T_psi_yy.numpy())

        T_wv = np.concatenate(T_wv_list, axis=0)
        T_wv_x = np.concatenate(T_wv_x_list, axis=0)
        T_wv_y = np.concatenate(T_wv_y_list, axis=0)
        T_wv_xx = np.concatenate(T_wv_xx_list, axis=0)
        T_wv_yy = np.concatenate(T_wv_yy_list, axis=0)

        T_psi_x = np.concatenate(T_psi_x_list, axis=0)
        T_psi_y = np.concatenate(T_psi_y_list, axis=0)
        T_psi_xx = np.concatenate(T_psi_xx_list, axis=0)
        T_psi_yy = np.concatenate(T_psi_yy_list, axis=0)

        return T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy


    def get_Tevo_pred_linear(self, T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y,
                             T_psi_xx, T_psi_yy, f, psi_x, psi_y):

        f_evo_pred = np.zeros((self.st, self.Nx ** 2))

        # evo
        for i in range(self.st):
            H = self.Hnn(f)[0, :, :]

            DPxo = np.diag(psi_x[:, 0])
            DPyo = np.diag(psi_y[:, 0])

            # print(DPxo.shape, DPxo.shape)
            # zxc

            K11 = np.matmul(T_wv, H) + self.dt * np.matmul(DPyo, np.matmul(T_wv_x, H)) - self.dt * np.matmul(DPxo, np.matmul(T_wv_y,H)) - self.dt * self.nu * np.matmul(T_wv_xx, H) - self.dt * self.nu * np.matmul(T_wv_yy, H)
            K12 = np.zeros_like(K11)

            K21 = np.matmul(T_wv, H)
            K22 = np.matmul(T_psi_xx + T_psi_yy, H)

            K1 = np.concatenate([K11, K12], axis=1)
            K2 = np.concatenate([K21, K22], axis=1)

            K = np.concatenate([K1, K2], axis=0)

            # print(K1.shape, K2.shape, K.shape, f.shape, self.f_sc.shape)
            # zxc

            RHS = np.concatenate([f.T + self.dt * self.f_sc, np.zeros((self.Nx ** 2, 1))], axis=0)

            L = np.matmul(K.T, K)

            R = np.matmul(K.T, RHS)

            W_tmp = np.linalg.lstsq(L, R, 0.0000000001)[0]

            #W_tmp = np.linalg.lstsq(K, RHS, 0.0000000001)[0]

            W_tmp_wv = W_tmp[:self.q, [0]]
            W_tmp_psi = W_tmp[self.q:, [0]]

            # ee = np.mean(np.square(np.matmul(K, W_tmp) - RHS))

            # print('at ', i, ' step ee is', ee)

            f_next = np.matmul(np.matmul(T_wv, H), W_tmp_wv)

            psi_x = np.matmul(np.matmul(T_psi_x, H), W_tmp_psi)
            psi_y = np.matmul(np.matmul(T_psi_y, H), W_tmp_psi)

            f_evo_pred[[i], :] = f_next.T

            f = f_next.T

        f_next = f_evo_pred[0, :][:, None]

        return f_next, f_evo_pred

    def get_l1_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            error[0, i] = np.sqrt(np.sum(np.square(f1[i, :] - f2[i, :])) * self.dx * self.dy)
        return error

    def get_re_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            error[0, i] = np.sqrt(np.sum(np.square(f1[i, :] - f2[i, :])) / np.sum(np.square(f2[i, :])))
        return error

    def all_test_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx ** 2))

        p_start_time = time.time()

        for i in range(self.Nte):
            # print(self.all_test_f.shape, i)
            tmp_f = self.all_test_f[[i], :]

            _, f_tmp_pred = self.get_evo_pred(tmp_f)

            f_tmp_ref = self.all_test_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l1_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(
            np.square(self.all_test_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time

    def all_test_tr_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx ** 2))

        T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy = self.get_Tevo_mat_cat(self.Nx)

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_f = self.all_test_f[[i], :]
            tmp_psi_x = self.Test_psi_x[i, :][:, None]
            tmp_psi_y = self.Test_psi_y[i, :][:, None]

            _, f_tmp_pred = self.get_Tevo_pred_linear(T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y,
                                                      T_psi_xx, T_psi_yy, tmp_f, tmp_psi_x, tmp_psi_y)

            f_tmp_ref = self.all_test_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l1_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_evo_mat)) / np.sum(
            np.square(self.all_test_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time

    def check_with_visual(self):

        self.save(self.file_name)

        ### assign weight to H
        weights_B = self.Bnn.get_weights()
        self.W_f = weights_B[-1][:, 0, 0][:, None]
        # print(len(weights_B), len(self.Hnn.get_weights()), len(weights_B[:-1]), self.W_f)
        # zxc
        weights_H = weights_B[:-1]
        self.Hnn.set_weights(weights_H)

        T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x, T_psi_y, T_psi_xx, T_psi_yy = self.get_Tevo_mat_cat(self.Nx)

        ###### by PIDON
        f_0, f_evo_mat = self.get_evo_pred(self.f_test)

        test_evo_error, test_evo_re_error, test_evo_re_error_T, f_pred_tensor, pred_time = self.all_test_result()

        ###### by TPIDON
        f_tr_0, f_tr_evo_mat = self.get_Tevo_pred_linear(T_wv, T_wv_x, T_wv_y, T_wv_xx, T_wv_yy, T_psi_x,
                                                         T_psi_y, T_psi_xx, T_psi_yy, self.f_test,
                                                         self.Test_psi_x[0, :][:, None], self.Test_psi_y[0, :][:, None])

        ###############################################
        plt.rcParams.update({'font.size': 16})
        t_id = 0

        fig = plt.figure(2)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=0.05, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_tz.eps', format='eps')

        fig = plt.figure(3)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=0.05, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_tz.eps', format='eps')

        fig = plt.figure(4)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=0.05, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_tz.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 50

        fig = plt.figure(21)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_tzp5.eps', format='eps')

        fig = plt.figure(31)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_tzp5.eps', format='eps')

        fig = plt.figure(41)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_tzp5.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 100

        fig = plt.figure(25)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t1.eps', format='eps')

        fig = plt.figure(35)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t1.eps', format='eps')

        fig = plt.figure(45)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t1.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 200

        fig = plt.figure(22)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=2, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t2.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t2.eps', format='eps')

        fig = plt.figure(32)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=2, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t2.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t2.eps', format='eps')

        fig = plt.figure(42)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=2, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t2.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t2.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 399

        fig = plt.figure(23)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=4, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t4.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t4.eps', format='eps')

        fig = plt.figure(33)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=4, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t4.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t4.eps', format='eps')

        fig = plt.figure(43)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=4, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t4.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t4.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 999

        fig = plt.figure(24)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.f_test_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $w(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t_end.eps', format='eps')

        fig = plt.figure(34)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $w(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t_end.eps', format='eps')

        fig = plt.figure(44)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100,cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $w(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t_end.eps', format='eps')

        ################################################################

        plt.show()
        plt.close('all')


        test_tr_evo_error, test_tr_evo_re_error, test_tr_evo_re_error_T, f_tr_pred_tensor, pred_tr_time = self.all_test_tr_result()

        test_evo_error_avg, test_tr_evo_error_avg = np.sum(test_evo_error, axis=0) / self.Nte, np.sum(test_tr_evo_error,
                                                                                                      axis=0) / self.Nte
        test_evo_re_error_avg, test_tr_evo_re_error_avg = np.sum(test_evo_re_error, axis=0) / self.Nte, np.sum(
            test_tr_evo_re_error, axis=0) / self.Nte

        st_vec = np.linspace(1, self.st, self.st).T
        st_vec = st_vec[:, None]

        plt.rcParams.update({'font.size': 12})
        fig1 = plt.figure(1)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        plt.title('loss vs iteration')
        fig1.savefig('figs/' + self.fig_name + '_loss_vs_iter.eps', format='eps')


        fig8 = plt.figure(8)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2))
        fig8.savefig('figs/' + self.fig_name + '_PIDON_error.eps', format='eps')

        fig9 = plt.figure(9)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$Relative L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2))
        fig9.savefig('figs/' + self.fig_name + '_PIDON_re_error.eps', format='eps')

        fig10 = plt.figure(10)
        for i in range(self.Nte):
            plt.plot(st_vec, test_tr_evo_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_tr_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2))
        fig10.savefig('figs/' + self.fig_name + '_TPIDON_error.eps', format='eps')

        fig11 = plt.figure(11)
        for i in range(self.Nte):
            plt.plot(st_vec, test_tr_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_tr_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'Relative $L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2))
        fig11.savefig('figs/' + self.fig_name + '_TPIDON_re_error.eps', format='eps')

        plt.show()
        plt.close('all')

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

        print('Avg PIDON test l2 error: %.3e, Avg PIDON test l2 relative error: %.3e' %
              (1 / self.Nte * np.sum(test_evo_error) * self.dt, test_evo_re_error_T))
        print('Avg TPIDON test l2 error: %.3e, Avg TPIDON test l2 relative error: %.3e' %
              (1 / self.Nte * np.sum(test_tr_evo_error) * self.dt, test_tr_evo_re_error_T))
        print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time))

        with open(self.file_name, 'a') as fw:
            print('Avg long time PIDON test l2 error: %.3e, Avg long time PIDON test l2 relative error: %.3e' %
                  (1 / self.Nte * np.sum(test_evo_error) * self.dt, test_evo_re_error_T), file=fw)
            print(
                'Avg long time TPIDON test l2 error: %.3e, Avg long time TPIDON test l2 relative error: %.3e' %
                (1 / self.Nte * np.sum(test_tr_evo_error) * self.dt, test_tr_evo_re_error_T), file=fw)
            print('PIDON prediction time: %.3f, TPIDON prediction time: %.3f' % (pred_time, pred_tr_time), file=fw)

    def save(self, model_name):
        B_model_name = model_name + '_B.h5'
        T_wv_model_name = model_name + '_T_wv.h5'
        T_psi_model_name = model_name + '_T_psi.h5'
        self.Hnn.save(B_model_name)
        self.T_wvnn.save(T_wv_model_name)
        self.T_psinn.save(T_psi_model_name)


if __name__ == "__main__":

    N = int(sys.argv[1])
    Nte = int(sys.argv[2])
    p = int(sys.argv[3])
    q = int(sys.argv[4])
    st = int(sys.argv[5])

    nu = np.float16(sys.argv[6])
    #l = np.float16(sys.argv[7])
    l = int(sys.argv[7])

    ads = int(sys.argv[8])

    # N = 30000
    # Nte = 30
    # p = 100
    # q = 60
    # st = 1000
    #
    # nu = 0.001
    # l = 2
    #
    # ads = 200001

    nl = 5
    nr = 120
    bs = 100
    ins = 1

    # parameter
    dt = 0.01

    Ns_load = 5000
    Nte_load = 30

    st_load = 1000

    st_evo = 100

    Ns = Ns_load

    # define mesh size for x
    # define mesh size for x
    lx, ly = 1, 1
    Nx, Ny = 32, 32

    dx = lx / (Nx)
    dy = ly / (Ny)

    points_x = np.linspace(0, lx - dx, Nx).T
    x = points_x[:, None]
    points_y = np.linspace(0, ly - dy, Ny).T
    y = points_y[:, None]

    xx, yy = np.meshgrid(x, y)

    N_c = 30

    id_c = np.array(random.sample(range(Nx), N_c))
    id_c_2 = np.zeros(int(Nx / 2), )
    for i in range(int(Nx / 2)):
        id_c_2[i] = int(2 * i)

    id_c_2 = id_c_2.astype(int)


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

    filename = 'S_ns_sample_2d' + '_nu_' + num2str_deciaml(nu) + '_Nx_' + num2str_deciaml(
        Nx) + '_l_' + num2str_deciaml(l) + '_dt_' + num2str_deciaml(
        dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(Ns) + '_Nte_' + num2str_deciaml(
        Nte_load)
    npy_name = filename + '.npy'

    with open(npy_name, 'rb') as ss:
        xs = np.load(ss)
        Train_B_load = np.load(ss)
        Train_psi_x_load = np.load(ss)
        Train_psi_y_load = np.load(ss)

        Test_B_load = np.load(ss)

        all_test_f_load = np.load(ss)
        all_test_psi_x_load = np.load(ss)
        all_test_psi_y_load = np.load(ss)
        all_test_evo_mat_load = np.load(ss)

    all_test_f = all_test_f_load[:Nte, :]
    all_test_psi_x = all_test_psi_x_load[:Nte, :]
    all_test_psi_y = all_test_psi_y_load[:Nte, :]
    all_test_evo_mat = all_test_evo_mat_load[:Nte, :st, :]

    Train_B_idx = np.array(random.sample(range(Ns_load), Ns))

    Train_B = Train_B_load[Train_B_idx, :]
    Train_psi_x = Train_psi_x_load[Train_B_idx, :]
    Train_psi_y = Train_psi_y_load[Train_B_idx, :]

    Test_B = Test_B_load[:Nte, :]
    Test_psi_x = all_test_psi_x_load[:Nte, :]
    Test_psi_y = all_test_psi_y_load[:Nte, :]

    t_id = 3

    f_test = all_test_f[t_id, :][None, :]
    f_test_evo_mat = all_test_evo_mat[t_id, :st, :]

    id_all = np.array(random.sample(range(Ns * Nx * Ny), N))
    id_test = np.linspace(0, Nte * Nx * Ny - 1, Nte * Nx * Ny).astype(int)

    x_train = np.kron(x, np.ones((Ny, 1)))
    y_train = np.tile(y, (Nx, 1))

    f_sc = 0.1 * (np.sin(2 * np.pi * (x_train + y_train)) + np.cos(2 * np.pi * (x_train + y_train)))

    ########### check reference by fig #########################
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(1)
    ax = fig.add_subplot()
    cp = ax.contourf(xx, yy, f_test_evo_mat[0, :].reshape(Nx, Nx).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.xlabel('x', fontsize=26)
    plt.ylabel('y', fontsize=26)
    plt.title(r'Reference $f(t=0, x, y)$', fontsize=26)
    plt.tight_layout()
    file = open('figs/' + '_ref_tz.pickle', 'wb')
    pl.dump(fig, file)
    pl.dump(ax, file)
    pl.dump(cp, file)

    fig = plt.figure(2)
    ax = fig.add_subplot()
    cpp = ax.contourf(xx, yy, f_test_evo_mat[100, :].reshape(Nx, Nx).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.xlabel('x', fontsize=26)
    plt.ylabel('y', fontsize=26)
    plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $f(t=20, x, y)$', fontsize=26)
    plt.tight_layout()

    fig = plt.figure(3)
    ax = fig.add_subplot()
    cpp = ax.contourf(xx, yy, f_test_evo_mat[int(st / 2), :].reshape(Nx, Nx).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Reference $f(t=0, x, y)$', fontsize=20)
    plt.tight_layout()

    fig = plt.figure(4)
    ax = fig.add_subplot()
    cpp = ax.contourf(xx, yy, f_test_evo_mat[st - 1, :].reshape(Nx, Nx).T, 100, cmap=cm.jet)
    fig.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Reference $f(t=0, x, y)$', fontsize=20)
    plt.tight_layout()

    plt.show()
    plt.close('all')

    ############################################

    dtype = tf.float32
    num_ad_epochs = ads
    num_bfgs_epochs = 2000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3 / 4,
        decay_steps=3000,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    W_f = np.ones((q, 1))

    w_bc = 5

    train_evo_step = 0

    filename = 'C_ns_2d_dt_pb' + '_nu_' + num2str_deciaml(nu) + '_Nx_' + num2str_deciaml(
        Nx) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(
        Nte) + '_adstep_' + num2str_deciaml(num_ad_epochs) + '_q_' + num2str_deciaml(q) + '_l_' + num2str_deciaml(l)

    file_name = filename + '.txt'
    fig_name = filename
    npy_name = filename + '.npy'

    mdl = NS2D(Nx, Ny, N_c, Ns, Nte, N, nu, lx, dt, x, y, dx, dy, f_sc, Train_B, Train_psi_x, Train_psi_y, Test_B,
               Test_psi_x, Test_psi_y, f_test, f_test_evo_mat,
               all_test_f, all_test_evo_mat, dtype, optimizer, num_ad_epochs,
               num_bfgs_epochs, fig_name, file_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, w_bc, st,
               id_all, id_test)
    mdl.fit()

    model_name = filename
    mdl.save('mdls/' + model_name)