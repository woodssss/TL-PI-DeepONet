################################ TLPIDON rte dt ########################################
# Physics-informed DeepONet with Transfer learning for radiative transfer
# equation in discretized time setting.
# consider (f^{n+1} - f^n) / \Delta t =  <f^{n+1}> - f^{n+1}
# with inflow BC
# f(t,0,v<0)=1, f(t,1,v>0)=1/2
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy import sparse
import random
import pickle as pl


class rte():
    # initialize
    def __init__(self, eps, Nx, Ny, Nx_r, Ny_r, Nv, Ns, Nte, N, dt, lx, x, dx, ly, y, dy, v, w, Train_B_rho, Train_B_g, Train_B_f, id_all,
              x_train, y_train, v_train, Test_B_rho, Test_B_g, Test_B_f, rho_test, g_test, rho_test_evo, g_test_evo, all_test_rho, all_test_g, all_test_f, all_test_rho_evo_mat, all_test_f_evo_mat, f_test_evo, dtype, optimizer, num_ad_epochs, file_name, fig_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st, w_bc_flip):

        self.dtype = dtype

        self.eps = eps

        self.nl, self.nr = nl, nr

        self.Nx, self.Ny, self.Nv, self.Ns = Nx, Ny, Nv, Ns
        self.Nx_r, self.Ny_r = Nx_r, Ny_r
        self.Nte, self.N = Nte, N
        self.lx, self.ly = lx, ly

        self.id_all = id_all
        self.id_test = np.linspace(0, Nte * self.Nx**2 - 1, Nte* self.Nx**2).astype(int)

        self.dt, self.dx, self.dy = dt, dx, dy

        self.xx, self.yy = np.meshgrid(x,y)
        self.xxv, self.vv = np.meshgrid(x, v)

        self.x = tf.convert_to_tensor(x, dtype=self.dtype)
        self.y = tf.convert_to_tensor(y, dtype=self.dtype)

        self.xy = np.kron(x, np.ones((Ny, 1)))
        self.yx = np.tile(y, (Nx, 1))

        self.x_r = tf.convert_to_tensor(self.xy, dtype=self.dtype)
        self.y_r = tf.convert_to_tensor(self.yx, dtype=self.dtype)

        self.v, self.w = tf.convert_to_tensor(v, dtype=self.dtype), tf.convert_to_tensor(w, dtype=self.dtype)

        self.Train_B_rho_tensor = Train_B_rho.reshape(Ns, Nx, Ny)
        self.Train_B_g_tensor = Train_B_g.reshape(Ns, Nx, Ny, Nv)
        self.Train_B_f_tensor = Train_B_f.reshape(Ns, Nx, Ny, Nv)

        self.Test_B_rho_tensor = Test_B_rho.reshape(Nte, Nx, Ny)
        self.Test_B_g_tensor = Test_B_g.reshape(Nte, Nx, Ny, Nv)
        self.Test_B_f_tensor = Test_B_f.reshape(Nte, Nx, Ny, Nv)

        self.Train_B_f = tf.convert_to_tensor(Train_B_f, dtype=self.dtype)
        self.Test_B_f = tf.convert_to_tensor(Test_B_f, dtype=self.dtype)

        self.x_train = tf.convert_to_tensor(x_train, dtype=self.dtype)
        self.y_train = tf.convert_to_tensor(y_train, dtype=self.dtype)
        self.v_train = tf.convert_to_tensor(v_train, dtype=self.dtype)

        self.all_test_f, self.all_test_rho_evo_mat, self.all_test_f_evo_mat = all_test_f, all_test_rho_evo_mat, all_test_f_evo_mat
        self.all_test_rho, self.all_test_g = all_test_rho, all_test_g

        self.rho_test, self.g_test = rho_test, g_test
        self.g_test_evo, self.rho_test_evo = g_test_evo, rho_test_evo

        self.v_mat = tf.convert_to_tensor(np.diag(v_train.T[0]), dtype=self.dtype)

        self.v_x_mat = tf.convert_to_tensor(np.diag(np.cos(v_train).T[0]), dtype=self.dtype)
        self.v_y_mat = tf.convert_to_tensor(np.diag(np.sin(v_train).T[0]), dtype=self.dtype)


        # sample

        self.f_test_evo, self.rho_test_evo = f_test_evo, rho_test_evo

        self.mass_vec = tf.ones([self.Ns, 1])

        self.file_name = file_name
        self.fig_name = fig_name
        self.npy_name = npy_name

        self.num_ad_epochs = num_ad_epochs

        self.optimizer = optimizer

        self.H_input_size = self.Nx * self.Ny * self.Nv

        self.p = p

        self.q = q

        self.kernel_size = 5

        self.W_f = W_f

        self.T_output_size = self.p

        # define parameters for stochastic GD
        self.batch_size = bs  # number of sample at each batch
        self.inner_step = ins  # inner adam step for small batch
        self.inner_bfgs_step = 1  # inner BFGS step for small batch
        self.train_evo_step = train_evo_step

        self.st = st

        self.stop = 0.00000001
        self.w_bc = 1/w_bc_flip

        self.epoch_vec = []
        self.emp_loss_vec = []
        self.test_loss_vec = []

        # Initialize NN
        # Initialize NN
        self.Bnn = self.get_B()
        self.Hnn = self.get_H()
        self.T_rhonn = self.get_T_rho()
        self.T_gnn = self.get_T_g()

        # self.Bnn.summary()
        # self.Tnn.summary()


    def prepare_AVG(self, bs):
        AVG_red = np.kron(np.eye(bs), w.T)
        AVG = np.kron(np.eye(bs), np.tile(w.T, (self.Nv, 1)))

        ext_mat = np.kron(np.eye(bs), np.ones((self.Nv, 1)))

        AVG_red, AVG = tf.convert_to_tensor(AVG_red, dtype=self.dtype), tf.convert_to_tensor(AVG, dtype=self.dtype)

        ext_mat = tf.convert_to_tensor(ext_mat, dtype=self.dtype)

        return AVG_red, AVG, ext_mat

    def get_B(self):

        input_Branch_rho = tf.keras.Input(shape=(self.Nx**2,))
        input_Branch_g = tf.keras.Input(shape=(self.Nx**2 * self.Nv,))

        # get feature
        fea_rho = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                        kernel_initializer='glorot_normal')(
            input_Branch_rho)

        fea_g = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                      kernel_initializer='glorot_normal')(
            input_Branch_g)

        feature_rhog = tf.keras.layers.concatenate([fea_rho, fea_g])

        # define nn for Branch
        input_Branch = feature_rhog

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

        model = tf.keras.Model(inputs=[input_Branch_rho, input_Branch_g], outputs=out)

        return model

    def get_H(self):
        input_Branch_rho = tf.keras.Input(shape=(self.Nx**2,))
        input_Branch_g = tf.keras.Input(shape=(self.Nx**2 * self.Nv,))

        # get feature
        fea_rho = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                        kernel_initializer='glorot_normal')(
            input_Branch_rho)

        fea_g = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                      kernel_initializer='glorot_normal')(
            input_Branch_g)

        feature_rhog = tf.keras.layers.concatenate([fea_rho, fea_g])

        # define nn for Branch
        input_Branch = feature_rhog

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

        model = tf.keras.Model(inputs=[input_Branch_rho, input_Branch_g], outputs=out)

        return model

    def get_T_rho(self):
        ### For rho

        input_Trunk_rho = tf.keras.Input(shape=(2,))

        def pd_feature_rho(ip):
            x = ip[:, 0]
            y = ip[:, 1]
            b1 = tf.cos(x)
            b2 = tf.cos(y)
            out = tf.stack([b1, b2], axis=1)
            return out

        feature_rho = tf.keras.layers.Lambda(pd_feature_rho, name='my_pd_rho_feature')(input_Trunk_rho)

        UT_rho = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                       kernel_initializer='glorot_normal')(
            feature_rho)

        VT_rho = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                       kernel_initializer='glorot_normal')(
            feature_rho)

        input_Trunk_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                    kernel_initializer='glorot_normal')(
            feature_rho)

        for i in range(self.nl - 1):
            B1 = tf.keras.layers.Multiply()([input_Trunk_rho_mid, UT_rho])

            N = tf.keras.layers.Subtract()([tf.ones_like(input_Trunk_rho_mid), input_Trunk_rho_mid])
            B2 = tf.keras.layers.Multiply()([N, VT_rho])

            input_Trunk_rho_mid = tf.keras.layers.Add()([B1, B2])

        output_Trunk_rho = tf.keras.layers.Dense(units=self.T_output_size, activation=None,
                                                 kernel_initializer='glorot_normal')(
            input_Trunk_rho_mid)

        model = tf.keras.Model(inputs=input_Trunk_rho, outputs=output_Trunk_rho)

        return model

    def get_T_g(self):

        input_Trunk_g = tf.keras.Input(shape=(3,))

        def pd_feature_g(ip):
            x = ip[:, 0]
            y = ip[:, 1]
            v = ip[:, 2]
            b1 = tf.cos(x)
            b2 = tf.cos(y)
            out = tf.stack([b1, b2, v], axis=1)
            return out

        feature_g = tf.keras.layers.Lambda(pd_feature_g, name='my_pd_g_feature')(input_Trunk_g)

        UT_g = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                     kernel_initializer='glorot_normal')(
            feature_g)

        VT_g = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                     kernel_initializer='glorot_normal')(
            feature_g)

        input_Trunk_g_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                  kernel_initializer='glorot_normal')(
            feature_g)

        for i in range(self.nl - 1):
            B1 = tf.keras.layers.Multiply()([input_Trunk_g_mid, UT_g])

            N = tf.keras.layers.Subtract()([tf.ones_like(input_Trunk_g_mid), input_Trunk_g_mid])
            B2 = tf.keras.layers.Multiply()([N, VT_g])

            input_Trunk_g_mid = tf.keras.layers.Add()([B1, B2])

        output_Trunk_g = tf.keras.layers.Dense(units=self.T_output_size, activation=tf.nn.tanh,
                                               kernel_initializer='glorot_normal')(
            input_Trunk_g_mid)

        model = tf.keras.Model(inputs=input_Trunk_g, outputs=output_Trunk_g)

        return model



    ################ define loss function ####################################################
    def get_Bf(self, x, y, vx, vy):
        return tf.nn.relu(vx) * x * y * (1 - y) + tf.nn.relu(-vx) * (1 - x) * y * (1 - y) + tf.nn.relu(vy) * y * x * (
                1 - x) + tf.nn.relu(-vy) * (1 - y) * x * (1 - x) + tf.nn.relu(vx) * tf.nn.relu(vy) * x * y + tf.nn.relu(
            vx) * tf.nn.relu(-vy) * x * (1 - y) + tf.nn.relu(-vx) * tf.nn.relu(vy) * (1 - x) * y + tf.nn.relu(
            -vx) * tf.nn.relu(-vy) * (1 - x) * (1 - y)

    def get_Bx(self, x, y, vx, vy):
        return y * (1 - y) * vx + tf.nn.relu(vy) * y * (1 - 2 * x) + tf.nn.relu(-vy) * (1 - y) * (
                    1 - 2 * x) + vx * tf.nn.relu(vy) * y + vx * tf.nn.relu(-vy) * (1 - y)

    def get_By(self, x, y, vx, vy):
        return tf.nn.relu(vx) * x * (1 - 2 * y) + tf.nn.relu(-vx) * (1 - x) * (1 - 2 * y) + x * (
                    1 - x) * vy + vy * tf.nn.relu(vx) * x + vy * tf.nn.relu(-vx) * (1 - x)

    def get_A(self, x, y):
        return x*(1-x)*y*(1-y)

    def get_Ax(self, x, y):
        return (1-2*x)*y*(1-y)

    def get_Ay(self, x, y):
        return (1-2*y)*x*(1-x)

    def get_C(self, x, y):
        return (0.25 - (y - 0.5) ** 2) * (1 - x) + 0.25

    def get_Cx(self, x, y):
        return -0.25 + (y - 0.5) ** 2

    def get_Cy(self, x, y):
        return -2 * (y - 0.5) * (1 - x)

    def get_loss(self, tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g):

        bs = tmp_B_rho.shape[0]

        # print(bs, tmp_B_rho.shape, tmp_B_g.shape, tmp_x.shape, tmp_IC_rho.shape, tmp_IC_g.shape)
        # zxc

        tmp_T_x = tmp_x
        tmp_T_y = tmp_y

        tmp_T_x_vec = np.kron(tmp_T_x.numpy(), np.ones((self.Nv, 1)))
        tmp_T_y_vec = np.kron(tmp_T_y.numpy(), np.ones((self.Nv, 1)))
        tmp_T_v_vec = np.tile(v, (bs, 1))

        tmp_T_x_vec = tf.convert_to_tensor(tmp_T_x_vec, dtype=self.dtype)
        tmp_T_y_vec = tf.convert_to_tensor(tmp_T_y_vec, dtype=self.dtype)
        tmp_T_v_vec = tf.convert_to_tensor(tmp_T_v_vec, dtype=self.dtype)

        ext_B = tf.convert_to_tensor(np.kron(np.eye(bs), np.ones((self.Nv, 1))), dtype=self.dtype)


        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_T_x)
            tape.watch(tmp_T_y)
            tape.watch(tmp_T_x_vec)
            tape.watch(tmp_T_y_vec)

            Train_T_rho = tf.concat([tmp_T_x, tmp_T_y], axis=1)
            Train_T_g = tf.concat([tmp_T_x_vec, tmp_T_y_vec, tmp_T_v_vec], axis=1)

            T_rho = self.T_rhonn(Train_T_rho)
            T_g = self.T_gnn(Train_T_g)

            B_ori = self.Bnn([tmp_B_rho, tmp_B_g])[:, :, 0]

            B = tf.matmul(ext_B, B_ori)

            d_rho = tf.reduce_sum(B_ori * T_rho, axis=1, keepdims=True)
            d_g = tf.reduce_sum(B * T_g, axis=1, keepdims=True)

            d_rho_x = tape.gradient(d_rho, tmp_T_x)
            d_rho_y = tape.gradient(d_rho, tmp_T_y)

            d_g_x = tape.gradient(d_g, tmp_T_x_vec)
            d_g_y = tape.gradient(d_g, tmp_T_y_vec)


        AVG_red, AVG, ext_mat = self.prepare_AVG(bs) # time 1/2/pi afterward

        rho_old = tf.reshape(tmp_IC_rho, [bs, 1])
        g_old = tf.reshape(tmp_IC_g, [bs * self.Nv, 1])

        A = self.get_A(tmp_T_x, tmp_T_y)
        A_x = self.get_Ax(tmp_T_x, tmp_T_y)
        A_y = self.get_Ay(tmp_T_x, tmp_T_y)

        C = self.get_C(tmp_T_x, tmp_T_y)
        C_x = self.get_Cx(tmp_T_x, tmp_T_y)
        C_y = self.get_Cy(tmp_T_x, tmp_T_y)

        B = self.get_Bf(tmp_T_x_vec, tmp_T_y_vec, tf.cos(tmp_T_v_vec), tf.sin(tmp_T_v_vec))
        B_x = self.get_Bx(tmp_T_x_vec, tmp_T_y_vec, tf.cos(tmp_T_v_vec), tf.sin(tmp_T_v_vec))
        B_y = self.get_By(tmp_T_x_vec, tmp_T_y_vec, tf.cos(tmp_T_v_vec), tf.sin(tmp_T_v_vec))

        rho = d_rho*A + C + self.eps / 2 / np.pi * tf.linalg.matmul(AVG_red, d_g * B)
        rho_x = d_rho_x * A + d_rho * A_x + C_x + self.eps / 2 / np.pi * tf.linalg.matmul(AVG_red,
                                                                                          d_g_x * B + d_g * B_x)
        rho_y = d_rho_y * A + d_rho * A_y + C_y + self.eps / 2 / np.pi * tf.linalg.matmul(AVG_red,
                                                                                          d_g_y * B + d_g * B_y)

        rho_x_vec = tf.matmul(ext_B, rho_x)
        rho_y_vec = tf.matmul(ext_B, rho_y)

        g = d_g * B - tf.linalg.matmul(AVG, d_g * B) / 2 / np.pi
        g_x = d_g_x * B + d_g * B_x - tf.linalg.matmul(AVG, d_g_x * B + d_g * B_x) / 2 / np.pi
        g_y = d_g_y * B + d_g * B_y - tf.linalg.matmul(AVG, d_g_y * B + d_g * B_y) / 2 / np.pi


        # pde1
        # rho_t + \partial_x <vg> = 0

        vxgx_plus_vygy = tf.cos(tmp_T_v_vec) * g_x + tf.sin(tmp_T_v_vec) * g_y
        vxrhox_plus_vyrhoy_vec = tf.cos(tmp_T_v_vec) * rho_x_vec + tf.sin(tmp_T_v_vec) * rho_y_vec

        pde1 = rho - rho_old + self.dt / 2 / np.pi * tf.linalg.matmul(AVG_red, vxgx_plus_vygy)

        # print(f_old.shape, f.shape, bs, pde.shape)
        # zxc

        # print(tmp_T_v_vec, tf.cos(tmp_T_v_vec), tf.sin(tmp_T_v_vec))
        # zxc

        # pde2

        pde2 = self.eps ** 2 * (g - g_old) + self.eps * self.dt * vxgx_plus_vygy - self.dt * self.eps * tf.linalg.matmul(AVG, vxgx_plus_vygy)/2/np.pi + self.dt*vxrhox_plus_vyrhoy_vec + self.dt*g


        loss_pde = tf.reduce_mean(tf.square(pde1)) + tf.reduce_mean(tf.square(pde2))

        loss = loss_pde

        return loss

    def get_total_loss(self, id_all, N, Nl, B_rho, B_g, B_rho_tensor, B_g_tensor):

        # Nl is the small batch size for total loss

        part_num = N // Nl

        total_loss = 0

        for i in range(part_num):
            id_bs = id_all[i * Nl:(i + 1) * Nl]

            # print(id_all.shape, id_bs1.shape, id_bs.shape)
            # zxc

            id_k_vec, id_i_vec, id_j_vec = self.get_id_bs_vec(id_bs, self.Nx, self.Ny)

            tmp_B_rho = tf.convert_to_tensor(B_rho[id_k_vec, :], dtype=self.dtype)

            tmp_B_g = tf.convert_to_tensor(B_g[id_k_vec, :], dtype=self.dtype)

            tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

            tmp_y = tf.convert_to_tensor(y[id_j_vec, :], dtype=self.dtype)

            tmp_IC_rho = tf.convert_to_tensor(B_rho_tensor[id_k_vec, id_i_vec, id_j_vec], dtype=self.dtype)

            tmp_IC_g = tf.convert_to_tensor(B_g_tensor[id_k_vec, id_i_vec, id_j_vec, :], dtype=self.dtype)

            # print(tmp_y.shape, tmp_f.shape)
            # zxc

            tmp_loss = self.get_loss(tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g)

            # print(tmp_loss)

            total_loss = total_loss + tmp_loss

        return total_loss / part_num



    def get_grad(self, tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g):
        with tf.GradientTape() as tape:
            loss = self.get_loss(tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g)
            trainable_weights_B= self.Bnn.trainable_variables
            trainable_weights_T_rho = self.T_rhonn.trainable_variables
            trainable_weights_T_g = self.T_gnn.trainable_variables

            trainable_weights = trainable_weights_B + trainable_weights_T_rho + trainable_weights_T_g

        grad = tape.gradient(loss, trainable_weights)

        return loss, trainable_weights, grad

    def id_2_id_kij(self, id, Nx, Ny):
        id_k = id // (Nx * Ny)
        id_i = (id - id_k * Nx * Ny) // Ny
        id_j = (id - id_k * Nx * Ny - id_i * Ny)

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

        id_k_vec, id_i_vec, id_j_vec = self.get_id_bs_vec(id_bs, self.Nx, self.Ny)

        tmp_B_rho = tf.convert_to_tensor(Train_B_rho[id_k_vec, :], dtype=self.dtype)

        tmp_B_g = tf.convert_to_tensor(Train_B_g[id_k_vec, :], dtype=self.dtype)

        tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

        tmp_y = tf.convert_to_tensor(y[id_j_vec, :], dtype=self.dtype)

        tmp_IC_rho = tf.convert_to_tensor(self.Train_B_rho_tensor[id_k_vec, id_i_vec, id_j_vec], dtype=self.dtype)

        tmp_IC_g = tf.convert_to_tensor(self.Train_B_g_tensor[id_k_vec, id_i_vec, id_j_vec, :], dtype=self.dtype)

        return tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g


    # Adam step
    def adam_step(self):
        tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g = self.get_random_sample(self.id_all)
        for i in range(self.inner_step):
            with tf.device('/device:GPU:0'):
                loss, trainable_weights, grad = self.get_grad(tmp_B_rho, tmp_B_g, tmp_x, tmp_y, tmp_IC_rho, tmp_IC_g)
                self.optimizer.apply_gradients(zip(grad, trainable_weights))

    def fit(self):
        start_time = time.time()
        Initial_time = time.time()
        self.check_with_visual()
        for epoch in range(self.num_ad_epochs):
            self.adam_step()
            if epoch % 5000 == 0:
                elapsed = time.time() - start_time
                total_elapsed = time.time() - Initial_time
                start_time = time.time()
                total_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B_rho, Train_B_g,
                                                 self.Train_B_rho_tensor, self.Train_B_g_tensor)
                test_loss = self.get_total_loss(self.id_test, self.Nte * self.Nx**2, 100, Test_B_rho, Test_B_g,
                                                self.Test_B_rho_tensor, self.Test_B_g_tensor)
                print('Nx: %d, Nv: %d, N: %d, Nte: %d, dt: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.Nv, self.N, self.Nte, self.dt, self.st, self.p, self.q))
                print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                      (epoch, total_loss, test_loss, elapsed, total_elapsed))
                with open(self.file_name, 'a') as fw:
                    print('Nx: %d, Nv: %d, N: %d, Nte: %d, dt: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.Nv, self.N, self.Nte, self.dt, self.st, self.p, self.q), file=fw)
                    print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f, Total Time: %.2f' %
                          (epoch, total_loss, test_loss, elapsed, total_elapsed), file=fw)

                self.epoch_vec.append(epoch)
                self.emp_loss_vec.append(total_loss)
                self.test_loss_vec.append(test_loss)

                if total_loss < self.stop:
                    print('Adam training finished')
                    self.check_with_visual()
                    with open(self.file_name, 'a') as fw:
                        print('Adam training finished', file=fw)
                    break

            #if epoch % (self.num_ad_epochs-1) == 0 and epoch>0:
            if epoch % 20000 == 0 and epoch > 0:
                self.check_with_visual()


        final_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B_rho, Train_B_g,
                                                 self.Train_B_rho_tensor, self.Train_B_g_tensor)
        print('Final loss is %.3e' % final_loss)
        with open(self.file_name, 'a') as fw:
            print('Final loss is %.3e' % final_loss, file=fw)

    def get_evo_pred(self, rho, g):

        AVG_red, AVG, ext_mat = self.prepare_AVG(self.Nx**2)  # time 0.5 afterward

        AVG_red, AVG, ext_mat = AVG_red.numpy(), AVG.numpy(), ext_mat.numpy()

        Train_T_rho = tf.concat([self.x_r, self.y_r], axis=1)
        Train_T_g = tf.concat([self.x_train, self.y_train, self.v_train], axis=1)

        T_rho = self.T_rhonn(Train_T_rho)
        T_g = self.T_gnn(Train_T_g)

        rho_evo = np.zeros((self.st, self.Nx * self.Ny))
        g_evo = np.zeros((self.st, self.Nx * self.Ny * self.Nv))

        Bf = self.get_Bf(self.x_train, self.y_train, tf.cos(self.v_train), tf.sin(self.v_train)).numpy()
        A = self.get_A(self.x_r, self.y_r).numpy()
        C = self.get_C(self.x_r, self.y_r).numpy()

        for i in range(self.st):

            B = self.Bnn([rho, g])[:, :, 0]

            d_rho = tf.reduce_sum(B * T_rho, 1, keepdims=True).numpy()
            d_g = tf.reduce_sum(B * T_g, 1, keepdims=True).numpy()

            rho = d_rho*A + C + 1/2/np.pi*AVG_red.dot(d_g*Bf)

            g = d_g*Bf - 1/2/np.pi*AVG.dot(d_g*Bf)

            rho, g = rho.T, g.T

            rho_evo[i, :] = rho
            g_evo[i, :] = g

        return rho_evo, g_evo

    def get_Tevo_mat_g_sm(self, tmp_x, tmp_y, tmp_v):
        # T
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_x)
            tape.watch(tmp_y)

            Train_T = tf.concat([tmp_x, tmp_y, tmp_v], axis=1)

            T_g = self.T_gnn(Train_T)

            T_g_x_list = []
            T_g_y_list = []

            for i in range(self.p):
                T_g_tmp = T_g[:, i][:, None]

                T_g_x_tmp = tape.gradient(T_g_tmp, tmp_x)
                T_g_y_tmp = tape.gradient(T_g_tmp, tmp_y)

                T_g_x_list.append(T_g_x_tmp)
                T_g_y_list.append(T_g_y_tmp)

        T_g_x = tf.concat(T_g_x_list, axis=1)
        T_g_y = tf.concat(T_g_y_list, axis=1)

        return T_g, T_g_x, T_g_y

    def get_Tevo_mat_rho(self):
        tmp_x, tmp_y = self.x_r, self.y_r
        # T
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_x)
            tape.watch(tmp_y)

            Train_T = tf.concat([tmp_x, tmp_y], axis=1)

            T_rho = self.T_rhonn(Train_T)

            T_rho_x_list = []
            T_rho_y_list = []

            for i in range(self.p):
                T_rho_tmp = T_rho[:, i][:, None]

                T_rho_x_tmp = tape.gradient(T_rho_tmp, tmp_x)
                T_rho_y_tmp = tape.gradient(T_rho_tmp, tmp_y)

                T_rho_x_list.append(T_rho_x_tmp)
                T_rho_y_list.append(T_rho_y_tmp)

        T_rho_x = tf.concat(T_rho_x_list, axis=1)
        T_rho_y = tf.concat(T_rho_y_list, axis=1)

        del tape

        return T_rho, T_rho_x, T_rho_y

    def get_Tevo_mat_cat(self, Nl):

        T_rho, T_rho_x, T_rho_y = self.get_Tevo_mat_rho()

        T_rho, T_rho_x, T_rho_y = T_rho.numpy(), T_rho_x.numpy(), T_rho_y.numpy()

        num_part = self.Nx*self.Ny*self.Nv // Nl

        T_g_list = []
        T_g_x_list = []
        T_g_y_list = []

        for i in range(num_part):
            tmp_x, tmp_y, tmp_v = self.x_train[i*Nl:(i+1)*Nl, 0][:, None], self.y_train[i*Nl:(i+1)*Nl, 0][:, None], self.v_train[i*Nl:(i+1)*Nl, 0][:, None]
            T_g_tmp, T_g_x_tmp, T_g_y_tmp = self.get_Tevo_mat_g_sm(tmp_x, tmp_y, tmp_v)

            T_g_list.append(T_g_tmp.numpy())
            T_g_x_list.append(T_g_x_tmp.numpy())
            T_g_y_list.append(T_g_y_tmp.numpy())

        T_g = np.concatenate(T_g_list, axis=0)
        T_g_x = np.concatenate(T_g_x_list, axis=0)
        T_g_y = np.concatenate(T_g_y_list, axis=0)

        # print(H.shape, H_tmp.shape)
        # zxc

        return T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y



    def get_Tevo_pred(self, T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y, rho, g):

        # prepare matrix
        AVG_red, AVG, ext_mat = self.prepare_AVG(self.Nx*self.Ny)  # time 1/2/pi afterward

        AVG_red, AVG, ext_mat = AVG_red.numpy()/2/np.pi, AVG.numpy()/2/np.pi, ext_mat.numpy()

        rho_evo = np.zeros((self.st, self.Nx * self.Ny))
        g_evo = np.zeros((self.st, self.Nx * self.Ny * self.Nv))

        # def A, B, C
        A = self.get_A(self.x_r, self.y_r).numpy()
        # DA = np.diag(A[:, 0])
        # DA2 = sparse.spdiags(A[:, 0], 0, self.Nx ** 2, self.Nx ** 2)
        #
        # print(np.max(np.abs(DA - DA2.todense())))
        # zxc



        A_x = self.get_Ax(self.x_r, self.y_r).numpy()
        A_y = self.get_Ay(self.x_r, self.y_r).numpy()

        B = self.get_Bf(self.x_train, self.y_train, tf.cos(self.v_train), tf.sin(self.v_train)).numpy()
        B_x = self.get_Bx(self.x_train, self.y_train, tf.cos(self.v_train), tf.sin(self.v_train)).numpy()
        B_y = self.get_By(self.x_train, self.y_train, tf.cos(self.v_train), tf.sin(self.v_train)).numpy()

        C = self.get_C(self.x_r, self.y_r).numpy()
        C_x = self.get_Cx(self.x_r, self.y_r).numpy()
        C_y = self.get_Cy(self.x_r, self.y_r).numpy()
        C_x_vec = self.get_Cx(self.x_train, self.y_train).numpy()
        C_y_vec = self.get_Cy(self.x_train, self.y_train).numpy()

        ####### define matrix by numpy ##############

        # DA = np.diag(A[:, 0])
        # DA_x = np.diag(A_x[:, 0])
        # DA_y = np.diag(A_y[:, 0])
        # DB = np.diag(B[:, 0])
        # DB_x = np.diag(B_x[:, 0])
        # DB_y = np.diag(B_y[:, 0])
        # DC = np.diag(C[:, 0])
        # DC_x = np.diag(C_x[:, 0])
        # DC_y = np.diag(C_y[:, 0])
        #
        # DVx = np.diag((tf.cos(self.v_train)).numpy()[:, 0])
        # DVy = np.diag((tf.sin(self.v_train)).numpy()[:, 0])
        #
        # Ext = np.kron(np.eye(self.Nx ** 2), np.ones((Nv, 1)))
        # PI = np.eye(self.Nx ** 2 * self.Nv) - AVG

        ##############################################

        ########## define matrix by sparse ##################

        DA = sparse.spdiags(A[:, 0], 0, self.Nx ** 2, self.Nx ** 2)
        DA_x = sparse.spdiags(A_x[:, 0], 0, self.Nx ** 2, self.Nx ** 2)
        DA_y = sparse.spdiags(A_y[:, 0], 0, self.Nx ** 2, self.Nx ** 2)

        DB = sparse.spdiags(B[:, 0], 0, self.Nx ** 2*self.Nv, self.Nx ** 2*self.Nv)
        DB_x = sparse.spdiags(B_x[:, 0], 0, self.Nx ** 2 * self.Nv, self.Nx ** 2 * self.Nv)
        DB_y = sparse.spdiags(B_y[:, 0], 0, self.Nx ** 2 * self.Nv, self.Nx ** 2 * self.Nv)

        DC = sparse.spdiags(C[:, 0], 0, self.Nx ** 2, self.Nx ** 2)
        DC_x = sparse.spdiags(C_x[:, 0], 0, self.Nx ** 2, self.Nx ** 2)
        DC_y = sparse.spdiags(C_y[:, 0], 0, self.Nx ** 2, self.Nx ** 2)

        DVx = sparse.spdiags((tf.cos(self.v_train)).numpy()[:, 0], 0, self.Nx ** 2 * self.Nv, self.Nx ** 2 * self.Nv)
        DVy = sparse.spdiags((tf.sin(self.v_train)).numpy()[:, 0], 0, self.Nx ** 2 * self.Nv, self.Nx ** 2 * self.Nv)

        Ext = np.kron(np.eye(self.Nx ** 2), np.ones((Nv, 1)))
        PI = np.eye(self.Nx ** 2 * self.Nv) - AVG

        ######################################################

        # print(DB.shape, T_g.shape, T_g_x.shape, DVx.shape)
        # zxc

        vgx_plus_vgy = DVx.dot(PI.dot(DB.dot(T_g_x) + DB_x.dot(T_g))) + DVy.dot(PI.dot(DB.dot(T_g_y) + DB_y.dot(T_g)))

        L11 = DA.dot(T_rho)
        L12 = self.dt * AVG_red.dot(vgx_plus_vgy) + self.eps*AVG_red.dot(DB.dot(T_g))

        L21 = self.dt*DVx.dot(Ext.dot(DA.dot(T_rho_x) + DA_x.dot(T_rho))) + self.dt*DVy.dot(Ext.dot(DA.dot(T_rho_y) + DA_y.dot(T_rho)))
        L22 = (self.eps**2 + self.dt)*PI.dot(DB.dot(T_g)) + self.dt*self.eps*PI.dot( DVx.dot(PI.dot(DB.dot(T_g_x) + DB_x.dot(T_g))) + DVy.dot(PI.dot(DB.dot(T_g_y) + DB_y.dot(T_g))) ) + self.dt*self.eps*DVx.dot(AVG.dot(DB.dot(T_g_x) + DB_x.dot(T_g))) + self.dt*self.eps*DVy.dot(AVG.dot(DB.dot(T_g_y) + DB_y.dot(T_g)))


        for i in range(self.st):

            H = self.Hnn([rho, g])[0, :, :]
            H = H.numpy()

            K11 = np.matmul(L11, H)
            K12 = np.matmul(L12, H)

            K21 = np.matmul(L21, H)
            K22 = np.matmul(L22, H)

            K1 = np.concatenate([K11, K12], axis=1)
            K2 = np.concatenate([K21, K22], axis=1)

            K = np.concatenate([K1, K2], axis=0)

            RHS1 = rho.T - C
            RHS2 = self.eps**2*g.T - self.dt*np.cos(self.v_train.numpy())*C_x_vec - self.dt*np.sin(self.v_train.numpy())*C_y_vec

            RHS = np.concatenate([RHS1, RHS2], axis=0)

            #W_tmp = np.linalg.lstsq(K, RHS, 0.00000000001)[0]

            L = np.matmul(K.T, K)
            R = np.matmul(K.T, RHS)

            W_tmp = np.linalg.lstsq(L, R, 0.0000000001)[0]

            # print(np.max(np.abs(np.matmul(K, W_tmp) - RHS)), np.max(np.abs(W_tmp)))
            # zxc

            W_tmp_rho = W_tmp[:self.q, [0]]
            W_tmp_g = W_tmp[self.q:, [0]]

            #d_rho = (np.matmul(np.matmul(T_rho, H), W_tmp_rho))
            d_rho = (T_rho.dot(H)).dot(W_tmp_rho)

            d_g = (np.matmul(np.matmul(T_g, H), W_tmp_g))

            # rho = np.matmul(np.matmul(np.matmul(DA, T_rho), H), W_tmp_rho) + C + self.eps * np.matmul(AVG_red,
            #                                                                                           B * d_g)

            rho = d_rho*A + C + self.eps*AVG_red.dot(B*d_g)
            g = (B * d_g - np.matmul(AVG, B * d_g))

            rho, g = rho.T, g.T

            # print(f_old.shape, T.shape, H.shape, W_tmp.shape, g.shape, f_next.shape, K.shape, f_old_and_more.shape)
            # zxc

            rho_evo[i, :] = rho
            g_evo[i, :] = g

        return rho_evo, g_evo

    def f2rho(self, f):
        f_mat = f.reshape(self.Nx, self.Ny, self.Nv)

        rho_tmp = np.zeros((self.Nx, self.Ny))

        for i in range(self.Nx):
            for j in range(self.Ny):
                rho_tmp[i, j] = np.sum(f_mat[[i], [j], :] * w.T) / 2 / np.pi

        return rho_tmp

    def f_mat2rho(self, f_mat):
        rho_tmp = np.zeros((self.Nx, 1))

        for i in range(self.Nx):
            rho_tmp[i, 0] = np.sum(f_mat[[i], :] * w.T) / 2

        return rho_tmp

    def get_l2_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            f1_mat = f1[i, :].reshape(self.Nx, self.Ny, self.Nv)
            f2_mat = f2[i, :].reshape(self.Nx, self.Ny, self.Nv)
            e_mat = np.square(f1_mat - f2_mat)
            error_tmp = 0
            for j in range(self.Nx):
                for k in range(self.Ny):
                    error_tmp = error_tmp + np.sum(e_mat[[j], [k], :] * w.T) * self.dx * self.dy

            error[0, i] = np.sqrt(error_tmp)

        return error

    def get_re_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            f1_mat = f1[i, :].reshape(self.Nx, self.Ny, self.Nv)
            f2_mat = f2[i, :].reshape(self.Nx, self.Ny, self.Nv)
            e_mat = np.square(f1_mat - f2_mat)
            error_tmp = 0
            f2_sq = np.square(f2_mat)
            f2_mass = 0
            for j in range(self.Nx):
                for k in range(self.Ny):
                    error_tmp = error_tmp + np.sum(e_mat[[j], [k], :] * w.T) * self.dx * self.dy
                    f2_mass = f2_mass + np.sum(f2_sq[[j], [k], :] * w.T) * self.dx * self.dy

            error[0, i] = np.sqrt(error_tmp/f2_mass)
        return error

    def all_test_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx * self.Ny * self.Nv))

        p_start_time = time.time()

        for i in range(self.Nte):
            # print(self.all_test_f.shape, i)
            tmp_rho = self.all_test_rho[[i], :]
            tmp_g = self.all_test_g[[i], :]

            rho_tmp_pred, g_tmp_pred = self.get_evo_pred(tmp_rho, tmp_g)

            rho_vec_tmp_pred = np.kron(rho_tmp_pred, np.ones((1, Nv)))
            f_tmp_pred = rho_vec_tmp_pred + self.eps * g_tmp_pred

            f_tmp_ref = self.all_test_f_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_f_evo_mat)) / np.sum(
            np.square(self.all_test_f_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time

    def all_test_tr_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        #T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y = self.get_Tevo_mat_cat(self.Nx)
        T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y = self.get_Tevo_mat_cat(self.Nv*2)

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx *self.Ny * self.Nv))

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_rho = self.all_test_rho[[i], :]
            tmp_g = self.all_test_g[[i], :]

            rho_tmp_pred, g_tmp_pred = self.get_Tevo_pred(T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y, tmp_rho, tmp_g)

            rho_vec_tmp_pred = np.kron(rho_tmp_pred, np.ones((1, Nv)))
            f_tmp_pred = rho_vec_tmp_pred + self.eps * g_tmp_pred

            f_tmp_ref = self.all_test_f_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_f_evo_mat)) / np.sum(
            np.square(self.all_test_f_evo_mat)))

        return error_mat, re_error_mat, re_error_T, f_pred_tensor, pred_time


    def check_with_visual(self):

        model_name = self.file_name
        self.save('mdls/' + model_name)

        # assign weight to H
        weights_B = self.Bnn.get_weights()
        weights_H = weights_B[:-1]
        self.Hnn.set_weights(weights_H)

        ###### by PIDON
        rho_evo_mat, g_evo_mat = self.get_evo_pred(self.rho_test, self.g_test)
        f_evo_mat = np.kron(rho_evo_mat, np.ones((1, self.Nv))) + self.eps * g_evo_mat

        test_evo_error, test_evo_re_error, test_evo_re_error_T, f_pred_tensor, pred_time = self.all_test_result()

        ###### by TPIDON
        #T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y = self.get_Tevo_mat_cat(self.Nx)
        T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y = self.get_Tevo_mat_cat(self.Nv*2)

        rho_tr_evo_mat, g_tr_evo_mat = self.get_Tevo_pred(T_rho, T_rho_x, T_rho_y, T_g, T_g_x, T_g_y, self.rho_test, self.g_test)
        f_tr_evo_mat = np.kron(rho_tr_evo_mat, np.ones((1, self.Nv))) + self.eps * g_tr_evo_mat

        ###############################################
        plt.rcParams.update({'font.size': 16})
        t_id = 1

        fig = plt.figure(2)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.rho_test_evo[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $\rho(t=0, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_tz.eps', format='eps')


        fig = plt.figure(3)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_tz.eps', format='eps')

        fig = plt.figure(4)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_tz.eps', format='eps')

        fig = plt.figure(5)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xxv, self.vv,
                         self.f_test_evo[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title(r'Reference $f(t=0, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_fxv_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_fxv_tz.eps', format='eps')

        fig = plt.figure(6)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                         f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $f(t=0, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_fxv_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_fxv_tz.eps', format='eps')

        fig = plt.figure(7)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $f(t=0, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_fxv_tz.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_fxv_tz.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 10

        fig = plt.figure(21)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.rho_test_evo[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $\rho(t=0.1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_tzp1.eps', format='eps')

        fig = plt.figure(31)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0.1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_tzp1.eps', format='eps')

        fig = plt.figure(41)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0.1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_tzp1.eps', format='eps')

        fig = plt.figure(51)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xxv, self.vv,
                         self.f_test_evo[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title(r'Reference $f(t=0.1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_fxv_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_fxv_tzp1.eps', format='eps')

        fig = plt.figure(61)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                         f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $ f(t=0.1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_fxv_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_fxv_tzp1.eps', format='eps')

        fig = plt.figure(71)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $ f(t=0.1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_fxv_tzp1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_fxv_tzp1.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 50

        fig = plt.figure(22)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.rho_test_evo[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $\rho(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_tzp5.eps', format='eps')

        fig = plt.figure(32)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_tzp5.eps', format='eps')

        fig = plt.figure(42)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $\rho(t=0.5, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_tzp5.eps', format='eps')

        fig = plt.figure(52)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xxv, self.vv,
                         self.f_test_evo[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title(r'Reference $f(t=0.5, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_fxv_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_fxv_tzp5.eps', format='eps')

        fig = plt.figure(62)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                         f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $ f(t=0.5, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_fxv_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_fxv_tzp5.eps', format='eps')

        fig = plt.figure(72)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $ f(t=0.5, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_fxv_tzp5.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_fxv_tzp5.eps', format='eps')

        ################################################################

        ###############################################
        t_id = 99

        fig = plt.figure(23)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.rho_test_evo[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $\rho(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t1.eps', format='eps')

        fig = plt.figure(33)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $\rho(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t1.eps', format='eps')

        fig = plt.figure(43)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $\rho(t=1, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t1.eps', format='eps')

        fig = plt.figure(53)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xxv, self.vv,
                         self.f_test_evo[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title(r'Reference $f(t=1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_fxv_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_fxv_t1.eps', format='eps')

        fig = plt.figure(63)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                         f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $ f(t=1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_fxv_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_fxv_t1.eps', format='eps')

        fig = plt.figure(73)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $ f(t=1, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_fxv_t1.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_fxv_t1.eps', format='eps')

        ###############################################
        t_id = 999

        fig = plt.figure(24)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xx, self.yy, self.rho_test_evo[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title(r'Reference $\rho(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_t_end.eps', format='eps')

        fig = plt.figure(34)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $\rho(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_t_end.eps', format='eps')

        fig = plt.figure(44)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xx, self.yy, rho_tr_evo_mat[t_id, :].reshape(self.Nx, self.Nx).T, 100, cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $\rho(t=10, x, y)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_t_end.eps', format='eps')

        fig = plt.figure(54)
        ax = fig.add_subplot()
        cp = ax.contourf(self.xxv, self.vv,
                         self.f_test_evo[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                         cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title(r'Reference $f(t=10, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_ref_fxv_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_ref_fxv_t_end.eps', format='eps')

        fig = plt.figure(64)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_tr_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' $ f(t=10, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_TPIDON_fxv_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_TPIDON_fxv_t_end.eps', format='eps')

        fig = plt.figure(74)
        ax = fig.add_subplot()
        cpp = ax.contourf(self.xxv, self.vv,
                          f_evo_mat[t_id, :].reshape(self.Nx, self.Ny, self.Nv)[:, int(Nx / 2), :].T, 100,
                          cmap=cm.jet)
        fig.colorbar(cp)
        plt.xlabel('x', fontsize=26)
        plt.ylabel(r'$\alpha$', fontsize=26)
        plt.title('DeepONet ' + chr(0x215F + 2) + r' $ f(t=10, x, y=0.5, \alpha)$', fontsize=26)
        plt.tight_layout()
        file = open('figs/' + self.fig_name + '_DON_fxv_t_end.pickle', 'wb')
        pl.dump(fig, file)
        fig.savefig('figs/' + self.fig_name + '_DON_fxv_t_end.eps', format='eps')

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

        plt.rcParams.update({'font.size': 11})

        fig1 = plt.figure(1)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        plt.title('loss vs iteration')
        fig1.savefig('figs/' + self.fig_name + '_loss_vs_iter.eps', format='eps')

        plt.rcParams.update({'font.size': 10})

        fig1 = plt.figure(1)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        plt.title('loss vs iteration')
        fig1.savefig('figs/' + self.fig_name + '_loss_vs_iter.eps', format='eps')

        fig8 = plt.figure(8)
        for i in range(self.Nte):
            plt.semilogy(st_vec, test_evo_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2))
        fig8.savefig('figs/' + self.fig_name + '_PIDON_error.eps', format='eps')

        fig9 = plt.figure(9)
        for i in range(self.Nte):
            plt.semilogy(st_vec, test_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'Relative $L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2))
        fig9.savefig('figs/' + self.fig_name + '_PIDON_re_error.eps', format='eps')

        fig10 = plt.figure(10)
        for i in range(self.Nte):
            plt.semilogy(st_vec, test_tr_evo_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_tr_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2))
        fig10.savefig('figs/' + self.fig_name + '_TPIDON_error.eps', format='eps')

        fig11 = plt.figure(11)
        for i in range(self.Nte):
            plt.semilogy(st_vec, test_tr_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.semilogy(st_vec, test_tr_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=1)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'Relative $L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2))
        fig11.savefig('figs/' + self.fig_name + '_TPIDON_re_error.eps', format='eps')

        plt.show()
        plt.close('all')

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

            np.save(ss, self.all_test_f_evo_mat)
            np.save(ss, f_pred_tensor)
            np.save(ss, f_tr_pred_tensor)




    def save(self, model_name):
        B_model_name = model_name + '_B.h5'
        H_model_name = model_name + '_H.h5'
        T_rho_model_name = model_name + '_T_rho.h5'
        T_g_model_name = model_name + '_T_g.h5'
        self.Bnn.save(B_model_name)
        self.Hnn.save(H_model_name)
        self.T_rhonn.save(T_rho_model_name)
        self.T_gnn.save(T_g_model_name)


if __name__ == "__main__":


    N = 10000
    Nte = 30
    p = 60
    q = 150
    st = 1000
    l1 = 1
    l2 = 1

    eps = 0.0001

    # parameter
    nl = 6
    nr = 100
    bs = 20
    ins = 1

    Ns_load = 500
    Ns = Ns_load
    Nte_load = 30

    dt = 0.01
    st_load = 1000

    # define mesh size for x, y
    lx, ly = 1, 1
    Nq = 24
    Nx, Ny = Nq, Nq
    dx, dy = lx / (Nx + 1), ly / (Ny + 1)

    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]
    points_y = np.linspace(dy, ly - dy, Ny).T
    y = points_y[:, None]

    # define mesh size for v

    Nv = 16

    points_v, weights = np.polynomial.legendre.leggauss(Nv)
    points_v = (points_v + 1) * np.pi
    weights = weights * np.pi
    v, w = np.float32(points_v[:, None]), np.float32(weights[:, None])

    Nx_r, Ny_r = 10, 10


    w_bc_flip = Nx**2*Nv

    ######### prepare NN #############################
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
        Nv) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st_load) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(Nte_load) + '_l1_' + num2str_deciaml(l1) + '_l2_' + num2str_deciaml(l2) + '_eps_' + num2str_deciaml(eps)
    npy_name = filename + '.npy'

    with open(npy_name, 'rb') as ss:
        Train_B_rho = np.load(ss)
        Train_B_g = np.load(ss)
        Train_B_f = np.load(ss)

        Test_B_rho_load = np.load(ss)
        Test_B_g_load = np.load(ss)
        Test_B_f_load = np.load(ss)

        all_test_rho_load = np.load(ss)
        all_test_g_load = np.load(ss)
        all_test_f_load = np.load(ss)

        all_test_rho_evo_mat_load = np.load(ss)
        all_test_g_evo_mat_load = np.load(ss)
        all_test_f_evo_mat_load = np.load(ss)

    id_all = np.array(random.sample(range(Ns * Nx * Ny), N))

    Test_B_rho = Test_B_rho_load[:Nte, :]
    Test_B_g = Test_B_g_load[:Nte, :]
    Test_B_f = Test_B_f_load[:Nte, :]

    all_test_rho = all_test_rho_load[:Nte, :]
    all_test_g = all_test_g_load[:Nte, :]
    all_test_f = all_test_f_load[:Nte, :]

    all_test_rho_evo_mat = all_test_rho_evo_mat_load[:Nte, :st, :]
    all_test_g_evo_mat = all_test_g_evo_mat_load[:Nte, :st, :]
    all_test_f_evo_mat = all_test_f_evo_mat_load[:Nte, :st, :]

    tmp_id = 0

    rho_test = all_test_rho[[tmp_id], :]
    g_test = all_test_g[[tmp_id], :]
    f_test = all_test_f[[tmp_id], :].reshape(Nx, Ny, Nv)

    rho_test_evo = all_test_rho_evo_mat[tmp_id, :st, :]
    g_test_evo = all_test_g_evo_mat[tmp_id, :st, :]
    f_test_evo = all_test_f_evo_mat[tmp_id, :st, :]

    x_train = np.kron(x, np.ones((Ny * Nv, 1)))
    y_train = np.tile(np.kron(y, np.ones((Nv, 1))), (Nx, 1))
    v_train = np.tile(v, (Nx * Ny, 1))

    xx, yy = np.meshgrid(x, y)

    xxv, vv = np.meshgrid(x, v)


    dtype = tf.float32
    num_ad_epochs = 200001
    # define adam optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3/2,
        decay_steps=5000,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    W_f = np.ones((q, 1))
    train_evo_step = 0

    filename = 'C_TPIDON_rte_dt_2d' + '_Nx_' + num2str_deciaml(
        Nx) + '_Nv_' + num2str_deciaml(Nv) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_batchN_' + num2str_deciaml(
        N) + '_Ns_' + num2str_deciaml(Ns) + '_Nte_' + num2str_deciaml(
        Nte) + '_N_' + num2str_deciaml(N) + '_p_' + num2str_deciaml(p) + '_q_' + num2str_deciaml(q) + '_l1_' + num2str_deciaml(l1) + '_l2_' + num2str_deciaml(l2) + '_eps_' + num2str_deciaml(eps)

    fig_name = filename
    file_name = filename + '.txt'
    npy_name = filename + '.npy'

    mdl = rte(eps, Nx, Ny, Nx_r, Ny_r, Nv, Ns, Nte, N, dt, lx, x, dx, ly, y, dy, v, w, Train_B_rho, Train_B_g, Train_B_f, id_all,
              x_train, y_train, v_train, Test_B_rho, Test_B_g, Test_B_f, rho_test, g_test, rho_test_evo, g_test_evo, all_test_rho, all_test_g, all_test_f, all_test_rho_evo_mat, all_test_f_evo_mat,
              f_test_evo, dtype, optimizer, num_ad_epochs, file_name, fig_name, npy_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st, w_bc_flip)
    mdl.fit()

    model_name = filename
    mdl.save('mdls/' + model_name)

