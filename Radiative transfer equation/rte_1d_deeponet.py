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
import time
import random

class rte():
    # initialize
    def __init__(self, eps, Nx, Nv, Ns, Nte, N, dt, lx, x, dx, lv, v, w, Train_B_rho, Train_B_g, Train_B_f, id_all,
              x_train, v_train, Test_B_rho, Test_B_g, Test_B_f, rho_test, g_test,
              rho_test_evo, g_test_evo, f_test_evo, all_test_rho, all_test_g, all_test_f, all_test_rho_evo_mat, all_test_g_evo_mat, all_test_f_evo_mat,
              xx, vv, dtype, optimizer, num_ad_epochs, file_name, npy_name, fig_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st):

        self.dtype = dtype

        self.eps=eps

        self.nl, self.nr = nl, nr

        self.Nx, self.Nv, self.Ns = Nx, Nv, Ns
        self.Nte, self.N = Nte, N
        self.lx, self.lv = lx, lv

        self.id_all = id_all
        self.id_test = np.linspace(0, Nte * Nx - 1, Nte * Nx).astype(int)

        self.dt, self.dx = dt, dx

        self.x = tf.convert_to_tensor(x, dtype=self.dtype)

        self.v, self.w = tf.convert_to_tensor(v, dtype=self.dtype), tf.convert_to_tensor(w, dtype=self.dtype)

        self.Train_B_rho_tensor = Train_B_rho.reshape(Ns, Nx)
        self.Train_B_g_tensor = Train_B_g.reshape(Ns, Nx, Nv)

        B_rho_tensor = Train_B_rho.reshape(Ns, Nx)
        B_g_tensor = Train_B_g.reshape(Ns, Nx, Nv)

        self.Test_B_rho_tensor = Test_B_rho.reshape(Nte, Nx)
        self.Test_B_g_tensor = Test_B_g.reshape(Nte, Nx, Nv)


        self.x_train = tf.convert_to_tensor(x_train, dtype=self.dtype)
        self.v_train = tf.convert_to_tensor(v_train, dtype=self.dtype)

        self.all_test_rho, self.all_test_g, self.all_test_rho_evo_mat, self.all_test_g_evo_mat = all_test_rho, all_test_g, all_test_rho_evo_mat, all_test_g_evo_mat
        self.all_test_f_evo_mat = all_test_f_evo_mat


        self.xxx, self.vvx = xx, vv

        self.xx, self.vv = np.meshgrid(x, v)

        self.v_mat = tf.convert_to_tensor(np.diag(v_train.T[0]), dtype=self.dtype)
        # print(self.v_mat)
        # zxc

        # test sample

        self.rho_test, self.g_test = rho_test, g_test
        self.g_test_evo, self.rho_test_evo = g_test_evo, rho_test_evo

        self.file_name = file_name
        self.fig_name = fig_name
        self.npy_name = npy_name

        self.num_ad_epochs = num_ad_epochs

        self.optimizer = optimizer

        self.H_input_size = self.Nx * self.Nv

        self.p = p

        self.q = q

        self.W_f = W_f

        self.T_output_size = self.p

        # define parameters for stochastic GD
        self.batch_size = bs  # number of sample at each batch
        self.inner_step = ins  # inner adam step for small batch
        self.inner_bfgs_step = 1  # inner BFGS step for small batch
        self.train_evo_step = train_evo_step

        self.st = st

        self.stop = 0.0000001

        self.epoch_vec = []
        self.emp_loss_vec = []
        self.test_loss_vec = []

        # Initialize NN
        # Initialize NN
        self.Bnn = self.get_B()
        self.Hnn = self.get_H()
        self.Tnn = self.get_T()

        #self.Bnn.summary()

    def prepare_AVG(self, bs):
        AVG_red = np.kron(np.eye(bs), w.T)
        AVG = np.kron(np.eye(bs), np.tile(w.T, (self.Nv, 1)))

        ext_mat = np.kron(np.eye(bs), np.ones((self.Nv, 1)))

        AVG_red, AVG = tf.convert_to_tensor(AVG_red, dtype=self.dtype), tf.convert_to_tensor(AVG, dtype=self.dtype)

        ext_mat = tf.convert_to_tensor(ext_mat, dtype=self.dtype)

        return AVG_red, AVG, ext_mat

    def get_B(self):

        input_Branch_rho = tf.keras.Input(shape=(self.Nx,))
        input_Branch_g = tf.keras.Input(shape=(self.Nx*self.Nv,))

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

        out_h = tf.keras.layers.Reshape((self.p * self.q, 1))(output_Branch)

        out = tf.keras.layers.Conv1D(filters=1, kernel_size=self.q, strides=self.q, input_shape=(None, self.q * self.p),
                                     activation=None, use_bias=False, kernel_initializer='glorot_normal')(out_h)

        model = tf.keras.Model(inputs=[input_Branch_rho, input_Branch_g], outputs=out)

        return model

    def get_H(self):
        input_Branch_rho = tf.keras.Input(shape=(self.Nx,))
        input_Branch_g = tf.keras.Input(shape=(self.Nx * self.Nv,))

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

    def get_T(self):
        # define nn for Trunk

        ### For rho

        input_Trunk_rho = tf.keras.Input(shape=(1,))

        def pd_feature_rho(ip):
            x = ip[:, 0]


            # basis
            b1 = tf.cos(2 * np.pi * x)
            b2 = tf.sin(2 * np.pi * x)

            # b6 = tf.cos(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b7 = tf.cos(2 * np.pi * x) * tf.sin(2 * np.pi * y)
            # b8 = tf.sin(2 * np.pi * x) * tf.cos(2 * np.pi * y)
            # b9 = tf.sin(2 * np.pi * x) * tf.sin(2 * np.pi * y)

            # out = tf.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9], axis=1)
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

        output_Trunk_rho = tf.keras.layers.Dense(units=self.T_output_size, activation=tf.nn.tanh,
                                             kernel_initializer='glorot_normal')(
            input_Trunk_rho_mid)

        ### For g
        input_Trunk_g = tf.keras.Input(shape=(2,))

        def pd_feature_g(ip):
            x = ip[:, 0]
            v = ip[:, 1]
            # basis
            b1 = tf.cos(2 * np.pi * x)
            b2 = tf.sin(2 * np.pi * x)

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

        model = tf.keras.Model(inputs=[input_Trunk_rho, input_Trunk_g], outputs=[output_Trunk_rho, output_Trunk_g])

        return model


    ################ define loss function ####################################################

    def get_loss(self, tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g):
        bs = tmp_B_rho.shape[0]

        tmp_T_x = tmp_x

        tmp_T_x_vec = np.kron(tmp_T_x.numpy(), np.ones((self.Nv, 1)))
        tmp_T_v_vec = np.tile(v, (bs, 1))

        tmp_T_x_vec = tf.convert_to_tensor(tmp_T_x_vec, dtype=self.dtype)
        tmp_T_v_vec = tf.convert_to_tensor(tmp_T_v_vec, dtype=self.dtype)

        ext_B = tf.convert_to_tensor(np.kron(np.eye(bs), np.ones((self.Nv, 1))), dtype=self.dtype)


        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tmp_T_x)
            tape.watch(tmp_T_x_vec)

            Train_T = tf.concat([tmp_T_x_vec, tmp_T_v_vec], axis=1)

            B_ori = self.Bnn([tmp_B_rho, tmp_B_g])[:, :, 0]

            B = tf.matmul(ext_B, B_ori)

            T_rhog = self.Tnn([tmp_T_x, Train_T])

            T_rho = T_rhog[0]
            T_g = T_rhog[1]

            # print(T_rho.shape, T_g.shape)
            # zxc

            rho = tf.reduce_sum(B_ori * T_rho, axis=1, keepdims=True)
            g = tf.reduce_sum(B * T_g, axis=1, keepdims=True)

            rho_x = tape.gradient(rho, tmp_T_x)
            g_x = tape.gradient(g, tmp_T_x_vec)

        AVG_red, AVG, ext_mat = self.prepare_AVG(bs) # time 0.5 afterward

        rho_old = tf.reshape(tmp_IC_rho, [bs, 1])
        g_old = tf.reshape(tmp_IC_g, [bs * self.Nv, 1])

        # print(rho.shape, g.shape, rho_x.shape, rho_old.shape, AVG_red.shape, (tmp_T_v_vec*g_x).shape)
        # zxc

        # pde1
        # rho_t + \partial_x <vg> = 0

        pde1 = rho-rho_old + self.dt/2*tf.linalg.matmul(AVG_red, tmp_T_v_vec*g_x)


        # pde2
        # eps^2 g_t
        rho_x_vec = tf.matmul(ext_B, rho_x)

        pde2 = self.eps**2/self.dt*(g-g_old) + self.eps*tmp_T_v_vec*g_x-self.eps*tf.linalg.matmul(AVG, tmp_T_v_vec*g_x)/2 + tmp_T_v_vec*rho_x_vec + g


        loss = tf.reduce_mean(tf.square(pde1)) + tf.reduce_mean(tf.square(pde2))


        return loss

    def get_total_loss(self, id_all, N, Nl, B_rho, B_g, B_rho_tensor, B_g_tensor):

        # Nl is the small batch size for total loss

        part_num = N // Nl

        total_loss = 0

        for i in range(part_num):

            id_bs = id_all[i*Nl:(i+1)*Nl]

            id_k_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nx)

            tmp_B_rho = tf.convert_to_tensor(B_rho[id_k_vec, :], dtype=self.dtype)

            tmp_B_g = tf.convert_to_tensor(B_g[id_k_vec, :], dtype=self.dtype)

            tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

            tmp_IC_rho = tf.convert_to_tensor(B_rho_tensor[id_k_vec, id_i_vec], dtype=self.dtype)

            tmp_IC_g = tf.convert_to_tensor(B_g_tensor[id_k_vec, id_i_vec, :], dtype=self.dtype)

            tmp_loss = self.get_loss(tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g)

            total_loss = total_loss + tmp_loss

        return total_loss/part_num

    def get_grad(self, tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g):
        with tf.GradientTape() as tape:
            loss = self.get_loss(tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g)
            trainable_weights_B= self.Bnn.trainable_variables
            trainable_weights_T = self.Tnn.trainable_variables


            trainable_weights = trainable_weights_B + trainable_weights_T

        grad = tape.gradient(loss, trainable_weights)

        return loss, trainable_weights, grad

    def id_2_id_ki(self, id, Nx):
        id_k = id // (Nx)
        id_i = (id - id_k * Nx)

        return id_k, id_i

    def get_id_bs_vec(self, id, Nx):
        bs = id.shape[0]

        id_k_vec = np.zeros_like(id)
        id_i_vec = np.zeros_like(id)

        for j in range(bs):
            id_k_vec[j], id_i_vec[j] = self.id_2_id_ki(id[j], Nx)

        return id_k_vec, id_i_vec



    def get_random_sample(self, id_all):

        id_bs = np.random.choice(id_all, self.batch_size)

        id_k_vec, id_i_vec = self.get_id_bs_vec(id_bs, self.Nx)

        tmp_B_rho = tf.convert_to_tensor(Train_B_rho[id_k_vec, :], dtype=self.dtype)

        tmp_B_g = tf.convert_to_tensor(Train_B_g[id_k_vec, :], dtype=self.dtype)

        tmp_x = tf.convert_to_tensor(x[id_i_vec, :], dtype=self.dtype)

        tmp_IC_rho = tf.convert_to_tensor(self.Train_B_rho_tensor[id_k_vec, id_i_vec], dtype=self.dtype)

        tmp_IC_g = tf.convert_to_tensor(self.Train_B_g_tensor[id_k_vec, id_i_vec, :], dtype=self.dtype)

        return tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g



    # Adam step
    def adam_step(self):
        tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g = self.get_random_sample(self.id_all)
        for i in range(self.inner_step):
            #with tf.device('/device:GPU:0'):
            with tf.device('/device:CPU:0'):
                loss, trainable_weights, grad = self.get_grad(tmp_B_rho, tmp_B_g, tmp_x, tmp_IC_rho, tmp_IC_g)
                self.optimizer.apply_gradients(zip(grad, trainable_weights))

    def fit(self):
        start_time = time.time()
        for epoch in range(self.num_ad_epochs):
            self.adam_step()
            if epoch % 10000 == 0:
                elapsed = time.time() - start_time
                start_time = time.time()
                total_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B_rho, Train_B_g, self.Train_B_rho_tensor, self.Train_B_g_tensor)
                test_loss = self.get_total_loss(self.id_test, self.Nte * self.Nx, 100, Test_B_rho, Test_B_g, self.Test_B_rho_tensor, self.Test_B_g_tensor)
                print('Nx: %d, N: %d, Nte: %d, dt: %.3f, st: %d, p: %d, q: %d' % (
                    self.Nx, self.N, self.Nte, self.dt, self.st, self.p, self.q))
                print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f' %
                      (epoch, total_loss, test_loss, elapsed))
                with open(self.file_name, 'a') as fw:
                    print('Nx: %d, N: %d, Nte: %d, dt: %.3f, st: %d, p: %d, q: %d' % (
                        self.Nx, self.N, self.Nte, self.dt, self.st, self.p, self.q), file=fw)
                    print('Epoch: %d, Empirical Loss: %.3e, Test Loss: %.3e, Time: %.2f' %
                          (epoch, total_loss, test_loss, elapsed), file=fw)

                self.epoch_vec.append(epoch)
                self.emp_loss_vec.append(total_loss)
                self.test_loss_vec.append(test_loss)

                # assign weight to H
                weights_B = self.Bnn.get_weights()
                weights_H = weights_B[:-1]
                self.Hnn.set_weights(weights_H)

                if total_loss < self.stop:
                    print('Adam training finished')
                    with open(self.file_name, 'a') as fw:
                        print('Adam training finished', file=fw)
                    break

            if epoch % (self.num_ad_epochs - 1) == 0 and epoch > 0:
            #if epoch % 2000 == 0 and epoch > 0:
                self.check_with_visual()

        final_loss = self.get_total_loss(self.id_all, self.N, 1000, Train_B_rho, Train_B_g, self.Train_B_rho_tensor, self.Train_B_g_tensor)
        print('Final loss is %.3e' % final_loss)
        with open(self.file_name, 'a') as fw:
            print('Final loss is %.3e' % final_loss, file=fw)


    def get_evo_pred(self, rho, g):

        Train_T = tf.concat([self.x_train, self.v_train], axis=1)
        T_rhog = self.Tnn([self.x, Train_T])

        T_rho = T_rhog[0]
        T_g = T_rhog[1]

        rho_evo = np.zeros((self.st, self.Nx))
        g_evo = np.zeros((self.st, self.Nx * self.Nv))

        for i in range(self.st):

            B = self.Bnn([rho, g])[:, :, 0]

            rho = tf.reduce_sum(B * T_rho, 1, keepdims=True).numpy().T
            g = tf.reduce_sum(B * T_g, 1, keepdims=True).numpy().T

            rho_evo[i, :] = rho
            g_evo[i, :] = g

        return rho_evo, g_evo


    def get_Tevo_mat(self):

        # T
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x)
            tape.watch(self.x_train)

            Train_T = tf.concat([self.x_train, self.v_train], axis=1)

            T_rhog = self.Tnn([self.x, Train_T])

            T_rho = T_rhog[0]
            T_g = T_rhog[1]

            T_rho_x_list = []
            T_g_x_list = []

            for i in range(self.p):

                T_rho_tmp = T_rho[:, i][:, None]
                T_g_tmp = T_g[:, i][:, None]

                T_rho_x_tmp = tape.gradient(T_rho_tmp, self.x)
                T_g_x_tmp = tape.gradient(T_g_tmp, self.x_train)

                T_rho_x_list.append(T_rho_x_tmp)
                T_g_x_list.append(T_g_x_tmp)

        T_rho_x = tf.concat(T_rho_x_list, axis=1)
        T_g_x = tf.concat(T_g_x_list, axis=1)

        return T_rho, T_g, T_rho_x, T_g_x


    def get_Tevo_pred(self, rho, g, T_rho, T_g, T_rho_x, T_g_x):

        # prepare matrix
        AVG_red, AVG, ext_mat = self.prepare_AVG(self.Nx)  # time 0.5 afterward

        AVG_red, AVG, ext_mat = AVG_red.numpy(), AVG.numpy(), ext_mat.numpy()


        # pde
        # rho_t + \partial_x <vg> = 0

        rho_evo_mat = np.zeros((self.st, self.Nx))
        g_evo_mat = np.zeros((self.st, self.Nx * self.Nv))
        Ext = np.kron(np.eye(Nx), np.ones((Nv, 1)))

        L_rho = T_rho + self.dt/2*np.matmul(AVG_red, np.matmul(self.v_mat, T_g_x))
        L_g = (1+self.eps**2/self.dt)*T_g + self.eps*np.matmul(self.v_mat, T_g_x) - self.eps/2*np.matmul(AVG, np.matmul(self.v_mat, T_g_x)) + np.matmul(self.v_mat, np.matmul(Ext, T_rho_x))


        for i in range(self.st):

            H = self.Hnn([rho, g])[0, :, :]

            K_rho = np.matmul(L_rho, H)
            K_g = np.matmul(L_g, H)

            K = np.concatenate([K_rho, K_g], axis=0)

            RHS = np.concatenate([rho.T, self.eps**2/self.dt*g.T], axis=0)

            L = np.matmul(K.T, K)

            R = np.matmul(K.T, RHS)

            W_tmp = np.linalg.lstsq(L, R, 0.000000001)[0]

            rho = (np.matmul(np.matmul(T_rho, H), W_tmp)).T

            g = (np.matmul(np.matmul(T_g, H), W_tmp)).T

            rho_evo_mat[i, :] = rho
            g_evo_mat[i, :] = g

        return rho_evo_mat, g_evo_mat



    def f2rho(self, f):
        f_mat = f.reshape(self.Nx, self.Nv)

        rho_tmp = np.zeros((self.Nx, 1))

        for i in range(self.Nx):
            rho_tmp[i, 0] = np.sum(f_mat[[i], :] * w.T) / 2

        return rho_tmp

    def f_mat2rho(self, f_mat):
        rho_tmp = np.zeros((self.Nx, 1))

        for i in range(self.Nx):
            rho_tmp[i, 0] = np.sum(f_mat[[i], :] * w.T) / 2

        return rho_tmp


    def get_re_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            f1_mat = f1[i, :].reshape(self.Nx, self.Nv)
            f2_mat = f2[i, :].reshape(self.Nx, self.Nv)
            e_mat = np.square(f1_mat - f2_mat)
            error_tmp = 0
            f2_mat_sq = np.square(f2_mat)
            f2_mass = 0
            for j in range(Nx):
                error_tmp = error_tmp + np.sum(e_mat[[j], :] * w.T) * self.dx
                f2_mass = f2_mass + np.sum(f2_mat_sq[[j], :] * w.T) * self.dx

            error[0, i] = np.sqrt(error_tmp)/np.sqrt(f2_mass)
        return error

    def get_l2_error_f_evo(self, f1, f2):
        error = np.zeros((1, self.st))
        for i in range(self.st):
            f1_mat = f1[i, :].reshape(self.Nx, self.Nv)
            f2_mat = f2[i, :].reshape(self.Nx, self.Nv)
            e_mat = np.square(f1_mat - f2_mat)
            error_tmp = 0
            for j in range(Nx):
                error_tmp = error_tmp + np.sum(e_mat[[j], :] * w.T) * self.dx

            error[0, i] = np.sqrt(error_tmp)

        return error

    def all_test_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx * self.Nv))

        p_start_time = time.time()

        for i in range(self.Nte):
            # print(self.all_test_f.shape, i)
            tmp_rho = self.all_test_rho[[i], :]
            tmp_g = self.all_test_g[[i], :]

            rho_tmp_pred, g_tmp_pred = self.get_evo_pred(tmp_rho, tmp_g)

            rho_vec_tmp_pred = np.kron(rho_tmp_pred, np.ones((1, Nv)))
            f_tmp_pred = rho_vec_tmp_pred + self.eps*g_tmp_pred

            f_tmp_ref = self.all_test_f_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_f_evo_mat)) / np.sum(
            np.square(self.all_test_f_evo_mat)))

        return error_mat, re_error_mat, re_error_T, pred_time

    def all_test_tr_result(self):

        error_mat, re_error_mat = np.zeros((self.Nte, self.st)), np.zeros((self.Nte, self.st))

        f_pred_tensor = np.zeros((self.Nte, self.st, self.Nx * self.Nv))

        T_rho, T_g, T_rho_x, T_g_x = self.get_Tevo_mat()
        T_rho, T_g, T_rho_x, T_g_x = T_rho.numpy(), T_g.numpy(), T_rho_x.numpy(), T_g_x.numpy()

        p_start_time = time.time()

        for i in range(self.Nte):
            tmp_rho = self.all_test_rho[[i], :]
            tmp_g = self.all_test_g[[i], :]

            rho_tmp_pred, g_tmp_pred = self.get_Tevo_pred(tmp_rho, tmp_g, T_rho, T_g, T_rho_x, T_g_x)

            rho_vec_tmp_pred = np.kron(rho_tmp_pred, np.ones((1, Nv)))
            f_tmp_pred = rho_vec_tmp_pred + self.eps * g_tmp_pred

            f_tmp_ref = self.all_test_f_evo_mat[i, :, :]

            f_pred_tensor[i, :, :] = f_tmp_pred

            error_mat[i, :] = self.get_l2_error_f_evo(f_tmp_pred, f_tmp_ref)

            re_error_mat[i, :] = self.get_re_error_f_evo(f_tmp_pred, f_tmp_ref)

        pred_time = time.time() - p_start_time

        re_error_T = np.sqrt(np.sum(np.square(f_pred_tensor - self.all_test_f_evo_mat)) / np.sum(
            np.square(self.all_test_f_evo_mat)))

        return error_mat, re_error_mat, re_error_T, pred_time

    # define check function
    def check_with_visual(self):

        # assign weight to H
        weights_B = self.Bnn.get_weights()
        weights_H = weights_B[:-1]
        self.Hnn.set_weights(weights_H)

        ###### by PIDON
        rho_evo_mat, g_evo_mat = self.get_evo_pred(self.rho_test, self.g_test)

        test_evo_error, test_evo_re_error, test_evo_re_error_T, pred_time = self.all_test_result()

        ###### by TPIDON
        T_rho, T_g, T_rho_x, T_g_x = self.get_Tevo_mat()
        T_rho, T_g, T_rho_x, T_g_x = T_rho.numpy(), T_g.numpy(), T_rho_x.numpy(), T_g_x.numpy()


        rho_tr_evo_mat, g_tr_evo_mat = self.get_Tevo_pred(self.rho_test, self.g_test, T_rho, T_g, T_rho_x, T_g_x)
        #rho_tr_evo_mat, f_tr_evo_mat = self.get_Tevo_pred_nonl(self.f_test, T, T_x)


        test_tr_evo_error, test_tr_evo_re_error, test_tr_evo_re_error_T, pred_tr_time = self.all_test_tr_result()

        test_evo_error_avg, test_tr_evo_error_avg = np.sum(test_evo_error, axis=0) / self.Nte, np.sum(test_tr_evo_error,
                                                                                                      axis=0) / self.Nte
        test_evo_re_error_avg, test_tr_evo_re_error_avg = np.sum(test_evo_re_error, axis=0) / self.Nte, np.sum(
            test_tr_evo_re_error, axis=0) / self.Nte

        st_vec = np.linspace(1, self.st, self.st).T
        st_vec = st_vec[:, None]

        t_c, x_c = np.meshgrid(st_vec, x)

        abs_rho_error_in_time_mat = np.abs(rho_evo_mat - self.rho_test_evo)
        abs_rho_tr_error_in_time_mat = np.abs(rho_tr_evo_mat - self.rho_test_evo)


        lst = 4

        plt.figure(1)
        plt.plot(self.x, self.rho_test_evo[0, :], 'r-*')
        for i in range(int(self.st/lst)):
            plt.plot(self.x, self.rho_test_evo[i*lst, :], 'b--')
            plt.plot(self.x, rho_evo_mat[i*lst, :], 'c')

        plt.title('sample rho evo ref')

        plt.figure(2)
        plt.plot(self.x, self.rho_test_evo[0, :], 'r-*')
        for i in range(int(self.st/lst)):
            plt.plot(self.x, self.rho_test_evo[i*lst, :], 'b--')
            plt.plot(self.x, rho_tr_evo_mat[i*lst, :], 'c')

        plt.title('sample rho transfer evo ref')

        fig3 = plt.figure(3)
        ax = fig3.add_subplot()
        cp = ax.contourf(t_c, x_c, self.rho_test_evo.T)
        fig3.colorbar(cp)
        plt.title('Reference')
        plt.xlabel('t')
        plt.ylabel('x')

        fig4 = plt.figure(4)
        ax = fig4.add_subplot()
        cp1 = ax.contourf(t_c, x_c, abs_rho_error_in_time_mat.T)
        fig4.colorbar(cp1)
        plt.title('DeepONet '+ chr(0x215F + 2) + r' absolute error of $\rho$')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        fig4.savefig('figs/' + self.fig_name + '_PIDON_error_contour.eps', format='eps')

        fig5 = plt.figure(5)
        ax = fig5.add_subplot()
        cp = ax.contourf(t_c, x_c, abs_rho_tr_error_in_time_mat.T)
        fig5.colorbar(cp1)
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' absolute error of $\rho$')
        plt.xlabel('t')
        plt.ylabel('x')
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        fig5.savefig('figs/' + self.fig_name + '_TPIDON_error_contour.eps', format='eps')


        plt.figure(6)
        plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r', label='empirical loss')
        plt.semilogy(self.epoch_vec, self.test_loss_vec, 'b', label='test loss')
        plt.legend()
        plt.title('loss vs iteration')

        fig7 = plt.figure(7)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2) + r' with N=' + str(self.N) + ' q=' + str(self.q))
        fig7.savefig('figs/' + self.fig_name + '_PIDON_error.eps', format='eps')

        fig8 = plt.figure(8)
        for i in range(self.Nte):
            plt.plot(st_vec, test_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'Relative $L^2$ error')
        plt.title('DeepONet ' + chr(0x215F + 2) + ' with N=' + str(self.N) + ' q=' + str(self.q))
        fig8.savefig(self.fig_name + '_PIDON_re_error.eps', format='eps')

        fig9 = plt.figure(9)
        for i in range(self.Nte):
            plt.plot(st_vec, test_tr_evo_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_tr_evo_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'$L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' with N=' + str(self.N) + ' q=' + str(self.q))
        fig9.savefig(self.fig_name + '_TPIDON_error.eps', format='eps')

        fig10 = plt.figure(10)
        for i in range(self.Nte):
            plt.plot(st_vec, test_tr_evo_re_error[i, :], 'b--', linewidth=0.5)

        plt.plot(st_vec, test_tr_evo_re_error_avg, 'r', linewidth=2)
        xticks = np.arange(0, self.st + 1, int(self.st / 5))
        labels = np.round(np.arange(0, (self.st + 1) * self.dt, int(self.st / 5) * self.dt), decimals=2)
        plt.xticks(xticks, labels)
        plt.xlabel('t')
        plt.ylabel(r'Relative $L^2$ error')
        plt.title('TL-DeepONet ' + chr(0x215F + 2) + r' with N=' + str(self.N) + ' q=' + str(self.q))
        fig10.savefig(self.fig_name + '_TPIDON_re_error.eps', format='eps')

        plt.show()

        with open(self.npy_name, 'wb') as ss:
            np.save(ss, test_evo_error)
            np.save(ss, test_evo_error_avg)
            np.save(ss, test_tr_evo_error)
            np.save(ss, test_tr_evo_error_avg)

            np.save(ss, test_evo_re_error)
            np.save(ss, test_evo_re_error_avg)
            np.save(ss, test_tr_evo_re_error)
            np.save(ss, test_tr_evo_re_error_avg)

            # np.save(ss, f_evo_mat)
            # np.save(ss, f_tr_evo_mat)

        print('Avg PIDON test l1 error: %.3e, Avg PIDON test l2 relative error: %.3e' %
              (1 / self.Nte * np.sum(test_evo_error) * self.dt, test_evo_re_error_T))
        print('Avg TPIDON test l1 error: %.3e, Avg TPIDON test l2 relative error: %.3e' %
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
        H_model_name = model_name + '_H.h5'
        T_model_name = model_name + '_T.h5'
        self.Bnn.save(B_model_name)
        self.Hnn.save(H_model_name)
        self.Tnn.save(T_model_name)


if __name__ == "__main__":

    ##################### prepare/load training set ##############################################################
    # define parameter
    nl = 5
    nr = 100
    bs = 10
    ins = 1

    l1 = 0.5
    l2 = 0.5

    eps = 0.0001

    p = 100
    q = 40

    Ns = 50
    Nte_load = 30

    Nte = 30
    N = 1000
    st = 20

    dt = 0.02
    st_load = 20

    # define mesh size for x
    lx = 1
    Nx = 32

    dx = lx / (Nx + 1)
    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]

    # define mesh size for v
    lv = 1
    Nv = 16

    points_v, weights = np.polynomial.legendre.leggauss(Nv)
    points_v = lv * points_v
    weights = lv * weights
    v, w = np.float32(points_v[:, None]), np.float32(weights[:, None])

    xx, vv = np.meshgrid(x, v)

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


    npy_name = 'A_rte_sample_1d_pb_evo_dp' + '_Nx_' + num2str_deciaml(Nx) + '_Nv_' + num2str_deciaml(
        Nv) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(
        Ns) + '_Nte_' + num2str_deciaml(Nte) + '_l1_' + num2str_deciaml(l1) + '_l2_' + num2str_deciaml(l2) + '_eps_' + num2str_deciaml(eps) + '.npy'

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



    id_all = np.array(random.sample(range(Ns * Nx), N))

    Test_B_rho = Test_B_rho_load[:Nte, :]
    Test_B_g = Test_B_g_load[:Nte, :]
    Test_B_f = Test_B_f_load[:Nte, :]

    all_test_rho = all_test_rho_load[:Nte, :]
    all_test_g = all_test_g_load[:Nte, :]
    all_test_f = all_test_f_load[:Nte, :]

    all_test_rho_evo_mat = all_test_rho_evo_mat_load[:Nte, :st, :]
    all_test_g_evo_mat = all_test_g_evo_mat_load[:Nte, :st, :]
    all_test_f_evo_mat = all_test_f_evo_mat_load[:Nte, :st, :]

    rho_test = all_test_rho[[0], :]
    g_test = all_test_g[[0], :]

    rho_test_evo = all_test_rho_evo_mat[0, :st, :]
    g_test_evo = all_test_g_evo_mat[0, :st, :]
    f_test_evo = all_test_f_evo_mat[0, :st, :]

    x_train = np.kron(x, np.ones((Nv, 1)))
    v_train = np.tile(v, (Nx, 1))


    dtype = tf.float32
    num_ad_epochs = 300001
    # define adam optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=3000,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=1e-3/2,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-08)


    W_f = np.ones((q, 1))
    train_evo_step = 0

    filename = 'C_rte_1d_dt_' + '_Nx_' + num2str_deciaml(
        Nx) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_batchN_' + num2str_deciaml(
        N) + '_Nte_' + num2str_deciaml(
        Nte) + '_p_' + num2str_deciaml(p) + '_q_' + num2str_deciaml(q)

    fig_name = filename
    npy_name = filename + '.npy'
    file_name = filename + '.txt'

    mdl = rte(eps, Nx, Nv, Ns, Nte, N, dt, lx, x, dx, lv, v, w, Train_B_rho, Train_B_g, Train_B_f, id_all,
              x_train, v_train, Test_B_rho, Test_B_g, Test_B_f, rho_test, g_test,
              rho_test_evo, g_test_evo, f_test_evo, all_test_rho, all_test_g, all_test_f, all_test_rho_evo_mat, all_test_g_evo_mat, all_test_f_evo_mat,
              xx, vv, dtype, optimizer, num_ad_epochs, file_name, npy_name, fig_name, nl, nr, p, q, W_f, bs, ins, train_evo_step, st)
    mdl.fit()

    model_name = filename
    mdl.save('mdls/' + model_name)