################################ Data generation ns   ##################################
# This code is adapted from the FNO's project
# https://github.com/zongyi-li/fourier_neural_operator
# Author: Wuzhe Xu
# Date: 07/27/2022
########################################################################################

import torch
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

# from drawnow import drawnow, figure

from timeit import default_timer

import scipy.io


####################################################################
#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t, record_steps):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.rfft(w0, 2, normalized=False, onesided=False)

    #Forcing to Fourier space
    f_h = torch.rfft(f, 2, normalized=False, onesided=False)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Initial velocity field
    psi_h = w_h.clone()
    psi_h[...,0] = psi_h[...,0]/lap
    psi_h[...,1] = psi_h[...,1]/lap

    #Velocity field in x-direction = psi_y
    q = psi_h.clone()
    temp = q[...,0].clone()
    q[...,0] = -2*math.pi*k_y*q[...,1]
    q[...,1] = 2*math.pi*k_y*temp
    psi_y_0 = torch.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

    #Velocity field in y-direction = -psi_x
    v = psi_h.clone()
    temp = v[...,0].clone()
    v[...,0] = 2*math.pi*k_x*v[...,1]
    v[...,1] = -2*math.pi*k_x*temp
    psi_x_0 = -torch.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N,N))

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_psi_x = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_psi_y = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        #print('total steps=', steps, ' current step=', j)
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        #Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q = torch.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v = torch.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x = torch.irfft(w_x, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y = torch.irfft(w_y, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.rfft(q*w_x + v*w_y, 2, normalized=False, onesided=False)

        #Dealias
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]

        #Cranck-Nicholson update
        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.irfft(w_h, 2, normalized=False, onesided=False, signal_sizes=(N,N))

            psi_h = w_h.clone()
            psi_h[...,0] = psi_h[...,0]/lap
            psi_h[...,1] = psi_h[...,1]/lap

            #Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q[...,0].clone()
            q[...,0] = -2*math.pi*k_y*q[...,1]
            q[...,1] = 2*math.pi*k_y*temp
            q = torch.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

            #Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v[...,0].clone()
            v[...,0] = 2*math.pi*k_x*v[...,1]
            v[...,1] = -2*math.pi*k_x*temp
            v = torch.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N,N))

            #Record solution and time
            sol[...,c] = w
            sol_psi_x[...,c] = -v
            sol_psi_y[...,c] = q
            sol_t[c] = t

            c += 1


    return psi_x_0, psi_y_0, sol, sol_psi_x, sol_psi_y, sol_t


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### parameters
nu = 0.1
m_dt = 1e-4
T_s = 1
T = 10

#Number of snapshots from solution
record_steps_tr = 100
record_steps_te = 1000

dt = 0.01

st_evo = 100
st = 1000

Nb = 50
Nte = 30
Ns = Nb*st_evo

l=2

#Resolution
s = 32
sub = 1

Nx = s

#Number of solutions to generate

#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]
x = t


########### define GRF function
def my_grf(Ns, Nx, lx, l):
    Ny = Nx

    dx = lx / (Nx)
    dy = dx

    points_x = np.linspace(0, lx - dx, Nx).T
    x = points_x[:, None]

    points_y = points_x

    Corv = np.zeros((Nx * Ny, Nx * Ny))

    for i in range(Nx * Ny):
        for j in range(Nx * Ny):
            nx1 = i // Nx
            my1 = i - nx1 * Nx
            nx2 = j // Nx
            my2 = j - nx2 * Nx

            Corv[i, j] = np.exp(-(np.sin(np.pi * (points_x[nx1] - points_x[nx2]) / 1) ** 2 + np.sin(
                np.pi * (points_y[my1] - points_y[my2]) / 1) ** 2) / l ** 2/4)

    g_mat = np.zeros((Ns, Nx * Ny))
    mean = np.zeros((Nx * Ny,))

    for i in range(Ns):
        g_mat[[i], :] = (np.random.multivariate_normal(mean, Corv))

    # zero mean
    for i in range(Ns):
        tmp_mass = np.sum(g_mat[[i], :])/s**2
        g_mat[[i], :] = g_mat[[i], :] - tmp_mass


    f_tensor = g_mat.reshape(Ns, Nx, Nx)

    res = torch.from_numpy(f_tensor)

    return res

X,Y = torch.meshgrid(t, t)
# f = nu*(2*np.pi)**2*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

mx = np.linspace(0, 1, s)
# print(mx.shape)
# zxc
xx, yy = np.meshgrid(mx, mx)

#Batch size
bsize = 10

####################################################
c = 0
t0 =default_timer()
#Inputs
a = torch.zeros(Nb, s, s)
psi_x_ic = torch.zeros(Nb, s, s)
psi_y_ic = torch.zeros(Nb, s, s)
#Solutions
u = torch.zeros(Nb, s, s, record_steps_tr)
psi_x = torch.zeros(Nb, s, s, record_steps_tr)
psi_y = torch.zeros(Nb, s, s, record_steps_tr)
for j in range(Nb//bsize):

    #Sample random feilds
    w0 = my_grf(bsize, Nx, 1, l)

    #Solve NS
    psi_x_0, psi_y_0, sol, sol_psi_x, sol_psi_y, sol_t = navier_stokes_2d(w0, f, nu, T_s, m_dt, record_steps_tr)

    a[c:(c+bsize),...] = w0
    psi_x_ic[c:(c+bsize),...] = psi_x_0
    psi_y_ic[c:(c+bsize),...] = psi_y_0
    u[c:(c+bsize),...] = sol
    psi_x[c:(c+bsize),...] = sol_psi_x
    psi_y[c:(c+bsize),...] = sol_psi_y

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)


my_ic_train = a.cpu().detach().numpy()
my_psi_x_ic_train = psi_x_ic.detach().numpy()
my_psi_y_ic_train = psi_y_ic.detach().numpy()

my_sol_train = u.cpu().detach().numpy()
my_psi_x_train = psi_x.detach().numpy()
my_psi_y_train = psi_y.detach().numpy()

########### check reference by fig #########################

fig = plt.figure(1)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, my_ic_train[0, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(2)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, my_ic_train[1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(3)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, my_ic_train[2, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)


plt.show()

############################################

####################################################
c = 0
t0 =default_timer()
#Inputs
a = torch.zeros(Nte, s, s)
psi_x_ic = torch.zeros(Nte, s, s)
psi_y_ic = torch.zeros(Nte, s, s)
#Solutions
u = torch.zeros(Nte, s, s, record_steps_te)

for j in range(Nte//bsize):

    #Sample random feilds
    w0 = my_grf(bsize, Nx, 1, l)

    #Solve NS
    psi_x_0, psi_y_0, sol, sol_psi_x, sol_psi_y, sol_t = navier_stokes_2d(w0, f, nu, T, m_dt, record_steps_te)

    a[c:(c+bsize),...] = w0
    psi_x_ic[c:(c+bsize),...] = psi_x_0
    psi_y_ic[c:(c+bsize),...] = psi_y_0
    u[c:(c+bsize),...] = sol

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)

my_ic_test = a.cpu().detach().numpy()
my_psi_x_test = psi_x_ic.detach().numpy()
my_psi_y_test = psi_y_ic.detach().numpy()
my_sol_test = u.cpu().detach().numpy()

################# previous code copied from FNO project #################
########## save reference for DON ########################

Train_B = np.zeros((Ns, Nx**2))
Train_psi_x = np.zeros((Ns, Nx**2))
Train_psi_y = np.zeros((Ns, Nx**2))

for i in range(Nb):
    for j in range(st_evo):
        tmp = my_sol_train[i, :, :, j]
        tmp_psi_x = my_psi_x_train[i, :, :, j]
        tmp_psi_y = my_psi_y_train[i, :, :, j]
        Train_B[i*st_evo+j, :] = tmp.reshape(1, Nx**2)
        Train_psi_x[i*st_evo+j, :] = tmp_psi_x.reshape(1, Nx**2)
        Train_psi_y[i*st_evo+j, :] = tmp_psi_y.reshape(1, Nx**2)

all_test_f = my_ic_test.reshape(Nte, Nx**2)
all_test_psi_x = my_psi_x_test.reshape(Nte, Nx**2)
all_test_psi_y = my_psi_y_test.reshape(Nte, Nx**2)

all_test_evo_mat = np.zeros((Nte, st, Nx**2))

for i in range(Nte):
    for j in range(st):
        all_test_evo_mat[i,j, :] = my_sol_test[i, :, :, j].reshape(1, Nx**2)

Test_B = all_test_f


########### check reference by fig #########################
fig = plt.figure(1)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, Train_B[st_evo-1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(2)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, Train_B[2*st_evo-1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(3)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, Train_B[-1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(4)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, all_test_evo_mat[0, st - 1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(5)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, all_test_evo_mat[2, st - 1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

fig = plt.figure(6)
ax = fig.add_subplot()
cp = ax.contourf(xx, yy, all_test_evo_mat[-1, st-1, :].reshape(Nx, Nx).T, cmap=cm.jet)
fig.colorbar(cp)

plt.show()

############################################




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
    s) + '_l_' + num2str_deciaml(l) + '_dt_' + num2str_deciaml(dt) + '_st_' + num2str_deciaml(st) + '_Ns_' + num2str_deciaml(Ns) + '_Nte_' + num2str_deciaml(
    Nte)
npy_name = filename + '.npy'

with open(npy_name, 'wb') as ss:
    np.save(ss, x)
    np.save(ss, Train_B)
    np.save(ss, Train_psi_x)
    np.save(ss, Train_psi_y)

    np.save(ss, Test_B)
    np.save(ss, all_test_f)
    np.save(ss, all_test_psi_x)
    np.save(ss, all_test_psi_y)
    np.save(ss, all_test_evo_mat)