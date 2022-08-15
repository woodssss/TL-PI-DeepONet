# TL-PI-DeepONet
In this project, we improved the physics-informed DeepONets via the one shot transfer learning. The numerical examples we consider 
includes: 1d nonlinear reaction diffusion equation; 1d,2d allen cahn equation; 1d,2d cahn-hilliard equation; navier stokes equation; and 1d,2d linear transfer equation.

# Requirements
Find requirements in requirements.txt. All the codes are ready to run (07/29/2022) on google colab without need of specifying packages version, except for the the ns_sample.py, which requires a older version of torch==1.6.0.

# Usage of code
The usage of code is basically the same for each example, here we only present the details of implementing the Nonlinear reaction diffusion equation. 
## Nonlinear reaction diffusion equation
Consider 
```
f_t = d f_xx + k f^2,
```
with zero BC
```
f(t,0)=f(t,1)=0.
```
### Data generation
```
python sample_nrd_1d_evo.py k d l
````
```
python sample_nrd_1d_nevo.py k d l
````
k,d are the parameters within nrd equation and l is the parameter within covariance kernel. This code generates 10000 training functions and 30 test functions. One may run it on colab as well
```
%run sample_nrd_1d.py k d l
````
### Model training and prediction 
#### Discretized time setting
```
python nrd_1d_dt_deeponet.py N Nte p q st d k l ads
```
N is the total number of training points, Nte is the total number of test functions, p is number of output feature of deeponet, q is length of last hidden layer, st is number of step and ads is max iteration number in training. See more detail in our paper.

#### Continuous time setting
```
python nrd_1d_cont_deeponet_cont.py N Nte p q ct d k l ads
```
