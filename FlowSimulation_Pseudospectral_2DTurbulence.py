#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:45:29 2018

@author: jiahan
"""

import scipy as sc
import matplotlib.pyplot as plt
import IncompNST_Tools.Pseudospectral as nst
from scipy.fftpack import fft2, ifft2

# Physical constants
nu = 0

# Length of field
lx = 1
ly = 1

# Number of grid points
ny = 512
nx = 512

# Grid increments
dx = lx/nx
dy = ly/ny

# Initialize vortex
omega, p = nst.vortices_64(nx, ny, dx, dy)
#omega, p = nst.vortex_pair(nx, ny, dx, dy)
#omega, p = nst.dancing_vortices(nx, ny, dx, dy)

# Gradient operators in Fourier domain for x- and y-direction
Kx, Ky = nst.Spectral_Gradient(nx, ny, lx, ly)

# 2D Laplace operator and 2D inverse Laplace operator in Fourier domain
K2, K2inv = nst.Spectral_Laplace(nx, ny, Kx, Ky)

# Simulation time
T_simu = 10000

# Set discrete time step by choosing CFL number (condition: CFL <= 1)
CFL = 0.1
u = sc.real(ifft2(-Ky*K2inv*fft2(omega)))
v = sc.real(ifft2(Kx*K2inv*fft2(omega)))
u_max = sc.amax(sc.absolute(u))
v_max = sc.amax(sc.absolute(v))
t_step = (CFL*dx*dy)/(u_max*dy+v_max*dx)

# Start Simulation
t_sum = 0
i = 0
#omega_old = sc.zeros([nx,ny])
while t_sum <= T_simu:
    # Runge-Kitta 4 time simulation
    omega = nst.RK4(t_step, omega, Kx, Ky, K2, K2inv, nu)
    
#    print('SUM:', sum(sum(abs(omega-omega_old))))
    
#    omega_old = omega


    # Plot every 100th frame
    if 0 == i % 10000:

        plt.imshow(omega)
        plt.pause(0.1)
        plt.imsave('frame'+str(i)+'.png', omega)

    i += 1
    t_sum += t_step
    
    if 0 == i % 1000:
        print(i, t_sum)
