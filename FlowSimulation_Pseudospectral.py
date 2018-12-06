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
nu = 0.001

# Length of field
lx = 2*sc.pi
ly = 2*sc.pi

# Number of grid points
nx = 256
ny = 256

# Grid increments
dx = lx/nx
dy = ly/ny

# Initialize vortex: choose one of the following:
# 1. vortex pair in the middle of the domain
omega, p = nst.vortex_pair(nx, ny, dx, dy)

# 2. three vortices dancing moving arround
#omega, p = nst.dancing_vortices(nx, ny, dx, dy)

# Gradient operators in Fourier domain for x- and y-direction
Kx, Ky = nst.Spectral_Gradient(nx, ny, lx, ly)

# 2D Laplace operator and 2D inverse Laplace operator in Fourier domain
K2, K2inv = nst.Spectral_Laplace(nx, ny, Kx, Ky)

# Simulation time
T_simu = 100

# Set discrete time step by choosing CFL number (condition: CFL <= 1)
CFL = 1
u = sc.real(ifft2(-Ky*K2inv*fft2(omega)))
v = sc.real(ifft2(Kx*K2inv*fft2(omega)))
u_max = sc.amax(sc.absolute(u))
v_max = sc.amax(sc.absolute(v))
t_step = (CFL*dx*dy)/(u_max*dy+v_max*dx)

# Start Simulation
t_sum = 0
i = 0
while t_sum <= T_simu:
    # Runge-Kitta 4 time simulation
    omega = nst.RK4(t_step, omega, Kx, Ky, K2, K2inv, nu)

    # Plot every 100th frame
    if 0 == i % 100:

        plt.imshow(omega)
        plt.pause(0.05)

    i += 1
    t_sum += t_step
