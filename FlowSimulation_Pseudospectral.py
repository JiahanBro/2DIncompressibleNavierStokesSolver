#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:45:29 2018

@author: jiahan
"""

import scipy as sc
import matplotlib.pyplot as plt
import IncompNST_Tools.Pseudospectral as nst

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

# Simulation time
T_simu = 10000
t_step = 0.04

# Initialize vortex: choose one of the following:
# 1. vortex pair in the middle of the domain
omega, p = nst.vortex_pair(nx, ny, dx, dy)

# 2. three vortices dancing moving arround
#  omega, p = nst.dancing_vortices(nx, ny, dx, dy)

# Gradient operators in Fourier domain for x- and y-direction
Kx, Ky = nst.Spectral_Gradient(nx, ny, lx, ly)

# 2D Laplace operator and 2D inverse Laplace operator in Fourier domain
K2, K2inv = nst.Spectral_Laplace(nx, ny, Kx, Ky)


for i in range(0, T_simu):

    omega = nst.RK4(t_step, omega, Kx, Ky, K2, K2inv, nu)

    if 0 == i % 100:

        plt.imshow(omega)
        plt.colorbar()
        plt.show()
