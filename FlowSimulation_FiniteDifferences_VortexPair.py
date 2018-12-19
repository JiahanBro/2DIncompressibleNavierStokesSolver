#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:09:36 2018

@author: jiahan
"""

import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import IncompNST_Tools.FiniteDifferences as nst

# Physical constants
nu = 0.001

# Length of field
lx = 1
ly = 1

# Number of grid points
nx = 64
ny = 64

# Grid increments
dx = lx/nx
dy = ly/ny

# Initialize vortex
u, v, p = nst.vortex_pair(nx, ny, dx, dy)

# Simulation time
T_simu = 10000

# Set discrete time step by choosing CFL number (condition: CFL <= 1)
CFL = 1
u_max = sc.amax(sc.absolute(u))
v_max = sc.amax(sc.absolute(v))
t_step = (CFL*dx*dy)/(u_max*dy+v_max*dx)
t_step_vec = sc.reshape(t_step*sc.ones(nx*ny), (nx*ny, 1), order="F")
t_step_vec = sp.csc_matrix(t_step_vec)

# Reshape velocity fields: u_vec, v_vec denote the vector-shaped velocity field
# and U, V denote the velocity values written in a diagonal matrix. The
# diagonal matrix form is needed to calculate the operator for the convective
# term (Transport Operator) in the NST-equation. Both the vector-shaped and
# diagonal-matrix shaped velocities are transformed to sparse matrices
u_vec = sc.reshape(u, (nx*ny, 1), order="F")
u_vec = sp.csc_matrix(u_vec)
U = sc.reshape(u, (nx*ny), order="F")
U = sc.diag(U)
U = sp.csc_matrix(U)

v_vec = sc.reshape(v, (nx*ny, 1), order="F")
v_vec = sp.csc_matrix(v_vec)
V = sc.reshape(v, (nx*ny), order="F")
V = sc.diag(V)
V = sp.csc_matrix(V)

# Reshape pressure field: p_vec, denotes the vector-shaped and sparse pressure
# field
p_vec = sc.reshape(p, (nx*ny, 1), order="F")
p_vec = sp.csc_matrix(p_vec)

# Gradient operators for x- and y-directions (made sparse)
Gx = nst.Grad_x(dx, nx, ny)
Gx = sp.csc_matrix(Gx)
Gy = nst.Grad_y(dy, nx, ny)
Gy = sp.csc_matrix(Gy)

# Divergence operators for x- and y-directions (made sparse)
Dx = nst.Div_x(dx, nx, ny)
Dx = sp.csc_matrix(Dx)
Dy = nst.Div_y(dy, nx, ny)
Dy = sp.csc_matrix(Dy)

# Laplace-Operator in 2D domain (is already sparse)
L = nst.Laplace(Gx, Gy, Dx, Dy)

# Transport Operator in 2D domain for the convective term (is already sparse)
Du = nst.Convective(Gx, Gy, Dx, Dy, U, V)

# Start Simulation
t_sum = 0
i = 0
while t_sum <= T_simu:
    # Make vector-shaped velocity to diagonal-shaped velocity for calculation
    # of transport operator for the convective term
    U = sc.asarray(sp.csc_matrix.todense(u_vec)).reshape(-1)
    U = sp.csc_matrix(sc.diag(U))
    V = sc.asarray(sp.csc_matrix.todense(v_vec)).reshape(-1)
    V = sp.csc_matrix(sc.diag(V))

    # Calculation of Transport operator
    Du = nst.Convective(Gx, Gy, Dx, Dy, U, V)

    # Runge-Kutta 4 for time integration
    u_vec = nst.RK4(t_step_vec, u_vec, p_vec, Gx, Du, L, nu)
    v_vec = nst.RK4(t_step_vec, v_vec, p_vec, Gy, Du, L, nu)

    # Solve Laplace equation to update pressure and to compensate for numerical
    # divergence
    (u_vec, v_vec, p_vec) = \
        nst.solve_laplace(p_vec, u_vec, v_vec, Gx, Gy, Dx, Dy, L, nx, ny)

    if 0 == i % 100:
        # Plot rotation field
        rot_uv = sc.dot(Dx, v_vec) - sc.dot(Dy, u_vec)
        rot_uv = sp.csc_matrix.todense(rot_uv)
        rot_uv = sc.reshape(rot_uv, (nx, ny), order="F")
        plt.imshow(rot_uv)
        plt.pause(0.05)

    i += 1
    t_sum += t_step
    print(i, t_sum)
