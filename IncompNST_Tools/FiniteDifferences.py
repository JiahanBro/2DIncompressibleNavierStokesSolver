#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:55:19 2018

@author: jiahan
"""

import numpy as np
import numpy.linalg as npla
import scipy as sc
import scipy.sparse.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt


def Grad_x(dx, nx, ny):
    Gx = sc.zeros([nx, nx])

    # Build gradient matrix
    for i in range(0, nx):
        Gx[(i-2), i] = 1/6
        Gx[(i-1), i] = -1
        Gx[i % nx, i] = 1/2
        Gx[(i+1) % nx, i] = 1/3

    # Divide through increments
    Gx /= dx

    # Expand gradient matrix to whole field
    Gx = sc.kron(sc.eye(ny), Gx)

    return Gx


def Grad_y(dy, nx, ny):
    Gy = sc.zeros([ny, ny])

    # Build gradient matrix
    for i in range(0, ny):
        Gy[i, (i-2)] = 1/6
        Gy[i, (i-1)] = -1
        Gy[i, i % ny] = 1/2
        Gy[i, (i+1) % ny] = 1/3

    # Divide through increments
    Gy /= dy

    # Expand gradient matrix to whole field
    Gy = sc.kron(Gy, sc.eye(nx))

    return Gy


def Div_x(dx, nx, ny):
    Dx = sc.zeros([nx, nx])

    # Build divergence matrix
    for i in range(0, nx):
        Dx[(i-1), i] = -1/3
        Dx[i % nx, i] = -1/2
        Dx[(i+1) % nx, i] = 1
        Dx[(i+2) % nx, i] = -1/6

    # Divide through increments
    Dx /= dx

    # Expand divergence matrix to whole field
    Dx = sc.kron(sc.eye(ny), Dx)

    return Dx


def Div_y(dy, nx, ny):
    Dy = sc.zeros([ny, ny])

    # Build divergence matrix
    for i in range(0, ny):
        Dy[i, (i-1)] = -1/3
        Dy[i, i % ny] = -1/2
        Dy[i, (i+1) % ny] = 1
        Dy[i, (i+2) % ny] = -1/6

    # Divide through increments
    Dy /= dy

    # Expand divergence matrix to whole field
    Dy = sc.kron(Dy, sc.eye(nx))

    return Dy


def Laplace(Gx, Gy, Dx, Dy):
    return sc.dot(Dx, Gx) + sc.dot(Dy, Gy)


def Convective(Gx, Gy, Dx, Dy, U, V):
    return (sc.dot(Dx, U) + sc.dot(U, Gx) + sc.dot(Dy, V) + sc.dot(V, Gy))/2


def rhs(u, p, G, Du, L, nu):
    return nu*sc.dot(L, u) - sc.dot(Du, u) - sc.dot(G, p)


def RK4(ht, u, p, G, Du, L, nu):

    k1 = rhs(u, p, G, Du, L, nu)
    k2 = rhs(u + k1.multiply(ht/2), p, G, Du, L, nu)
    k3 = rhs(u + k2.multiply(ht/2), p, G, Du, L, nu)
    k4 = rhs(u + k3.multiply(ht), p, G, Du, L, nu)
    return u + ht.multiply(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)


def solve_laplace(p, u, v, Gx, Gy, Dx, Dy, L, nx, ny):
    s = sc.dot(Dx, u) + sc.dot(Dy, v)
    dp = sp.csc_matrix(sc.reshape(la.spsolve(L, s), (nx*ny, 1), order="F"))

    u -= sc.dot(Gx, dp)
    v -= sc.dot(Gy, dp)
    p += dp

    return u, v, p


def taylor_green_vortex(a, b, nu, rho, nx, ny, dx, dy):
    # Initialize velocity field
    u = sc.zeros([nx, ny])
    v = sc.zeros([nx, ny])

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    for i in range(0, nx):
        for j in range(0, ny):
            u[i, j] = sc.exp(-2*nu) * sc.cos(a*i*dx) * sc.sin(b*j*dy)
            v[i, j] = sc.exp(-2*nu) * sc.sin(a*i*dx) * sc.cos(b*j*dy)
            p[i, j] = -(pow(sc.exp(-2*nu), 2)) * (rho/4) * \
                (sc.cos(2*i*dx) + sc.cos(2*j*dy))

    # Plot pressure field
    plt.imshow(p)
    plt.colorbar()
    plt.show()

    # Plot velocity field
    x = sc.linspace(dx, 2*sc.pi, nx)
    y = sc.linspace(dy, 2*sc.pi, ny)
    xv, yv = sc.meshgrid(x, y)
    plt.quiver(xv, yv, u, v, units='width')
    plt.show()

    print("Initialized Taylor-Green vortex")

    return u, v, p


def dancing_vortices(nx, ny, dx, dy):
    # Initial vortex x-position
    x0s = sc.array([sc.pi*0.75, sc.pi*1.25, sc.pi*1.25])
    # Initial vortex y-position
    y0s = sc.array([1, 1, 1+1/(2*sc.sqrt(2))]) * sc.pi

    # V ortex core size
    betas = sc.array([1, 1, 1]) / sc.pi
    # Strength
    alphas = sc.array([1, 1, -1/2]) * sc.pi

    # Build field
    x = sc.linspace(dx, 2*sc.pi, nx)
    y = sc.linspace(dx, 2*sc.pi, ny)
    x, y = sc.meshgrid(x, y)
    x = sc.transpose(x)
    y = sc.transpose(y)

    # Gradient operators
    Gx = Grad_x(dx, nx, ny)
    Gy = Grad_y(dy, nx, ny)

    # Divergence operators
    Dx = Div_x(dx, nx, ny)
    Dy = Div_y(dy, nx, ny)

    # Laplace-Operator in 2D
    L = Laplace(Gx, Gy, Dx, Dy)

    # Calculate omega
    omega = sc.zeros([nx, ny])
    for i in range(0, len(x0s)):
        x0 = x0s[i]
        y0 = y0s[i]
        beta = betas[i]
        alpha = alphas[i]
        R2 = (sc.multiply((x-x0), (x-x0)) + sc.multiply((y-y0), (y-y0))) / \
            pow(beta, 2)
        omega_part = alpha * np.exp(-R2)
        omega += omega_part

    omega = sc.reshape(omega, (nx*ny, 1), order="F")

    # Determine psi
    psi = npla.solve(L, omega)
    psi_x = np.dot(Gx, psi)
    psi_y = np.dot(Gy, psi)

    # Determine velocity components
    u = -psi_y
    v = psi_x

    # Compensate numerical divergence
    s = np.dot(Dx, u) + np.dot(Dy, v)
    dp = sc.reshape(npla.solve(L, s), (nx*ny, 1), order="F")

    u -= np.dot(Gx, dp)
    v -= np.dot(Gy, dp)

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    # Plot rotation of velocity field
    rot_uv = np.dot(Dx, v) - np.dot(Dy, u)
    rot_uv = sc.reshape(rot_uv, (nx, ny), order="F")
    print("Initialized three dancing vortices")
    plt.imshow(rot_uv)
    plt.colorbar()
    plt.pause(0.5)

    # Reshape velocity arrays back for output
    u = sc.reshape(u, (nx, ny), order="F")
    v = sc.reshape(u, (nx, ny), order="F")

    return u, v, p


def vortex_pair(nx, ny, dx, dy):
    # Domain size
    lx = nx * dx
    ly = ny * dy
        
    # Initial vortex x-position
    x0s = sc.array([0.4, 0.6])*lx
    # Initial vortex y-position
    y0s = sc.array([0.5, 0.5])*ly

    # Strength
    alphas = sc.array([-299.5, 299.5])

    # Build field
    x = sc.linspace(dx, lx, nx)
    y = sc.linspace(dx, ly, ny)
    x, y = sc.meshgrid(x, y)
    x = sc.transpose(x)
    y = sc.transpose(y)

    # Gradient operators
    Gx = Grad_x(dx, nx, ny)
    Gy = Grad_y(dy, nx, ny)

    # Divergence operators
    Dx = Div_x(dx, nx, ny)
    Dy = Div_y(dy, nx, ny)

    # Laplace-Operator in 2D
    L = Laplace(Gx, Gy, Dx, Dy)

    # Calculate omega
    omega = sc.zeros([nx, ny], dtype='float64')
    for i in range(0, len(x0s)):
        x0 = x0s[i]
        y0 = y0s[i]
        alpha = alphas[i]
        r = 10*sc.sqrt((x-x0)**2 + (y-y0)**2)
        omega_part = alpha * (1-(r**2)) * sc.exp(-r**2)
        omega += omega_part

    omega = sc.reshape(omega, (nx*ny, 1), order="F")

    # Determine psi
    psi = npla.solve(L, omega)
    psi_x = np.dot(Gx, psi)
    psi_y = np.dot(Gy, psi)

    # Determine velocity components
    u = -psi_y
    v = psi_x

    # Compensate numerical divergence
    s = np.dot(Dx, u) + np.dot(Dy, v)
    dp = sc.reshape(npla.solve(L, s), (nx*ny, 1), order="F")

    u -= np.dot(Gx, dp)
    v -= np.dot(Gy, dp)

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    # Plot rotation of velocity field
    rot_uv = np.dot(Dx, v) - np.dot(Dy, u)
    rot_uv = sc.reshape(rot_uv, (nx, ny), order="F")
    print("Initialized three dancing vortices")
    plt.imshow(rot_uv)
    plt.colorbar()
    plt.pause(0.5)

    # Reshape velocity arrays back for output
    u = sc.reshape(u, (nx, ny), order="F")
    v = sc.reshape(u, (nx, ny), order="F")

    return u, v, p



def vortex_pair2(nx, ny, dx, dy):
    # Domain size
    lx = nx * dx
    ly = ny * dy
        
    # Initial vortex x-position
    x0s = sc.array([0.4, 0.6])*lx
    # Initial vortex y-position
    y0s = sc.array([0.5, 0.5])*ly

    # Strength
    alphas = sc.array([-299.5, 299.5])

    # Build field
    x = sc.linspace(dx, lx, nx)
    y = sc.linspace(dx, ly, ny)
    x, y = sc.meshgrid(x, y)
    x = sc.transpose(x)
    y = sc.transpose(y)

    # Calculate omega
    omega = sc.zeros([nx, ny], dtype='float64')
    for i in range(0, len(x0s)):
        x0 = x0s[i]
        y0 = y0s[i]
        alpha = alphas[i]
        r = 10*sc.sqrt((x-x0)**2 + (y-y0)**2)
        omega_part = alpha * (1-(r**2)) * sc.exp(-r**2)
        omega += omega_part

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    print("Initialized vortex pair")
    plt.imshow(omega)
    plt.colorbar()
    plt.pause(0.05)

    return omega, p
