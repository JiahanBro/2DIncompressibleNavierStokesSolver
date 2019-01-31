#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:27:19 2018

@author: jiahan
"""

import scipy as sc
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt


def Spectral_Gradient(nx, ny, lx, ly):
    # Create wavenumber vector for x-direction
    tmp1 = sc.linspace(0, nx/2, int(nx/2+1))*2*sc.pi/lx
    tmp2 = sc.linspace(1-nx/2, -1, int(nx/2-1))*2*sc.pi/lx
    kx = sc.concatenate((tmp1, tmp2))

    # Create wavenumber vector for y-direction
    tmp1 = sc.linspace(0, ny/2, int(ny/2+1))*2*sc.pi/ly
    tmp2 = sc.linspace(1-ny/2, -1, int(ny/2-1))*2*sc.pi/ly
    ky = sc.concatenate((tmp1, tmp2))

    # Dealiasing with the 2/3 rule
    trunc_x_low = int(sc.floor(2/3*nx/2))+1
    trunc_x_high = int(sc.ceil(4/3*nx/2))
    kx[trunc_x_low:trunc_x_high] = sc.zeros(trunc_x_high - trunc_x_low)

    trunc_y_low = int(sc.floor(2/3*ny/2))+1
    trunc_y_high = int(sc.ceil(4/3*ny/2))
    ky[trunc_y_low:trunc_y_high] = sc.zeros(trunc_y_high - trunc_y_low)

    # Create Gradient operators in Fourier domain for x- and y-direction
    Kx, Ky = sc.meshgrid(ky, kx)
    Kx = 1j*Kx
    Ky = 1j*Ky

    return Kx, Ky


def Spectral_Laplace(nx, ny, Kx, Ky):
    # Create 2D Laplace operator in Fourier domain
    K2 = Kx*Kx + Ky*Ky

    # Create Inverse 2D Laplace operator in Fourier domain
    K2inv = sc.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if K2[i, j] != 0:
                K2inv[i, j] = 1/K2[i, j]

    return K2, K2inv


def rhs(omega, Kx, Ky, K2, K2inv, nu):
    # Transform vorticity to Fourier space
    omega_hat = fft2(omega)

    # Derivative of vorticity in x-direction and afterwards transformed to Real
    # space
    omega_x = sc.real(ifft2(Kx*omega_hat))

    # Derivative of vorticity in y-direction and afterwards transformed to Real
    # space
    omega_y = sc.real(ifft2(Ky*omega_hat))

    # Velocity in x-direction by solving Poisson equation and afterwards
    # transformed to Real space
    u = sc.real(ifft2(-Ky*K2inv*omega_hat))

    # Velocity in y-direction by solving Poisson equation and afterwards
    # transformed to Real space
    v = sc.real(ifft2(Kx*K2inv*omega_hat))

    # Calculation of Diffusion term and afterwards transformed to Real space
    diffusion = sc.real(ifft2(nu*K2*omega_hat))

    # RHS of 2D Vorticity equation
    RHS = diffusion - u*omega_x - v*omega_y
    return RHS


def RK4(t_step, omega, Kx, Ky, K2, K2inv, nu):
    k1 = rhs(omega, Kx, Ky, K2, K2inv, nu)
    k2 = rhs(omega + k1*t_step/2, Kx, Ky, K2, K2inv, nu)
    k3 = rhs(omega + k2*t_step/2, Kx, Ky, K2, K2inv, nu)
    k4 = rhs(omega + k3*t_step, Kx, Ky, K2, K2inv, nu)
    return omega + t_step*(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)


def dancing_vortices(nx, ny, dx, dy):
    # Initial vortex x-position
    x0s = sc.array([sc.pi*0.75, sc.pi*1.25, sc.pi*1.25])
    # Initial vortex y-position
    y0s = sc.array([1, 1, 1+1/(2*sc.sqrt(2))]) * sc.pi

    # Vortex core size
    betas = sc.array([1, 1, 1]) / sc.pi
    # Strength
    alphas = sc.array([1, 1, -1/2]) * sc.pi

    # Build field
    x = sc.linspace(dx, 2*sc.pi, nx)
    y = sc.linspace(dx, 2*sc.pi, ny)
    x, y = sc.meshgrid(x, y)
    x = sc.transpose(x)
    y = sc.transpose(y)

    # Calculate omega
    omega = sc.zeros([nx, ny], dtype='float64')
    for i in range(0, len(x0s)):
        x0 = x0s[i]
        y0 = y0s[i]
        beta = betas[i]
        alpha = alphas[i]
        R2 = (sc.multiply((x-x0), (x-x0)) + sc.multiply((y-y0), (y-y0))) / \
            pow(beta, 2)
        omega_part = alpha * sc.exp(-R2)
        omega += omega_part

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    print("Initialized three dancing vortices")
    plt.imshow(omega)
    plt.colorbar()
    plt.pause(0.05)

    return omega, p


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
    y = sc.linspace(dy, ly, ny)
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


def random_vortices(nx, ny):
    omega_hat = sc.zeros([nx, ny])
    tmp = sc.randn(3) + 1j*sc.randn(3)
    omega_hat[0, 4] = tmp[0]
    omega_hat[1, 1] = tmp[1]
    omega_hat[3, 0] = tmp[2]
    omega = sc.real(ifft2(omega_hat))
    omega = omega/sc.amax(sc.amax(omega))

    # Initialize pressure field
    p = sc.zeros([nx, ny])

    print("Initialized random vortices")
    plt.imshow(omega)
    plt.colorbar()
    plt.pause(0.05)

    return omega, p
