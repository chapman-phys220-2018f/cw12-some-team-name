#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Jacob Anabi, Gage Kizzar
# Student ID: 2294644, 2291700
# Email: anabi@chapman.edu, kizzar@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW12
###

"""
sombrero Module Description:
This module attempts to solve a two coupled set of ODES, also known
as the 'Duffing oscillator'
"""

import numpy as np
import numba as nb
from matplotlib import pyplot as plt

@nb.jit
def scatter_plot(x, y, t, linestyle="-", xlabel=" ", ylabel=" ", title=" "):
    """
    parametric_plot_2d function description:
        This function plots the parametric curve (x(t),y(t))

        Args:
            x - x(t)
            y - y(t)
            t - time
            labels - legend label
            linestyles - linestyle
            xlabel - label for the domain
            ylabel - label for the range
            title - title of the graph
    """
    # Plotting
    fig, axes = plt.subplots()
    axes.scatter(x,y)
    plt.xlabel(xlabel) # x-axis label for graph
    plt.ylabel(ylabel) # y-axis label for graph
    plt.title(title) # the title of the graph
    plt.show() # show the graph

@nb.jit
def drdt(r, t, m=1, v=0.25, omega=1, F=0.18):
    """
    drdt function description:
    Computes the derivate of r with respect to t

    Args:
        r0 - the initial r-value
        t - the domain
    Default Args:
        m - the mass (default 1)
        v - the velocity (default 0.25)
        omega - angular frequency (default 1)
        F - driving force (default 0.18)
    """
    x, y = r[0], r[1] # x and y from r vector
    return np.array([y, (x-x**3-v*y+F*np.cos(omega*t))/m]) # derivative of x, derivative of y

def runge_kutta_4thOrd(r0, t, m=1, v=0.25, omega=1, F=0.18):
    """
    runge_kutta_4thOrd function description:
        This method computes the vector r(t)'s using Runge Kutta 4th Order's method.

        Args:
            r0 - the initial r-value
            t - the domain
        Default Args:
            m - the mass (default 1)
            v - the velocity (default 0.25)
            omega - angular frequency (default 1)
            F - driving force (default 0.18)
     """
    dt = 0.001 # delta t
    r = np.zeros((len(t), 2)) # creates the 5N by 2 array
    r[0] = r0 # initial r vector
    for i in range(1, len(t)):
        k1 = dt*(drdt(r[i-1], t[i-1], m=m, v=v, omega=omega, F=F)) # k1
        k2 = dt*(drdt(r[i-1]+k1/2, t[i-1]+dt/2, m=m, v=v, omega=omega, F=F)) # k2
        k3 = dt*(drdt(r[i-1]+k2/2, t[i-1]+dt/2, m=m, v=v, omega=omega, F=F)) # k3
        k4 = dt*(drdt(r[i-1]+k3, t[i-1]+dt, m=m, v=v, omega=omega, F=F)) # k4
        r[i] = r[i-1] + (k1+2*k2+2*k3+k4)/6 # point at index i
    return r

def plot(x, y, label=" ", linestyle="-", xlabel=" ", ylabel=" ", title=" "):
    """
    plot function description:
        This function plots some generic x and y values

        Args:
            x - the domain
            y - the range
            labels - legend label
            linestyles - linestyle
            xlabel - label for the domain
            ylabel - label for the range
            title - title of the graph
    """
    # Plotting
    fig = plt.figure(figsize=(8,6)) # Setting funciton figure size (width, height)
    axes = plt.axes() # Creating function plot axes
    axes.plot(x, y, label=label, linestyle=linestyle) # plotting graph
    axes.legend() # add legend
    plt.xlabel(xlabel) # x-axis label for graph
    plt.ylabel(ylabel) # y-axis label for graph
    plt.title(title) # the title of the graph
    plt.legend() #plot legend

    plt.show() # show the graph

def parametric_plot_2d(x, y, t, label=" ", linestyle="-", xlabel=" ", ylabel=" ", title=" "):
    """
    parametric_plot_2d function description:
        This function plots the parametric curve (x(t),y(t))

        Args:
            x - x(t)
            y - y(t)
            t - time
            labels - legend label
            linestyles - linestyle
            xlabel - label for the domain
            ylabel - label for the range
            title - title of the graph
    """
    # Plotting
    fig = plt.figure(figsize=(8,6)) # Setting funciton figure size (width, height)
    axes = plt.axes() # Creating function plot axes
    axes.plot(x, y, label=label, linestyle=linestyle) # plotting graph
    axes.legend() # add legend
    plt.xlabel(xlabel) # x-axis label for graph
    plt.ylabel(ylabel) # y-axis label for graph
    plt.title(title) # the title of the graph
    plt.legend() #plot legend

    plt.show() # show the graph

def poincare_plot(x, y, t, linestyle="-", xlabel=" ", ylabel=" ", title=" ", n=51):
    """
    poincare_plot function description:
        This function plots the poincare section of the parametric curve

        Args:
            x - x(t)
            y - y(t)
            t - time
            labels - legend label
            linestyles - linestyle
            xlabel - label for the domain
            ylabel - label for the range
            title - title of the graph
    """
    dt = 0.001
    x_p = np.zeros(n)
    y_p = np.zeros(n)
    t_p = np.zeros(n)
    for i in range(0, n):
        x_p[i] = x[int((i*2*np.pi)/dt)]
        y_p[i] = y[int((i*2*np.pi)/dt)]
        t_p[i] = t[int((i*2*np.pi)/dt)]
    scatter_plot(x_p, y_p, t_p, linestyle, xlabel, ylabel, title)