import math
import numpy as np
import sombrero

def test_sombrero_initial():
    t = np.arange(0, 2*np.pi*50+0.001, 0.001) # t-values, dt of 0.001
    r0 = np.array([0, 0]) # initial conditions [x(0),y(0)]
    r = sombrero.runge_kutta_4thOrd(r0, t, F=0.4) # r vector
    x = np.array([i[0] for i in r]) # x points
    y = np.array([i[1] for i in r]) # y points
    assert (x[0] == 0 and y[0] == 0) # checking initial conditions

def test_sombrero_derivative_x():
    t = np.arange(0, 2*np.pi*50+0.001, 0.001) # t-values, dt of 0.001
    r0 = np.array([0, 0]) # initial conditions [x(0),y(0)]
    r = sombrero.runge_kutta_4thOrd(r0, t, F=0.4) # r vector
    x = np.array([i[0] for i in r]) # x points
    assert ((np.amin(x) > -1.5 or math.isclose(np.amin(x), -1.5, rel_tol=1e-01)) and (np.amax(x) < 1.5 or math.isclose(np.amax(x), 1.5, rel_tol=1e-01)))

def test_sombrero_derivative_y():
    t = np.arange(0, 2*np.pi*50+0.001, 0.001) # t-values, dt of 0.001
    r0 = np.array([0, 0]) # initial conditions [x(0),y(0)]
    r = sombrero.runge_kutta_4thOrd(r0, t, F=0.4) # r vector
    y = np.array([i[1] for i in r]) # y points
    assert ((np.amin(y) > -1.1 or math.isclose(np.amin(y), -1.1, rel_tol=1e-01)) and (np.amax(y) < 1.1 or math.isclose(np.amax(y), 1.1, rel_tol=1e-01)))