
def approximate(r0):
    '''
    approximate()
    
    This function attempts to use the 4th Order Runge-Kutta approximation on the function given in the ReadMe.
    '''
    delta_t = 0.001 # delta t
    r = np.zeros((5*N, 2)) # creates the 5N by 2 array
    r[0] = r0
    J = np.array(([0,1],[-1,0]))
    for i in range(1, 5*N):
        k1 = delta_t*(J@(r[i-1]))
        k2 = delta_t*(J@(r[i-1] + k1/2))
        k3 = delta_t*(J@(r[i-1] + k2/2))
        k4 = delta_t*(J@(r[i-1] + k3))
        r[i] = r[i-1] + (k1+2*k2+2*k3+k4)/6
    return r

def sombrero():
    