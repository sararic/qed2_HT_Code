"""
Helper module to compute the bound state wave functions in the Schwinger model.
Can be run with a set of fermion masses, which will output a plot of the corresponding wave functions.
"""
import os
from operator import itemgetter

import numpy as np
from numba import jit
from matplotlib import pyplot as plt

dx = 0.001
max_cache_size = 200 #KibiBytes

max_cache_size = int((2<<9)*max_cache_size)


def manage_cache():
    """
    Ensure the cache folder doesn't go above a specified size,
    by removing old unused cache.
    """
    cache = [
        {
            'name': name,
            'size': os.stat(name).st_size,
            'date': os.stat(name).st_atime
        } 
        for name in map(lambda s: '__pycache__/'+s, os.listdir('__pycache__'))]
    cache.sort(key=itemgetter('date'))
    cache.reverse()

    while sum(file['size'] for file in cache) > max_cache_size:
        os.remove(cache.pop()['name'])


def integral(f, a, b):
    """
    Compute the numeric integral of an array over a given interval
    """
    return sum(dx*(f[x]+f[x+1])/2 for x in range(a,b-1))


@jit(cache=True) # just in time compilation
def ker(x, y, m, M2, epsilon):
    """
    Kernel of the integral equation
    """ 
    # K is singular on its diagonal.
    # Since we're taking epsilon = dx, the principal value prescription
    # amounts to setting the diagonal to 0
    if x == y: return 0
    if m == 1: # we must ensure no division by 0 occurs
        return (1 - 1/(x-y)**2)/(M2 - 2/epsilon)
    return x * (1-x) * (1 - 1/(x-y)**2)/( (M2 - 2/epsilon)*x*(1-x) + 1 - m**2 )


@jit(cache=True) # just in time compilation
def ker_array(m, M2, space, epsilon):
    """
    Kernel of the integral equation, as an array
    """
    return [ [ ker( x, y, m, M2, epsilon ) for x in space ] for y in space]


def psi_recursive(m, M2, n=10):
    """
    Attempt to solve the equation recursively.
    This method takes a lot of time to converge, and has a lot of artefacts
    unless we use a very small dx. Thus we use it only when absolutely
    necessary (m < 1)
    """
    # account for the fact that the sequence is alternating
    n = 2*n
    # discretize the kernel as a matrix:
    space = np.linspace(0, 1, int(1/dx))
    K = np.array(ker_array(m, M2, space, dx))
    # Tail recursive function
    def psir(f, n, i):
        if i == n: return f
        newf = np.array([
            integral( K[x,:]*f, 0, x-1 ) +
             integral( K[x,:]*f, x+1, len(space) ) for x in range(len(space)) ])
        return psir(newf, n, i+1)

    f0 = np.array([.5]*int(1/dx))
    f0[0] = f0[-1] = 0
    f = psir(f0, n, 0)
    # normalize
    return space, f/integral(f,0,len(f))


def psi_linalg(m, M2, plot_spectrum=False):
    """
    Solve the equation as an eigenvector problem.
    This method works best when K is not smooth on its diagonal
    (http://www.numerical-methods.com/inteq.htm) which luckily is the case here.
    Furthermore, it will diverge when m < 1, because of the fact that
    our Kernel has a factor of epsilon of which we've not yet taken the limit.
    """
    #discretize the kernel as a matrix:
    space = np.linspace(0, 1, int(1/dx))
    K = np.array(ker_array(m, M2, space, dx))

    w, v = np.linalg.eig(K)
    psi = _findclosestsolution(w, v)

    if plot_spectrum:
        plt.plot(range(len(w)), np.sort(w), '.')
        plt.xlabel('index')
        plt.ylabel('Eigenvalue')
        plt.title(f'Spectrum of K(x,y) for dx = {dx}')
        plt.show()
    return space, psi


def _findclosestsolution(w,v):
    """
    extract eigenvector from v such that the eigenvalue in w
    is as close to 1/dx as possible
    """
    win = 0
    for i in range(len(w)):
        if w[i] > w[win]:
            win = i
    f = v[:,win]
    return f/integral(f, 0, len(f))


def solve_psi_M(m, n=2):
    """
    Recursively solve for M^2, using psi_linalg for m >= 1
    and psi_recursive for m < 1.
    This is because when m < 1, and epsilon is not infinite,
    the solution has artefacts near x = 0,1. Precisely, it has poles at:
    x_s = 1/2 ± sqrt(1/4 - epsilon*(1-m**2)/(2 - epsilon*M**2))
    """
    x = np.linspace(0, 1, int(1/dx))
    M2 = 1
    bra_M2 = np.zeros(len(x))
    bra_M2[1:-1] = m**2 /(x[1:-1] * (1 - x[1:-1]))

    alg = psi_recursive if m < 1 else psi_linalg

    for i in range(n):
        x, psi = alg(m, M2)
        M2 = 1 + dx*np.dot(bra_M2, psi)
    return x, psi, np.sqrt(M2)


def main(*masses):
    for m in masses:
        x, psi, M = solve_psi_M(m, 5)
        plt.plot(x, psi, label=f"m={m}, M = {round(M,1)}")
    manage_cache()


if __name__ == "__main__":
    import argparse    

    parser = argparse.ArgumentParser(
             description =
             """
Helper module to compute the bound state wave functions in the Schwinger model.
Can be run with a set of fermion masses, which will output a plot of the corresponding wave functions.
             """)

    parser.add_argument('masses', metavar='m', type=float, nargs='+',
        help = "Masses for which to plot a wave function")

    args = parser.parse_args()

    main(*args.masses)

    plt.xlabel('x')
    plt.ylabel('$\psi$')
    plt.title('Solutions to the bound state wave function')
    plt.legend()
    plt.show()
