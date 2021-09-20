"""
Tools for truncation:
    eigenvectors, eigenvalues, spectral densities, etc.
"""
import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from numpy.linalg.linalg import eig
from scipy.integrate import quad

import primaries_2p as pr
import QED_2p

from operator import itemgetter
import pickle

def print_mat(M):
    for row in M.T:
        for val in row:
            print(f"{val.real:.2f}", end='\t')
        print()
    print()

def eig_M(del_max, pretty_eigenvectors=False, interaction=True, mass=1):
    """
    Return the eigenvalues and eigenstates of the mass matrix
    for max conformal dimension del_max as eigvals, eigvecs.
    Running 'generate(del_max)' first is required.

    Set "pretty_eigenvectors" to True, if you want them to be of type primaries.Primary
    Note that this takes significantly more time to compute, but can facilitate manipulations.

    Set "interaction" to False if you only want the mass term (default is True).

    Set "mass" to 0 if you only want the Gauge term, or to whatever value
    in units of e/sqrt(pi) (default is 1).
    """
    print(f"Loading from file '2DQED_trunc_D{del_max}.bin'...")
    try:
        with open(f'2DQED_trunc_D{del_max}.bin', 'rb') as f:
            d = pickle.load(f)
    except FileNotFoundError:
        raise(RuntimeError(
            f'Run <module>.generate({del_max}) before running eig_M({del_max})'))

    print('Loading basis and matrix elements...')
    M, V, B = np.array(d['mass']), np.array(d['inter']), pr.DirichletBasis()
    B.load_raw(d['basis'])

    print('Diagonalizing Hamiltonian...')
    w, v = np.linalg.eigh( (mass**2) * M + (V if interaction else 0) )
    w, v = zip(*sorted(zip(w,v.T), key=itemgetter(0)))
    v = np.array(v).T

    if(pretty_eigenvectors):
        print('Building eigenvectors...')
        v = [ pr.mono_sum(c*B[i] for i, c in enumerate(el)) for el in v.T ]

    w = np.array(w)
    return w, v


def lowest_eig_convergence_plot(mass, interaction, *delta):
    """
    Plot the convergence of the lowest eigenvalue on a range
    specified by delta.
    """
    masses = []
    for d in delta:
        #QED_2p.generate(d)
        w = eig_M(del_max=d, mass=mass, interaction=interaction)[0]
        masses.append((1/d**2, min(w)))
    
    masses.sort(key=itemgetter(0))
    plt.plot(*zip(*masses))
    plt.title(f"Convergence of the lowest $M^2$ eigenvalue\n$m^2={mass**2}$" + (
        ', with interaction' if interaction else ', without interaction'))
    plt.xlabel('$1/\Delta^2$')
    plt.ylabel('$m_0^2$')
    plt.show()


def bound_state_mass(m_max, del_max):
    """
    return a curve (two arrays) of the bound state mass as m evolves between 0 and m_max
    """
    x = np.linspace(0,m_max, 100)
    y = np.array([sqrt(min(w)) for w,v in (eig_M(del_max=del_max, mass=xi) for xi in x)])
    return x,y


def bound_state_psi(m, del_max, dx):
    """
    Return a curve (two arrays) of the bound state momentum distribution, as well as the mass.
    """
    w, v = eig_M(del_max=del_max, pretty_eigenvectors=True, mass=m, interaction=True)
    M2, psi = w[0], v[0]
    x = np.linspace(0,1,int(1/dx))
    y = np.empty(len(x))
    y = sum(psi[m].real * x**m.k1 * (1 - x)**m.k2 for m in psi.monomials)
    if y[len(x)//2] < 0: y = -y
    return x,y,np.sqrt(M2)


def plot_psi_converges(mass, del_max, interaction=True, y_low=None, y_high=None):
    """
    Plot the lowest eigenstate for mass 'mass,' and for
    Delta_max ranging from 3 to del_max, in progressively
    thicker shades of blue.
    Set y_low and y_high if the plot doesn't come out right.
    Set interaction to false if applicable.
    """
    for i in range(3,del_max+1): 
        plt.plot(*bound_state_psi(m=mass, del_max=i, dx=0.01)[:2],
                label=f'$\Delta_M={i}$',
                color=f'#0000ff{round(239*2**(i-12)+16):x}') 
    plt.title(f"Convergence of the wave function at $m={mass}$, from $\Delta_M=3$ to $12$") 
    plt.xlabel('$x$', size='large') 
    plt.ylabel('$\psi$', size='large') 
    if y_low is not None and y_high is not None:
        plt.ylim(y_low,y_high) 
    plt.show()
