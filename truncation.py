"""
Tools for truncation:
    eigenvectors, eigenvalues, spectral densities, etc.
"""
import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt

import primaries_2p as pr
import QED_2p

from operator import itemgetter
import pickle


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
    w, v = np.linalg.eig( (mass**2) * M + (V if interaction else 0) )
    w = np.array([v.real for v in w])

    if(pretty_eigenvectors):
        print('Building eigenvectors...')
        v = [ pr.mono_sum(c*B[i] for i, c in enumerate(el)) for el in v.T ]
    return w, v


def lowest_eig_convergence_plot(mass, interaction, *delta):
    """
    Plot the convergence of the lowest eigenvalue on a range
    specified by delta.
    """
    masses = []
    for d in delta:
        #QED_2p.generate(d)
        w, v = eig_M(del_max=d, mass=mass, interaction=interaction)
        masses.append((1/d, min(w)))
    
    masses.sort(key=itemgetter(0))
    plt.plot(*zip(*masses))
    plt.title(f"Convergence of the lowset $M^2$ eigenvalue\n$m={mass}e^2/\pi$" + (
        ', with interaction' if interaction else ', without interaction'))
    plt.xlabel('$1/\Delta$')
    plt.ylabel('$m_0^2$')
    plt.show()


def bound_state_mass(m_max, del_max):
    """
    return a curve (two arrays) of the bound state mass as m evolves between 0 and m_max
    """
    x = np.linspace(0,m_max, 100)
    y = np.array([sqrt(min(w)) for w,v in (eig_M(del_max=del_max, mass=xi) for xi in x)])
    return x,y


def stress_spectral_density(w,v):
    """
    Takes eigenvalues and eigenvectors from the truncation output
    and computes the spectral density \\rho(m) for T--
    """
    # We know 3.0  ∂^1ψ† ∂^2ψ - 3.0  ∂^2ψ† ∂^1ψ is the 1st element in our basis
    O1 = pr.Primary(pr.Monomial(3,1,2) + pr.Monomial(-3,2,1))
    N2 = pr.Graham(O1,O1)
    x,y = zip(*sorted(
        [(m2, v[i,1]**2 * N2 * 4*sqrt(2)/3) for i, m2 in enumerate(w)], key=itemgetter(0)
    ))
    return np.array(x), np.array(y)
