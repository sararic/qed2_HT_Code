import numpy as np
from numpy import sqrt
from scipy.linalg import orth
from scipy.linalg.matfuncs import fractional_matrix_power
from scipy.special import factorial

class Monomial:
    """
    Dataclass for monomials, consisting of an even number of particles, half of which
    are complex conjugates of the field.
    """
    def __init__(self, coeff, k1, k2):
        self.coeff = coeff
        self.k1 = k1
        self.k2 = k2
    
    def conf_dim(self):
        return self.k1 + self.k2 + 1

    def __repr__(self):
        rep = "  ∂^{}ψ† ∂^{}ψ"
        res = '' if self.coeff == 1 else str(self.coeff)
        res += rep.format(self.k1,self.k2)
        return res
 
    def __hash__(self):
        if not self.coeff: return 0
        return hash(str(self))
 
    def simplifies_with(self, B):
        return self.k1 == B.k1 and self.k2 == B.k2

    def __eq__(self, B):
        if not self.coeff: return not B.coeff
        return self.simplifies_with(B) and self.coeff == B.coeff

    def __mul__(self, B):
        return Monomial(B*self.coeff, self.k1, self.k2)

    def __rmul__(self, a):
        return self*a
 
    def __add__(self, B):
        """takes a Polynomial/monomial, returns a Polynomial"""
        return (Polynomial(self) + B).simplify()


class Polynomial:
    """Implements Polynomials"""
    def __init__(self, *monomials):
        self.monomials = list(monomials)
        self.simplify()
 
    @classmethod
    def from_array(cls, a, max_n, max_d):
        l = []
        for i, m in enumerate(ordered_indexing(max_n, max_d)):
            l.append(a[i]*m)
        return Polynomial(*l)

    def to_array(self, max_n, max_d):
        r = []
        for m in ordered_indexing(max_n, max_d):
            r.append(next((el.coeff for el in self.monomials if el.simplifies_with(m)), 0))
        return np.array(r)

    def simplify(self, l = None, idx = 0):
        if l is None:
            l = self.monomials
        if idx+1 == len(l): return self
        match = next((i for i in range(idx+1,len(l)) if l[idx].simplifies_with(l[i])), None)
        if (match is not None) or (not l[idx].coeff):
            if l[idx].coeff:
                l[match] = (1+l[match].coeff/l[idx].coeff)*l[idx]
            l.pop(idx)
            idx -= 1
        return self.simplify(l, idx+1)

    def __getitem__(self, mono):
        """Returns the coefficient of the monomial mono in self"""
        if type(mono) is not Monomial:
            raise TypeError('Index is not a Monomial')
        return next((m.coeff for m in self.monomials if m.simplifies_with(mono)), 0)

    def __add__(self, B):
        try:
            return Polynomial(*self.monomials, *B.monomials)
        except AttributeError:
            return Polynomial(*self.monomials, B)
    
    def __mul__(self, B):
        try:
            B = B.monomials
        except AttributeError:
            B = [B]
        r = []
        for m1 in self.monomials:
            for m2 in B:
                r.append(m1*m2)
        return Polynomial(*r)
    
    def __rmul__(self, A):
        return self*A

    def __repr__(self):
        r = ''
        for m in self.monomials:
            if m.coeff :
                r += f'{m}  +  '
        if not r: r = '0     '
        return r[:-5]
    
    def __hash__(self):
        return sum(hash(m) for m in self.monomials)
    
    def __eq__(self, B):
        return hash(self) == hash(B)


class Primary(Polynomial):
    def __init__(self, obj):
        """obj can be a monomial or polynomial"""
        if type(obj) is Monomial:
            Polynomial.__init__(self, obj)
        else:
            Polynomial.__init__(self, *obj.monomials)

    def conf_dim(self):
        return next(m.conf_dim() for m in self.monomials if m.coeff )


def mono_sum(l):
    """Sum a list of monomials together"""
    r = next(l)
    for m in l: r = r + m
    return r


def double_trace(l):
    """
    Implements the double-trace operator
    """
    return Primary(mono_sum(Monomial(
        (-1)**m * factorial(l+2)**2/(
            factorial(m) * factorial(m+2) * factorial(l-m) * factorial(l-m+2)
        ), m+1, l-m+1) for m in range(l+1)))


def ordered_indexing(max_d):
    """
    Generates a well ordered sequence of monomials that span all possible combinations
    for the given arguments. Can be used for indexing a polynomial/primary.
    max_n: maximum particle number
    max_d:  maximum conformal dimension
    """
    l = max_d - 1
    for k1 in range(1, l):
        for k2 in range(1, l+1-k1):
            yield Monomial(1, k1, k2)


class DirichletBasis:
    def __init__(self):
        self.states = None

    def generate(self, Del_max):
        self.states = [double_trace(l) for l in range(Del_max-2)]
        self._orth()

    def load_raw(self, raw_states):
        """
        Load a basis from a raw python list, derived from a json file for example.
        See help(PrimaryBasis.dump_raw) for formatting.
        """
        self.states = [Primary(Polynomial(*(Monomial(*m) for m in p))) for p in raw_states]

    def dump_raw(self):
        """
        Dump the basis in a raw python list, for writing to a json file for example.
        The list will be formatted as follows:
            [ [ (m.coeff, m.k1, m.k2) for m in O.monomials ] for O in self.states ]
        """
        return [ [ ( m.coeff, m.k1, m.k2 )
                    for m in O.monomials if m.coeff ] for O in self.states ]

    def __getitem__(self, i):
        return self.states[i]

    def _orth(self):
        """Orthonormalize the basis"""
        self.states.sort(key = lambda p: p.conf_dim())

        N = len(self.states)
        graham = np.empty((N,N), dtype=complex)
        symmetric_index = (i for i in range(N**2) if i%N <= i//N)
        for i in symmetric_index:
            j = i % N
            i //= N
            graham[i,j] = Graham(self[i], self[j])
            graham[j,i] = graham[i,j].conjugate()

        w,v = np.linalg.eig(graham)

        zero = 1e-20 # we will forget states that have this norm

        self.states = [
            mono_sum(
                c*self[i] for i,c in enumerate(v[:,j])
            )*(1/sqrt(N2.real)) for j,N2 in enumerate(w) if N2.real > zero
        ]


def Graham(O1, O2):
    """Compute the Graham matrix elements between two primary operators."""
    D1, D2 = O1.conf_dim(), O2.conf_dim()
    r =  1j**(D1-D2)/factorial(D1+D2-1)
    r *= sum(sum( O1[m1]*O2[m2]*(
        factorial(m1.k1 + m2.k1) * factorial(m1.k2 + m2.k2)
    ) for m1 in O1.monomials ) for m2 in O2.monomials )
    return r