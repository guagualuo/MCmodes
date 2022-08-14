import sympy

import quicc.recurrence.symbolic_chebyshev as mod

symbolic = mod.SymbolicChebyshev()


def i2r4lapl():
    """Spherical shell 2nd integral of r^4 laplacian operator"""

    # Setup terms in recurrence
    l = sympy.Symbol('l')
    terms = [{'q': 2, 'p': 4, 'd': 2, 'c': 1}, {'q': 2, 'p': 3, 'd': 1, 'c': 2},
             {'q': 2, 'p': 2, 'd': 0, 'c': -l * (l + 1)}]
    terms = symbolic.change_variable(terms, 'linear_r2x')
    r = symbolic.build_recurrence(terms, {0: 1})
    n = sympy.Symbol('n')

    # Print recurrence relation per diagonals
    for k, rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")


def i2r3d1():
    """Spherical shell 2nd integral of r^3 D operator"""

    # Setup terms in recurrence
    l = sympy.Symbol('l')
    terms = [{'q':2, 'p':3, 'd':1, 'c':1}]
    terms = symbolic.change_variable(terms, 'linear_r2x')
    r = symbolic.build_recurrence(terms, {0:1})
    n = sympy.Symbol('n')

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")


def i2r4d1():
    """Spherical shell 2nd integral of r^4 D operator"""

    # Setup terms in recurrence
    l = sympy.Symbol('l')
    terms = [{'q':2, 'p':4, 'd':1, 'c':1}]
    terms = symbolic.change_variable(terms, 'linear_r2x')
    r = symbolic.build_recurrence(terms, {0:1})
    n = sympy.Symbol('n')

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")
