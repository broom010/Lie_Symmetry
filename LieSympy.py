from sympy import *
from itertools import *
import numpy as np
import copy as copy
import time

###############################
# Prolongation Formula
###############################

def nJet(X, U, n):
    marker = '%s'*n
    combs = [marker % x for x in combinations_with_replacement(X, n)]
    return [dep + ind for dep in U for ind in combs]
    
def fullJet(X, U, n):
    return X + [x for i in range(n + 1) for x in nJet(X, U, i)]

def uJalpha(u, x, X, U):
    temp = [s for s in U if s in str(u)][0]
    n = len(temp)
    J = nJet(X, U, len(str(u)) - n + 1)
    for i in range(len(J)):
        if (str(u)[:n] == J[i][:n]) and (sorted(str(u)[n:] + str(x)) == sorted(J[i][n:])):
            return var(J[i])
            
def divUJi(X, U, u, J, i):
    f = copy.copy(u)
    for j in range(len(J)):
        f = uJalpha(f, X[J[j]], X, U)
    f = uJalpha(f, X[i], X, U)
    return f
    
def totDiv(X, U, P, i, n):
    expr = diff(P, X[i])
    J = filter(lambda v: v not in X, fullJet(X, U, n))
    for j in range(len(J)):
        expr = expr + diff(P, J[j])*uJalpha(J[j], X[i], X, U)
    return expr

def TotDiv(X, U, P, n, J):
    expr = totDiv(X, U, P, J[0], n)
    for i in range(1, len(J)):
        expr = totDiv(X, U, expr, J[i], n)
    return expr

def phiAlpha(X, U, v, J, q):
    phi = copy.copy(v[len(X):])
    xi = copy.copy(v[:len(X)])
    a = 0
    b = 0
    for i in range(len(X)):
        a = a + xi[i]*uJalpha(U[q], X[i], X, U)
    for i in range(len(X)):
            b = b + xi[i]*divUJi(X, U, U[q], J, i)
    c = TotDiv(X, U, phi[q] - a, len(J), J) + b
    return c

def diffOrd(p, n):
    b = []
    for i in range(1, n + 1):
        b.append(list(combinations_with_replacement(range(p), i)))
    return b

def Prolong(A, v, n):
    C = A[0] + A[1]
    for i in range(1, n + 2):
        C = C + [w[: - i] + '_' + w[-i:] for w in nJet(A[0], A[1], i)]
    sub = dict((symbols(a), symbols(b)) for a, b in zip(fullJet(A[0], A[1], n + 2), C))
    a = copy.copy(v)
    J = diffOrd(len(A[0]), n)
    for i in range(n):
        for j in range(len(A[0])):
            for q in range(len(A[1])):
                c = phiAlpha(A[0], A[1], v, J[i][j], q) + 0*Symbol('x')
                c = c.subs(sub)
                a.append(c)
    return a

def print_vector(v, Jet):
    temp = 0*x
    for i in range(len(v)):
        temp = temp + v[i]*Symbol('\\, \\dfrac{\\partial}{\\partial ' + str(Jet[i]) + '}', commutative=False)
    return temp

############################################
# Dictionary and subset solution Functions
############################################

# Define dictionary merge and reverse dict
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def reverse_dict(D):
    return {v: k for k, v in D.iteritems()}

def findsubsets(S,m):
    return set(combinations(S, m))

def find_solutions(eqs, params):
    sols = []
    for C in findsubsets(eqs, len(params)):
        temp = solve(C, params)
        if len(temp) == len(params):
            sols.append(temp)
    return sols[0]

###############################
# Initialize variables
###############################

# Define variables on the manifold
x = Symbol('x')
dx = Symbol('dx')
Dx = Symbol('D_x')
Ds = Symbol('D_s')

#################################
# List Based Dictionary Creation
#################################

def str_usc(A, f):
    if f in A[0]:
        return f
    else:
        dummy = [s for s in A[1] if s in f][0]
        return dummy + "_"*int(len(f) > len(dummy)) + f[len(dummy):]

def str_sym(A, n):
    B = {}
    for i in range(len(A[1])):
        for j in range(len(A[0])):
            B = merge_two_dicts(B, dict((p, symbols(str_usc(A, p))) for p in fullJet(A[0], A[1], n+1)))
    return B

def str_fun(A, n):
    B = {}
    for i in range(len(A[1])):
        for j in range(n + 1):
            B = merge_two_dicts(B, {A[1][i] + A[0][0]*j: diff(Function(A[1][i])(A[0][0]), A[0][0], j)})
    B = merge_two_dicts(B, {symbols(A[0][0]):A[0][0]})
    return B

def sym_fun(A, n):
    B = {}
    for i in range(len(A[1])):
        for j in range(n + 1):
            B = merge_two_dicts(B, {symbols(str(A[1][i]) + '_'*(1 - int(j==0)) + A[0][0]*j): diff(Function(A[1][i])(A[0][0]), A[0][0], j)}) 
    B = merge_two_dicts(B, {symbols(A[0][0]):A[0][0]})
    return B

def Contact_sym(A, n):
    contact_sym = [0]
    for i in range(n):
        for p in A[1]:
            if i == 0:
                contact_sym.append(symbols('vartheta^' + p))
            else:
                contact_sym.append(symbols('vartheta^' + p + '_' + str(i)))
    return contact_sym

def Contact_sym_nc(A, n):
    contact_sym = [0]
    for i in range(n):
        for p in A[1]:
            if i == 0:
                contact_sym.append(symbols('vartheta^' + p, commutative=False))
            else:
                contact_sym.append(symbols('vartheta^' + p + '_' + str(i), commutative=False))
    return contact_sym

#############################################
# Method - Based Dictionary and List Creation
#############################################

def create_vectors(B, S, T):
    params = B[2]
    r = B[3]

    vectors = []
    for i in range(r):
        vectors.append(vect_Field(B, S, T, params[i]))

    return vectors

def initialize_normals(B):
    A = B[0]
    n = B[1]

    C = ['H']
    for i in range(n + 1):
        for p in A[1]:
            C.append('I^' + str(p) + '_' + str(i))
    return C

def moving_frame(B, S, T):
        params = B[2]
        transform = T[1]
        K = T[2]
        cross_section_indx = T[3]

        C = [transform[cross_section_indx[i]]  -  K[i] for i in range(len(K))]
        C = [c.xreplace(S[5]) for c in C]
        return solve(C, params, dict=True)[0]

def create_normals(B, E, T):
    A_jet = E[1]
    K = T[2]
    cross_section_indx = T[3]

    C = initialize_normals(B)
    normals_substitution = dict((symbols(A_jet[i]), symbols(C[i])) for i in range(len(A_jet)))
    j = 0
    for i in cross_section_indx:
        normals_substitution[symbols(A_jet[i])] = K[j]
        j += 1
    return normals_substitution

def gen_invs(B, S, T):
    A = B[0]
    n = B[1]
    K = T[3]
    normals_substitution = T[9]

    t = len(A[1])
    C1 = []
    C2 = []
    for i in range(t):
        s = A[1][i]
        C3 = [str(p[:len(s)]) + '_'*(1-len(p)==len(s)) + str(p[len(s):]) for p in fullJet(A[0], [A[1][i]], n)]
        y = filter(lambda a: a not in K, [symbols(C3[i]).subs(S[0]).subs(normals_substitution) for i in range(len(C3))])[0]
        C1.append(y)
        start = str(y).index('^')
        stop = str(y).index('_')
        C2.append(symbols('kappa_' + str(y)[start+1:stop]))
    return dict((k, v) for k, v in zip(C1, C2))

def rec_Relations(B, S, T):
    r = B[3]
    Ri = B[4]
    transformed_subs_forward = T[5]
    vectors = T[6]
    Phantoms = T[7]
    normals_substitution = T[9]

    C = []
    for w in Phantoms:
        s = w.subs(transformed_subs_forward)
        C1 = [apply_vect(B, S, vectors[i], s.subs(transformed_subs_forward)).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
        C1.insert(0, diff(s.xreplace(S[3]), x).subs(S[5]).xreplace(normals_substitution))
        C2 = Ri[:]
        C2.insert(0, 1)
        expression = sum([a*b for (a, b) in zip(C1, C2)])
        C.append(expression)
    return find_solutions(C, Ri)

def rec_Relations_Forms(B, S, T, E):
    r = B[3]
    ep = B[5]
    contact_symbol = B[7]
    cross_section_indx = T[3]
    transformed_subs_forward = T[5]
    vectors = T[6]
    Phantoms = T[7]
    normals_substitution = T[9]

    Z = []
    C = [contact_symbol[i] for i in cross_section_indx]
    j = 0
    for w in Phantoms:
        s = w.subs(transformed_subs_forward)
        C1 = [apply_vect(B, S, vectors[i], s.subs(transformed_subs_forward)).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
        C1.insert(0, C[j])
        C2 = ep[:]
        C2.insert(0, 1)
        expression = sum([a*b for (a, b) in zip(C1, C2)])
        Z.append(expression)
        j  += 1
    return find_solutions(Z, ep)

def contact_Reduction(B, S, T):
    A = B[0]
    contact_symbol = B[7]
    contact_zeros = B[8]

    contact_reduction = dict((contact_symbol[i + 3], solve(Ds*contact_symbol[i + 1] - inv_D_x_contact(B, S, T, i, 0), contact_symbol[i + 3])[0]) for i in range(len(A[1])))
    contact_zeros = dict((p, 0) for p in contact_symbol)
    return [contact_reduction, [contact_zeros]]

###############################
# Function Definitions
###############################

def add_Diff_Forms(C, D):
    s = max(len(C), len(D))
    C = C + [0]*(s - len(C))
    D = D + [0]*(s - len(D))
    return [a + b for a, b in zip(C, D)]

def total_diff(S, f):
    var = copy.copy(S[0].keys())
    var.remove('x')
    var.sort()
    expr = diff(f, x) + 0*x
    for i in range(len(var) - 1):
        expr = expr + S[0][var[i + 1]]*diff(f, S[0][var[i]])
    return expr.xreplace(S[5])

def exterior_diff(B, S, f):
    A = B[0]
    n = B[1]

    C = []
    f = f.subs(S[3])
    n = max([ode_order(f, Function(var)) for var in A[1]])
    f = f.subs(S[5])
    var = fullJet(A[0], A[1], n)
    for w in var:
        C.append(diff(f, S[0][w]))
    return C

def vertical_diff(B, S, f):
    A = B[0]
    contact_symbol = B[7]

    C = []
    f = f.subs(S[3])
    n = max([ode_order(f, Function(var)) for var in A[1]])
    var = fullJet(A[0], A[1], n)[1:]
    f = f.subs(S[5])
    for w in var:
        C.append(diff(f, S[0][w]))
    expr = 0*x
    for i in range(len(C)):
        expr = expr + C[i]*contact_symbol[i + 1]
    return expr

def apply_vect(B, S, v, f):
    A = B[0]

    n = max([ode_order(f, Function(var)) for var in A[1]])
    f = f.subs(S[5])
    var = fullJet(A[0], A[1], n)
    vect = Prolong(A, v, n)
    g = 0
    for i in range(len(vect)):
        g = g + diff(f, S[0][var[i]])*v[i]
    return g.subs(S[5])

def vect_Field(B, S, T, y):
    params = B[2]
    m = B[6]
    identity = T[0]
    transform = T[1]

    v = []
    C = transform[:m]
    for i in range(len(C)):
        v.append((diff(C[i], y).subs(dict((params[j], identity[j]) for j in range(len(identity)))) + 0*x).xreplace(S[5]))
    return v

def Dnx(B, E, S, T, f, n):
    A_flat = E[0]
    transform = T[1]
    transformed_subs_backward = T[4]

    for i in range(n):
        f = f.xreplace(transformed_subs_backward).subs(S[3])
        f = (1/(diff(transform[0], A_flat[0]))*diff(f, A_flat[0])).xreplace(S[5])
    return simplify(f.xreplace(S[5]))

def invariantization(S, T, f):
    transformed_subs_backward = T[4]
    frame = T[8]

    f = f.xreplace(S[5]).xreplace(transformed_subs_backward)
    return simplify(f.xreplace(frame))

def DI_x(B, S, T, f):
    r = B[3]
    Ri = B[4]
    vectors = T[1]
    transformed_subs_forward = T[5]
    normals_substitution = T[9]
    mc_invariants = T[11]
    
    s = f.subs(transformed_subs_forward)
    C1 = [apply_vect(B, S, vectors[i], s.subs(transformed_subs_forward)).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
    temp = diff(s.xreplace(S[3]), x)
    temp = temp.xreplace(S[5]).xreplace(normals_substitution)
    C1.insert(0, temp)
    C2 = [mc_invariants[Ri[i]] for i in range(r)]
    C2.insert(0, 1)
    expression = sum([a*b for (a, b) in zip(C1, C2)])
    return expression

def horizontal_diff(S, f):
    return total_diff(S, f)*dx

def Lie_contact_diff(B, S, v, var_index, i):
    A = B[0]
    contact_symbol = B[7]

    n = i + 1
    v_p = Prolong(A, v, n)
    var = fullJet(A[0], A[1], n)
    C = exterior_diff(B, S, v_p[1 + var_index + 2*i])
    D = exterior_diff(B, S, v_p[0])
    U_xi = [S[1][var[var_index + 2*i + 3]]]*len(D)
    D = [-a*b for a, b in zip(D, U_xi)]
    s = max(len(C), len(D))
    C = C + [0]*(s - len(C))
    D = D + [0]*(s - len(D))
    F = [-v_p[var_index + 2*i + 3]] + [0]*(s - 1)
    Contact = [sum(t).subs(S[5]) for t in zip(C, D, F)]
    expr = 0
    for i in range(len(Contact) - 1):
        expr = expr + Contact[i + 1]*contact_symbol[i + 1]
    return expr.xreplace(S[0]).xreplace(S[3]).xreplace(S[5])

def inv_D_x_contact(B, S, T, var_index, i):
    r = B[3]
    Ri = B[4]
    contact_symbol = B[7]
    vectors = T[6]
    normals_substitution = T[9]
    curvature_subs = T[10]
    mc_invariants = T[11]

    expr = 0
    for j in range(r):
        expr = expr + Lie_contact_diff(B, S, vectors[j], var_index, i)*mc_invariants[Ri[j]]
    return (contact_symbol[var_index + 3] + expr).subs(normals_substitution).subs(curvature_subs)

def inv_vert_diff(B, S, T, var_index, i):
    A = B[0]
    r = B[3]
    ep = B[5]
    contact_symbol = B[7]
    vectors = T[6]
    normals_substitution = T[9]
    curvature_subs = T[10]
    rec_forms = T[12]

    expr = contact_symbol[var_index + 2*i + 1]
    C = [simplify((rec_forms[ep[j]])*(Prolong(A, vectors[j], i)[1 + var_index + 2*i])+0*x).xreplace(S[5]).xreplace(normals_substitution).xreplace(curvature_subs) for j in range(r)]
    return expr + sum(C)

def inv_Euler(B, S, T):
    A = B[0]
    contact_symbol = B[7]
    contact_zeros = B[8]
    contact_reduction = T[13]

    C = []
    for i in range(len(A[1])):
        temp = []
        for j in range(len(A[1])):
            temp.append(inv_vert_diff(B, S, T, i, 1).subs(contact_reduction).subs({contact_symbol[j + 1]:1}).subs(contact_zeros).subs({Ds: - Ds}))
        C.append(temp)
    return Matrix(C).T

def invariant_Hamilton(B, S, T):
    A = B[0]
    r = B[3]
    Ri = B[4]
    ep = B[5]
    contact_symbol = B[7]
    contact_zeros = B[8]
    vectors = T[6]
    normals_substitution = T[9]
    curvature_subs = T[10]
    mc_invariants = T[11]
    rec_forms = T[12]

    expr = 0*x
    for i in range(r):
        expr = expr + total_diff(S, vectors[i][0])*rec_forms[ep[i]]
        C = 0*x
        for j in range(len(A[1])):
            C = C  -  mc_invariants[Ri[i]]*diff(vectors[i][0].xreplace(S[5]), S[0][A[1][j]])*contact_symbol[j + 1]
        expr = expr + C
    return Matrix([expr.subs({contact_symbol[j + 1]:1}).subs(contact_zeros).subs({Ds: - Ds}).xreplace(normals_substitution).xreplace(curvature_subs) for j in range(len(A[1]))])

###############################
# Test/Validation Methods
###############################

def Cross_section_validate(B, S, T):
    A = B[0]
    n = B[1]
    cross_section_indx = T[3]
    vectors = T[6]

    vects = []
    vects = [Prolong(A, v, n) for v in vectors]
    temp = []
    for i in range(len(vects)):
        temp.append([vects[i][j] for j in cross_section_indx])
    M = Matrix(temp)
    if min(M.shape) == M.rank() :
        return 1
    else:
        return 0

def Cross_section_determinant(B, S, T):
    A = B[0]
    n = B[1]
    cross_section_indx = T[3]
    vectors = T[6]

    vects = []
    vects = [Prolong(A, v, n) for v in vectors]
    temp = []
    for i in range(len(vects)):
        temp.append([vects[i][j] for j in cross_section_indx])
    M = Matrix(temp)
    return simplify(M.det())

def Cross_section_matrix(B, S, T):
    A = B[0]
    n = B[1]
    cross_section_indx = T[3]
    vectors = T[6]

    vects = []
    vects = [Prolong(A, v, n) for v in vectors]
    temp = []
    for i in range(len(vects)):
        temp.append([vects[i][j] for j in cross_section_indx])
    M = Matrix(temp)
    return M


class groupAction:

    def __init__(self, A, params, n):

        # dictionary/list creation
        self.A_jet = [str_usc(A, p) for p in fullJet(A[0], A[1], n)]
        self.A_jet_no_underscore = fullJet(A[0], A[1], n)
        self.A_flat = [item for sublist in A for item in sublist]
        self.m = len(self.A_flat)
        self.str_to_sym = str_sym(A, n)
        self.sym_to_str = reverse_dict(self.str_to_sym)
        self.str_to_fun = str_fun(A, n)
        self.fun_to_str = reverse_dict(self.str_to_fun)
        self.sym_to_fun = sym_fun(A, n)
        self.fun_to_sym = reverse_dict(self.sym_to_fun)
        self.contact_symbol = Contact_sym(A, n)
        self.contact_zeros = dict((p, 0) for p in self.contact_symbol)
        self.Ri = [symbols('R%d'%k) for k in range(1, len(params) + 1)]
        self.ep = [symbols('varepsilon%d'%k) for k in range(1, len(params) + 1)]

        # master dictionaries/lists that will be used as args in later functions
        self.base = [A, n, params, len(params), self.Ri, self.ep, self.m, self.contact_symbol, self.contact_zeros]
        self.prolonged_base = [self.A_flat, self.A_jet, self.A_jet_no_underscore]
        self.rep_sub = [self.str_to_sym, self.str_to_fun, self.sym_to_str, self.sym_to_fun, self.fun_to_str, self.fun_to_sym]

        # to be modified later
        self.identity = 0            
        self.cross_section_indx = 0
        self.transform = 0            
        self.K = 0
        self.frame = 0
        self.normals_substitution = 0
        self.Phantoms = 0
        self.vectors = 0
        self.mc_invariants = 0
        self.rec_forms = 0
        self.contact_reduction = 0
        self.contact_zeros = 0
        self.transformed_subs_backward = 0
        self.transformed_subs_forward = 0
        self.gen_subs = 0
        self.normals = 0
        self.cross_validate = 0

        # add independent variables and group parameters to the module global variable list
        globals().update({'d' + A[0][0]: symbols('d' + A[0][0]), 'D' + A[0][0]: symbols('D_' + A[0][0])})
        globals().update({'iota':symbols('iota')})
        globals().update(dict((p, symbols(p)) for p in params))
        globals().update(dict((a, symbols(a)) for a in A[0]))
        globals().update(self.rep_sub[0])
        globals().update(dict((p, Function(p)(*A[0])) for p in A[1]))
        globals().update(self.rep_sub[5])
        globals().update(dict((p, symbols(p)) for p in initialize_normals(self.base)))

    def Def_transformation(self, expression, id, cross):
        self.identity = id
        self.transform = expression
        self.K = cross[0]
        B = self.base
        S = self.rep_sub
        E = self.prolonged_base

        A = B[0]
        n = B[1]

        # create cross-section index list
        self.cross_section_indx = []
        for p in cross[1]:
            self.cross_section_indx.append(self.A_jet_no_underscore.index((p.xreplace(S[5])).xreplace(S[2])))

        # Initialized the creation of substitution dictionaries for transformed coordinates
        self.transformed_subs_backward = {symbols(self.A_flat[i]): expression[i].xreplace(S[5])  for i in range(self.m)}
        self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}

        # Create gen_subs (This will serve as the T list that is passed to many of the dictionary creation functions)
        self.gen_subs = [self.identity, self.transform, self.K, self.cross_section_indx, self.transformed_subs_backward, self.transformed_subs_forward]

        # Finish the creation of substitution dictionaries for transformed coordinates
        for i in range(1, n + 1):
            for p in A[1]:
                self.transform.append(Dnx(B, E, S, self.gen_subs, symbols(p), i))
        self.transformed_subs_backward = {symbols(self.A_jet[i]): self.transform[i].xreplace(S[5])  for i in range(len(self.A_jet))}
        self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}

        # Update the transformed substitution dictionaries in gen_subs
        self.gen_subs[-2] = self.transformed_subs_backward
        self.gen_subs[-1] = self.transformed_subs_forward

        # Create gen_subs (This will serve as the T list that is passed to many of the dictionary creation functions)
        self.gen_subs = [self.identity, self.transform, self.K, self.cross_section_indx, self.transformed_subs_backward, self.transformed_subs_forward]

        # Create infinitesimal generators and add them to gen_subs
        self.vectors = create_vectors(self.base, self.rep_sub, self.gen_subs)
        self.gen_subs.append(self.vectors)

        # Preform a cross-section validation test
        # self.cross_validate = 1
        self.cross_validate = self.validate()
        if self.cross_validate == 0:
            print('The chosen cross-section is not valid. Try another cross-section.')

        # Create the Phantom invariant substitution dicitonary and append it to gen_subs
        self.Phantoms = [self.transform[self.cross_section_indx[i]].xreplace(S[5]) for i in range(len(self.K))]
        self.gen_subs.append(self.Phantoms)

    
    def Apply_Frame_Analysis(self):
        if self.cross_validate == 0:
            print('Choose a valid cross-section and/or run Def_transformation().')

        else:
            B = self.base
            E = self.prolonged_base
            S = self.rep_sub

            A = B[0]
            cs = B[7]
            C = initialize_normals(B)
            
            # Create the moving frame dictionary and append it to gen_subs
            self.frame = moving_frame(B, S, self.gen_subs)
            self.gen_subs.append(self.frame)

            # Create a dictionary for substituting the normal invariants if needed
            self.normals = dict((symbols(C[i]), invariantization(S, self.gen_subs, symbols(self.A_jet[i]))) for i in range(len(self.A_jet)))

            # Create the normalized invariant substitution dictionary and append it to gen_subs
            self.normals_substitution = create_normals(B, E, self.gen_subs)
            self.gen_subs.append(self.normals_substitution)

            # Create dictionary for curvature invariant substitution
            self.curvature_subs = gen_invs(B, S, self.gen_subs)
            self.gen_subs.append(self.curvature_subs)        
            
            # Create the dictionary of Maurer-Cartan invariants and add to gen_subs
            self.mc_invariants = rec_Relations(B, S, self.gen_subs)
            self.gen_subs.append(self.mc_invariants)

            # Create invariant contact forms that appear in the recurrence relation for invariant vertical differentiations
            self.rec_forms = rec_Relations_Forms(B, S, self.gen_subs, E)
            self.gen_subs.append(self.rec_forms)

            # Create dictionary for contact reduction and make contact symbols non-commutative
            self.contact_reduction = dict((cs[i + 3], solve(Ds*cs[i + 1] - inv_D_x_contact(B, S, self.gen_subs, i, 0), cs[i + 3])[0]) for i in range(len(A[1])))
            self.gen_subs.append(self.contact_reduction)

    def Prolong(self, v, n):
        return Prolong(self.base[0], v, n)

    def validate(self):
        return Cross_section_validate(self.base, self.rep_sub, self.gen_subs)

    def Cross_section_determinant(self):
        return Cross_section_determinant(self.base, self.rep_sub, self.gen_subs)

    def Cross_section_matrix(self):
        return Cross_section_matrix(self.base, self.rep_sub, self.gen_subs)

    def invariantization(self, f):
        return invariantization(self.rep_sub, self.gen_subs, f)

    def vertical_diff(self, f):
        return vertical_diff(self.base, self.rep_sub, f)

    def horizontal_diff(self, f):
        return horizontal_diff(self.rep_sub, f)

    def Lie_contact_diff(self, v, var_index, i):
        return Lie_contact_diff(self.base, self.rep_sub, v, var_index, i)

    def inv_D_x_contact(self, var_index, i):
        return inv_D_x_contact(self.base, self.rec_forms, self.gen_subs, var_index, i)

    def inv_vert_diff(self, var_index, i):
        return inv_vert_diff(self.base, self.rep_sub, self.gen_subs, var_index, i)

    def inv_Euler(self):
        return inv_Euler(self.base, self.rep_sub, self.gen_subs)

    def invariant_Hamilton(self):
        return invariant_Hamilton(self.base, self.rep_sub, self.gen_subs)