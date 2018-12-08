from sympy import *
from itertools import *
import numpy as np
import copy as copy

###############################
# Prolongation Formula
###############################

def nJet(X,U,n):
    marker = '%s'*n
    combs = [marker % x for x in combinations_with_replacement(X,n)]
    return [dep+ind for dep in U for ind in combs]
    
def fullJet(X,U,n):
    return X+[x for i in range(n+1) for x in nJet(X,U,i)]

def uJalpha(u,x,X,U):
    J = nJet(X,U,len(str(u)))
    for i in range(len(J)):
        if sorted(str(u)+str(x)) == sorted(J[i]):
            return var(J[i])
            break
            
def divUJi(X,U,u,J,i):
    f = copy.copy(u)
    for j in range(len(J)):
        f = uJalpha(f,X[J[j]],X,U)
    f = uJalpha(f,X[i],X,U)
    return f
    
def totDiv(X,U,P,i,n):
    expr = diff(P,X[i])
    J = filter(lambda v: v not in X, fullJet(X,U,n))
    for j in range(len(J)):
        expr = expr + diff(P,J[j])*uJalpha(J[j],X[i],X,U)
    return expr

def TotDiv(X,U,P,n,J):
    expr = totDiv(X,U,P,J[0],n)
    for i in range(1,len(J)):
        expr = totDiv(X,U,expr,J[i],n)
    return expr

def phiAlpha(X,U,v,J,q):
    phi = copy.copy(v[len(X):])
    xi = copy.copy(v[:len(X)])
    a = 0
    b = 0
    for i in range(len(X)):
        a = a+xi[i]*uJalpha(U[q],X[i],X,U)
    for i in range(len(X)):
            b = b+xi[i]*divUJi(X,U,U[q],J,i)
    c = TotDiv(X,U,phi[q]-a,len(J),J)+b
    return c

def diffOrd(p,n):
    b = []
    for i in range(1,n+1):
        b.append(list(combinations_with_replacement(range(p),i)))
    return b

def Prolong(X,U,v,n):
    a = copy.copy(v)
    J = diffOrd(len(X),n)
    for i in range(n):
        for j in range(len(X)):
            for q in range(len(U)):
                c = phiAlpha(X,U,v,J[i][j],q)+0*Symbol('x')
                c = c.subs(master_str_to_symbol)
                a.append(c)
    return a

####################################
# Functions
####################################

# Define a function that returns the infinitesimal generators for a group action
# A is an array that contains expressions for how the coordinates transform
# param is the group parameter that should be used in the differentiation
def vect_Field(A, param):
    v = []
    for i in range(len(A)):
        v.append(diff(A[i],param).subs({param:0}))
    return v

def apply_vect(v,f,A):
    n = max([ode_order(f,Function(var)) for var in A[1]])
    f = f.subs(master_function_to_symbol)
    var = fullJet(A[0],A[1],n)
    v = Prolong(A[0],A[1], v, n)
    g = 0
    for i in range(len(v)):
        g = g + diff(f,master_str_to_symbol[var[i]])*v[i]
    return g.subs(master_function_to_symbol)

def Dx(f):
    f = f.xreplace(transformed_subs_backward).subs(reverse_dict(master_function_to_symbol))
    return 1/diff(expr_X.subs(reverse_dict(master_function_to_symbol)),x)*diff(f,x)

def normalized_invariant(f, n):
    f = Dx(f)
    for i in range(n-1):
        f = 1/diff(expr_X.subs(reverse_dict(master_function_to_symbol)),x)*diff(f,x)
    return simplify(f).subs(master_function_to_symbol)

# Create a moving frame dictionary to replace group parameters
def moving_frame(A,K):
    B = [A[i] - K[i] for i in range(len(A))] 
    B = [b.subs(master_function_to_symbol) for b in B]
    return solve(B)[0]

def invariantization(f,frame):
    f = f.xreplace(transformed_subs_backward)
    return simplify(f.subs(frame)).subs(master_function_to_symbol)

# Return the Maurer-Cartan invariants
def rec_Relations(Phantoms, frame):
    B = []
    
    for w in Phantoms:
        s = w.subs(transformed_subs_forward)
        w1 = apply_vect(v1,s.subs(transformed_subs_forward),A)
        w2 = apply_vect(v2,s.subs(transformed_subs_forward),A)
        w3 = apply_vect(v3,s.subs(transformed_subs_forward),A)
        expression = invariantization(diff(s.subs(reverse_dict(master_function_to_symbol)),x).subs(master_function_to_symbol),frame) + R1*invariantization(w1,frame) + R2*invariantization(w2,frame) + R3*invariantization(w3,frame)
        B.append(expression)
    return solve(B,[R1,R2,R3])

def DI_x(f, mc_invariants):
    s = f.subs(transformed_subs_forward)
    w1 = apply_vect(v1,s.subs(transformed_subs_forward),A)
    w2 = apply_vect(v2,s.subs(transformed_subs_forward),A)
    w3 = apply_vect(v3,s.subs(transformed_subs_forward),A)
    c1 = mc_invariants[R1]
    c2 = mc_invariants[R2]
    c3 = mc_invariants[R3]
    print(invariantization(diff(f.subs(master_str_to_function),x).subs(master_function_to_str),frame))
    expression = invariantization(diff(f.subs(master_str_to_function),x).subs(master_function_to_symbol),frame) + c1*invariantization(w1,frame) + c2*invariantization(w2,frame) + c3*invariantization(w3,frame)
    return expression

def exterior_diff(f,A):
    C = []
    f = f.subs(master_symbol_to_function)
    n = max([ode_order(f,Function(var)) for var in A[1]])
    f = f.subs(master_function_to_symbol)
    var = fullJet(A[0],A[1],n)
    for w in var:
        C.append(diff(f,master_str_to_symbol[w]))
    return C

def add_Diff_Forms(C,D):
    s = max(len(C),len(D))
    C = C+[0]*(s-len(C))
    D = D+[0]*(s-len(D))
    return [a+b for a,b in zip(C,D)]

def total_diff(f):
    var = copy.copy(master_str_to_symbol.keys())
    var.remove('x')
    var.sort()
    expr = diff(f,x)+0*x
    for i in range(len(var)-1):
        expr = expr + master_str_to_symbol[var[i+1]]*diff(f,master_str_to_symbol[var[i]])
    return expr

def vertical_diff(f):
    C = []
    f = f.subs(master_symbol_to_function)
    n = max([ode_order(f,Function(var)) for var in A[1]])
    var = fullJet(A[0],A[1],n)[1:]
    f = f.subs(master_function_to_symbol)
    for w in var:
        C.append(diff(f,master_str_to_symbol[w]))
    expr = 0*x
    for i in range(len(C)):
        expr = expr + C[i]*contact_symbol[i]
    return expr

def horizontal_diff(f):
    return total_diff(f)*dx

def rec_Relations_Forms(Phantoms, frame, A):
    B = []
    C = [0, contact_symbol[0], contact_symbol[1]]
    for i in range(len(Phantoms)):
        w = Phantoms[i]
        s = w.subs(transformed_subs_forward)
        w1 = apply_vect(v1,s.subs(transformed_subs_forward),A)
        w2 = apply_vect(v2,s.subs(transformed_subs_forward),A)
        w3 = apply_vect(v3,s.subs(transformed_subs_forward),A)
        expression = C[i] + epsilon_1*invariantization(w1,frame) + epsilon_2*invariantization(w2,frame) + epsilon_3*invariantization(w3,frame)
        B.append(expression)
    return solve(B,[epsilon_1,epsilon_2,epsilon_3])

def inv_D_x_contact(var_index, i):
    w1 = invariantization(Lie_contact_diff(v1, var_index, i),frame)
    w2 = invariantization(Lie_contact_diff(v2, var_index, i),frame)
    w3 = invariantization(Lie_contact_diff(v3, var_index, i),frame)
    return contact_symbol[var_index+2] + mc_invariants[R1]*w1+ mc_invariants[R2]*w2+ mc_invariants[R3]*w3

def invariant_vert_diff(var_index, i):
    expr = contact_symbol[var_index+2*i]
    w1 = rec_forms[epsilon_1]*invariantization(Prolong(A[0],A[1],v1,i)[1+var_index+2*i],frame)
    w2 = rec_forms[epsilon_2]*invariantization(Prolong(A[0],A[1],v2,i)[1+var_index+2*i],frame)
    w3 = rec_forms[epsilon_3]*invariantization(Prolong(A[0],A[1],v3,i)[1+var_index+2*i],frame)
    expr = expr + w1 + w2 + w3
    return expr

def invariant_Euler():
    a11 = invariant_vert_diff(0,1).subs(contact_reduction).subs({contact_symbol[0]:1,contact_symbol[1]:0}).subs({D_x:-D_x})
    a21 = invariant_vert_diff(0,1).subs(contact_reduction).subs({contact_symbol[0]:0,contact_symbol[1]:1}).subs({D_x:-D_x})
    a12 = invariant_vert_diff(1,1).subs(contact_reduction).subs({contact_symbol[0]:1,contact_symbol[1]:0}).subs({D_x:-D_x})
    a22 = invariant_vert_diff(1,1).subs(contact_reduction).subs({contact_symbol[0]:0,contact_symbol[1]:1}).subs({D_x:-D_x})
    return Matrix([[a11,a12],[a21,a22]])

def invariant_Hamilton():
    r = [R1,R2,R3]
    ep = [epsilon_1,epsilon_2,epsilon_3]
    vectors = [v1,v2,v3]
    w1 = 0*x
    w2 = 0*x
    for i in range(3):
        w1 = w1-mc_invariants[r[i]]*diff(vectors[i][0],u)*contact_symbol[0]+total_diff(vectors[i][0])*rec_forms[ep[i]]
        w2 = w2-mc_invariants[r[i]]*diff(vectors[i][0],P)*contact_symbol[1]
    b1 = (w1+w2).subs({contact_symbol[0]:1, contact_symbol[1]:0}).subs({D_x:-D_x})
    b2 = (w1+w2).subs({contact_symbol[0]:0, contact_symbol[1]:1}).subs({D_x:-D_x})
    return Matrix([[b1],[b2]])

def Euler(L,var_index):
    var = ['u','P']
    var_sub = [var[var_index]+'x'*k for k in range(1,3)]
    expr = 0*x
    for i in range(len(var_sub)):
        expr = expr + (-1)**i*diff(diff(L,master_str_to_symbol[var_sub[i]]).subs(master_symbol_to_function),x,i)
    return expr.subs(master_function_to_symbol)

def inv_Euler_Lagrange(L):
    inv_A = invariant_Euler()
    inv_B = invariant_Hamilton()
    expr1 = (inv_A[0,0]*Euler(L,0)+inv_A[0,1]*Euler(L,1)-L*inv_B[0,0]).subs({D_x:0})
    expr2 = (inv_A[1,0]*Euler(L,0)+inv_A[1,1]*Euler(L,1)-L*inv_B[1,0]).subs({D_x:0})
    return solve(Matrix([[expr1],[expr2]]))

###############################
# Initialize variables
###############################

# Define variables on the manifold
x = Symbol('x')
dx = Symbol('dx')
u = Function('u')(x)
varphi = Function('varphi')(x)
P = Function('P')(x)

# Define derivatives of u
n = 6
u_str_to_symbol = dict(('u'+'x'*k, symbols('u'+'_'*(1-int(k == 0))+'x'*k)) for k in range(n))
a = u_str_to_symbol.keys()
a.sort()
u_str_to_function = dict(zip(a, [diff(u,x,k) for k in range(n)]))
u_function_to_symbol = dict((diff(u,x,k), symbols('u'+'_'*(1-int(k == 0))+'x'*k)) for k in range(n))

# Define dictionary merge and reverse dict
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def reverse_dict(D):
    return {v: k for k, v in D.iteritems()}

# Define derivatives of varphi
phi_str_to_symbol = dict(('P'+'x'*k, symbols('varphi'+'_'*(1-int(k == 0))+'x'*k)) for k in range(n))
a = phi_str_to_symbol.keys()
a.sort()
phi_str_to_function = dict(zip(a, [diff(varphi,x,k) for k in range(n)]))
phi_function_to_symbol = dict((diff(varphi,x,k), symbols('varphi'+'_'*(1-int(k == 0))+'x'*k)) for k in range(n))

# Create master dictionaries for going between different representations
master_str_to_symbol = merge_two_dicts({'x': symbols('x')}, merge_two_dicts(u_str_to_symbol, phi_str_to_symbol))
master_str_to_function = merge_two_dicts({'x': symbols('x')}, merge_two_dicts(u_str_to_function, phi_str_to_function))
master_function_to_str = reverse_dict(master_str_to_function)
master_function_to_symbol = merge_two_dicts({'x': symbols('x')}, merge_two_dicts(u_function_to_symbol, phi_function_to_symbol))
master_symbol_to_function = reverse_dict(master_function_to_symbol)

locals().update(master_str_to_function)
locals().update(master_str_to_symbol)

# Group parameters
a = Symbol('a')
b = Symbol('b')
psi = Symbol('psi')

# Maurer-Cartan Invariants
R1 = Symbol('R1')
R2 = Symbol('R2')
R3 = Symbol('R3')

# Recurrence relation invariant-contacts
contact_symbol = [Symbol('vartheta^u'),Symbol('vartheta^varphi'),Symbol('vartheta^u_1'),Symbol('vartheta^varphi_1'),Symbol('vartheta^u_2'),Symbol('vartheta^varphi_2')]

epsilon_1 = Symbol('varepsilon_1')
epsilon_2 = Symbol('varepsilon_2')
epsilon_3 = Symbol('varepsilon_3')

# Define Group action
A = [['x'],['u','P']]

expr_X = cos(psi)*x-sin(psi)*u+a
expr_U = sin(psi)*x+cos(psi)*u+b
expr_Varphi = varphi+psi

Phantoms = [expr_X, expr_U, expr_Varphi]

transformed_subs_backward = {x:expr_X, u:expr_U, P:expr_Varphi}
transformed_subs_forward = {v: k for k, v in transformed_subs_backward.iteritems()}

v1 = vect_Field(Phantoms, a)
v2 = vect_Field(Phantoms, b)
v3 = vect_Field(Phantoms, psi)

# Define a class that will serve to define group actions. We will develop several class functions that
# will make working with group actions easy.
class groupAction:

    def __init__(self, A, params, n):

        # Define a few basic pieces of information about the problem
        self.A = A
        self.A_flat = [item for sublist in A for item in sublist]
        self.n = n
        self.params = params
        self.m = len(A)
        self.r = len(params)
        self.identity = 0             # to be modified later
        self.transform = 0            # to be modified later
        self.K = 0
        self.frame = 0

        self.master_str_to_symbol = self.M_str_sym()
        self.master_symbol_to_function = self.M_sym_fun()
        self.master_function_to_symbol = reverse_dict(self.master_symbol_to_function)
        self.contact_symbols = self.Contact_sym()
        self.Ri = [symbols('R%d'%k) for k in range(1,self.r + 1)]
        self.transformed_subs_backward = 0
        self.transformed_subs_forward = 0

        # add independent variables and group parameters to the module global variable list
        globals().update(dict((p, symbols(p)) for p in params))
        globals().update(dict((a, symbols(a)) for a in A[0]))
        globals().update(self.master_str_to_symbol)
        globals().update(dict((p, Function(p)(*A[0])) for p in A[1]))
        globals().update(self.master_function_to_symbol)
        globals().update(dict(('R%d'%k, symbols('R%d'%k)) for k in range(1,self.r + 1)))

    def Def_transformation(self, expression, id, cross):
        self.transform = expression
        self.identity = id
        self.K = cross
        self.transformed_subs_backward = {symbols(self.A_flat[i]): expression[i].xreplace(self.master_function_to_symbol)  for i in range(len(self.A_flat))}
        self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}

    def M_str_sym(self):
        B = {}
        for i in range(len(self.A[1])):
            for j in range(len(self.A[0])):
                B = merge_two_dicts(B, dict((p, symbols(str(p[0])+'_'*(1-int(len(p)==1))+str(p[1:]))) for p in fullJet(self.A[0],self.A[1],self.n)))
        return B

    def Dep_functions(self):
        B = {}
        for i in range(len(self.A[1])):
            for j in range(self.A[0]):
                B = merge_two_dicts(B, {self.A[1][i]: Function(self.A[1][j])(*self.A[0])})
        return B

    def M_sym_fun(self):
        B = {}
        for i in range(len(self.A[1])):
            for j in range(self.n+1):
                B = merge_two_dicts(B, {symbols(str(self.A[1][i])+'_'*(1-int(j==0))+self.A[0][0]*j): diff(Function(self.A[1][i])(self.A[0][0]),self.A[0][0], j)}) 
        B = merge_two_dicts(B,{symbols(self.A[0][0]):self.A[0][0]})
        return B

    def Contact_sym(self):
        contact_sym = []
        for i in range(self.n):
            for p in self.A[1]:
                contact_sym.append(Symbol('vartheta^'+p+'_'+str(i)))
        return contact_sym

    def vect_Field(self):
        v = []
        for i in range(len(self.A)):
            v.append(diff(self.A[i],self.param).subs({self.param:0}))
        return v

    def apply_vect(self, v,f):
        n = max([ode_order(f,Function(var)) for var in self.A[1]])
        f = f.subs(self.master_function_to_symbol)
        var = fullJet(self.A[0],self.A[1],n)
        v = Prolong(self.A[0],self.A[1], v, n)
        g = 0
        for i in range(len(v)):
            g = g + diff(f,self.master_str_to_symbol[var[i]])*v[i]
        return g.subs(self.master_function_to_symbol)

    def Dnx(self, f, n):
        for i in range(n):
            f = f.xreplace(self.transformed_subs_backward).subs(self.master_symbol_to_function)
            f = (1/(diff(self.transform[0],self.A_flat[0]))*diff(f,self.A_flat[0])).xreplace(self.master_function_to_symbol)
        return simplify(f.xreplace(self.frame).xreplace(self.master_function_to_symbol))

    # Create a moving frame dictionary to replace group parameters
    def moving_frame(self):
        B = [self.transform[i] - self.K[i] for i in range(len(self.transform))]
        B = [b.xreplace(self.master_function_to_symbol) for b in B]
        self.frame =  solve(B,self.params, dict=True)[0]

    def invariantization(f,frame):
        f = f.xreplace(transformed_subs_backward)
        return simplify(f.subs(frame)).subs(self.master_function_to_symbol)

    def normalized_invariant(U,n,frame):
        f = Dnx(U,n)
        return simplify(f.subs(frame)).subs(master_function_to_symbol)

    # Return the Maurer-Cartan invariants
    def rec_Relations(Phantoms, frame):
        B = []
        
        for w in Phantoms:
            s = w.subs(transformed_subs_forward)
            w1 = apply_vect(v1,s.subs(transformed_subs_forward),A)
            w2 = apply_vect(v2,s.subs(transformed_subs_forward),A)
            w3 = apply_vect(v3,s.subs(transformed_subs_forward),A)
            expression = invariantization(diff(s.subs(reverse_dict(master_function_to_symbol)),x).subs(master_function_to_symbol),frame) + R1*invariantization(w1,frame) + R2*invariantization(w2,frame) + R3*invariantization(w3,frame)
            B.append(expression)
        return solve(B,[R1,R2,R3])