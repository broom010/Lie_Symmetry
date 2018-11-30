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

def Dnx(f, n):
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
    def __init__(self,A,params,identity,transforms,n):
        self.A = A
        self.params = params
        self.m = len(self.A)
        self.r = len(self.params)
        self.identity = identity
        self.transforms = transforms

        self.ind_vars = dict((a, symbols(a)) for a in self.A[0])
        self.dep_vars = self.dep_vars_jet(A,n)
        self.dep_vars_functions = self.dep_vars_functions(A)



        locals().update(self.master_str_to_function)
        locals().update(self.master_str_to_symbol)

    def dep_vars_jet(A,n):
        B = {}
        for i in range(len(A[1])):
            for j in range(len(A[0])):
                B = merge_two_dicts(B, dict((A[1][i]+'_'*(1-int(k == 0))+A[0][j]*k, symbols(A[1][i]+'_'*(1-int(k == 0))+A[0][j]*k)) for k in range(n+1)))
        return B

    def dep_vars_jet_functions(A):
        B = {}
        for i in range(len(A[1])):
            for j in range(len(A[0])):
                B = merge_two_dicts(B, dict((A[1][i]+'_'*(1-int(k == 0))+A[0][j]*k, Function(A[1][i]+'_'*(1-int(k == 0))+A[0][j]*k)) for k in range(n+1)))
        return B

    def dep_vars_functions(A):
        B = {}
        for i in range(len(A[1])):
            for j in range(len(A[0])):
                B = merge_two_dicts(B, {A[1][i]: Function(A[1][j])(*A[0])})
        return B

    # Define a class function that returns a basis for the infinitesimal generators    
    def vect(self):
        v = []
        for j in range(self.r):
            f = []
            for i in range(self.m):
                f.append(diff(self.function[i],self.params[j]).subs(self.params[j],self.identity[i]))
            v.append(f)
        return v
        
    # Define an implicit differentiation operator d/dy
    def implicit(self,f):
        return (1/diff(self.function[0],self.X[0]))*diff(f,self.X[0])
    
    # Define a class function that returns the prolonged infinitesimal generators
    def prolong_Vect(self,n,i):
        return Prolong(self.X,self.U,self.vect()[i],n)
    
    # Define a class function that returns pr v^(n)[f]    
    def my_Op(self,v,j,n,f):
        my_Var = self.X 
        for i in range(n+1):
            my_Var = my_Var + nJet(self.X, self.U, i)
        prV = self.prolong_Vect(n,j)
        F = []
        G = 0
        for i in range(len(my_Var)):
           F.append(prV[i]*diff(f,my_Var[i]))
        for i in range(len(F)):
            G = G+F[i]
        print F
        return G