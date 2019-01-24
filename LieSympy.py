from sympy import *
from itertools import *
import numpy as np
import copy as copy
import time

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

####################################
# Functions
####################################

# Define dictionary merge and reverse dict
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def reverse_dict(D):
    return {v: k for k, v in D.iteritems()}

###############################
# Initialize variables
###############################

# # Define variables on the manifold
dx = Symbol('dx')
Dx = Symbol('D_x')

# Define a class that will serve to define group actions. We will develop several class functions that
# will make working with group actions easy.
class groupAction:

    def __init__(self, A, params, n):

        # Define a few basic pieces of information about the problem
        self.A = A
        self.A_jet = [str(p[0])+'_'*(1-int(len(p)==1))+str(p[1:]) for p in fullJet(A[0],A[1],n)]
        self.A_jet_no_underscore = fullJet(A[0],A[1],n)
        self.A_flat = [item for sublist in A for item in sublist]
        self.n = n
        self.m = len(self.A_flat)
        self.params = params
        self.r = len(params)
        self.identity = 0             # to be modified later
        self.cross_section_indx = 0
        self.transform = 0            # to be modified later
        self.K = 0
        self.frame = 0
        self.normals = 0
        self.normals_substitution = 0
        self.Phantoms = 0
        self.vectors = 0
        self.mc_invariants = 0
        self.rec_forms = 0
        self.contact_reduction = 0
        self.contact_zeros = 0

        self.master_str_to_symbol = self.M_str_sym()
        self.master_symbol_to_str = reverse_dict(self.master_str_to_symbol)
        self.master_str_to_function = self.M_str_fun()
        self.master_function_to_str = reverse_dict(self.master_str_to_function)
        self.master_symbol_to_function = self.M_sym_fun()
        self.master_function_to_symbol = reverse_dict(self.master_symbol_to_function)

        self.contact_symbol = self.Contact_sym()
        self.Ri = [symbols('R%d'%k) for k in range(1,self.r + 1)]
        self.ep = [symbols('varepsilon%d'%k) for k in range(1,self.r + 1)]
        self.transformed_subs_backward = 0
        self.transformed_subs_forward = 0

        # add independent variables and group parameters to the module global variable list
        globals().update({'d'+self.A[0][0]: symbols('d'+self.A[0][0]), 'D'+self.A[0][0]: symbols('D_'+self.A[0][0])})
        globals().update({'iota':symbols('iota')})
        globals().update(dict((p, symbols(p)) for p in params))
        globals().update(dict((a, symbols(a)) for a in A[0]))
        globals().update(self.master_str_to_symbol)
        globals().update(dict((p, Function(p)(*A[0])) for p in A[1]))
        globals().update(self.master_function_to_symbol)
        globals().update(dict((p, symbols(p)) for p in self.initialize_normals()))
        # globals().update(dict(('R%d'%k, symbols('R%d'%k)) for k in range(1,self.r + 1)))


    def Def_transformation(self, expression, id, cross):
        t0 = time.time()
        self.transform = expression
        self.identity = id
        self.K = cross[0]
        self.cross_section_indx = []
        print(time.time()-t0)
        for p in cross[1]:
            self.cross_section_indx.append(self.A_jet_no_underscore.index((p.xreplace(self.master_function_to_symbol)).xreplace(self.master_symbol_to_str)))
        print(time.time()-t0)
        self.transformed_subs_backward = {symbols(self.A_flat[i]): expression[i].xreplace(self.master_function_to_symbol)  for i in range(self.m)}
        print(time.time()-t0)
        self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}
        print(time.time()-t0)
        print(self.n+1)
        print(len(self.A[1]))
        for i in range(1,self.n+1):
            for p in self.A[1]:
                print('pass')
                self.transform.append(self.Dnx(symbols(p),i))
        print(time.time()-t0)
        self.transformed_subs_backward = {symbols(self.A_jet[i]): self.transform[i].xreplace(self.master_function_to_symbol)  for i in range(len(self.A_jet))}
        print(time.time()-t0)
        self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}
        print(time.time()-t0)
        self.create_vectors()
        print(self.vectors)
        self.moving_frame()
        print(time.time()-t0)
        C = self.initialize_normals()
        print(time.time()-t0)
        self.normals = dict((symbols(C[i]), self.invariantization(symbols(self.A_jet[i]))) for i in range(len(self.A_jet)))
        print(time.time()-t0)
        self.normals_substitution = dict((symbols(self.A_jet[i]), symbols(C[i])) for i in range(len(self.A_jet)))
        print(time.time()-t0)
        for i in self.cross_section_indx:
            self.normals_substitution[symbols(self.A_jet[i])] = 0
        print(time.time()-t0)
        self.Phantoms = [self.transform[self.cross_section_indx[i]].xreplace(self.master_function_to_symbol) for i in range(len(self.K))]
        print(time.time()-t0)
        self.rec_Relations()
        self.rec_Relations_Forms()
        self.gen_invs()
        self.contact_Reduction()
        # contact_reduction = {contact_symbol[2]:solve(D_x*contact_symbol[0]-inv_D_x_contact(0,0), contact_symbol[2])[0],contact_symbol[3]:solve(D_x*contact_symbol[1]-inv_D_x_contact(1,0), contact_symbol[3])[0]}

    def initialize_normals(self):
        C = ['H']
        for i in range(self.n+1):
            for p in self.A[1]:
                C.append('I^'+str(p)+'_'+str(i))
        return C

    def gen_invs(self):
        t = len(self.A[1])
        C1 = []
        C2 = []
        for i in range(t):
            C3 = [str(p[0])+'_'*(1-int(len(p)==1))+str(p[1:]) for p in fullJet(self.A[0],self.A[1][i],self.n)]
            y = np.trim_zeros([symbols(C3[i]).subs(self.normals_substitution) for i in range(len(C3))])[0]
            C1.append(y)
            C2.append(symbols('kappa^'+str(y)[2]))
        self.curvature_subs = dict((k,v) for k, v in zip(C1,C2))

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

    def M_str_fun(self):
        B = {}
        for i in range(len(self.A[1])):
            for j in range(self.n+1):
                B = merge_two_dicts(B, {self.A[1][i]+self.A[0][0]*j: diff(Function(self.A[1][i])(self.A[0][0]),self.A[0][0], j)})
        B = merge_two_dicts(B,{symbols(self.A[0][0]):self.A[0][0]})
        return B

    def Contact_sym(self):
        contact_sym = [0]
        for i in range(self.n):
            for p in self.A[1]:
                if i == 0:
                    contact_sym.append(Symbol('vartheta^'+p))
                else:
                    contact_sym.append(Symbol('vartheta^'+p+'_'+str(i)))
        return contact_sym

    def contact_Reduction(self):
        self.contact_reduction = dict((self.contact_symbol[i+3], solve(Dx*self.contact_symbol[i+1]-self.inv_D_x_contact(i,0), self.contact_symbol[i+3])[0]) for i in range(len(self.A[1])))
        self.contact_zeros = dict((p,0) for p in self.contact_symbol)

    def vect_Field(self, y):
        v = []
        C = self.transform[:self.m]
        for i in range(len(C)):
            v.append(diff(C[i],y).subs(dict((self.params[j],self.identity[j]) for j in range(len(self.identity)))))
        return v
    
    def create_vectors(self):
        self.vectors = []
        for i in range(self.r):
            self.vectors.append(self.vect_Field(self.params[i]))

    def apply_vect(self, v,f):
        n = max([ode_order(f,Function(var)) for var in self.A[1]])
        f = f.subs(self.master_function_to_symbol)
        var = fullJet(self.A[0],self.A[1],n)
        vect = self.Prolong(v,n)
        g = 0
        for i in range(len(vect)):
            g = g + diff(f,self.master_str_to_symbol[var[i]])*v[i]
        return g.subs(self.master_function_to_symbol)

    def Dnx(self, f, n):
        for i in range(n):
            f = f.xreplace(self.transformed_subs_backward).subs(self.master_symbol_to_function)
            f = (1/(diff(self.transform[0],self.A_flat[0]))*diff(f,self.A_flat[0])).xreplace(self.master_function_to_symbol)
        return simplify(f.xreplace(self.master_function_to_symbol))

    def normalized_invariant(self, f, n):
        for i in range(n):
            f = f.xreplace(self.transformed_subs_backward).subs(self.master_symbol_to_function)
            f = (1/(diff(self.transform[0],self.A_flat[0]))*diff(f,self.A_flat[0])).xreplace(self.master_function_to_symbol)
        return simplify(f.xreplace(self.frame).xreplace(self.master_function_to_symbol))

    def normalize(self):
        self.normals = ['I']
        for i in range(self.n):
            for p in self.A[1]:
                self.normals.append('H^'+str(p)+'_'+str(i))

    # Create a moving frame dictionary to replace group parameters
    
    def moving_frame(self):
        B = [self.transform[self.cross_section_indx[i]] - self.K[i] for i in range(len(self.K))]
        print('pass')
        B = [b.xreplace(self.master_function_to_symbol) for b in B]
        print('pass')
        print(B)
        self.frame =  solve(B,self.params, dict=True)[0]
        print('pass')

    def Prolong(self,v,n):
        a = copy.copy(v)
        J = diffOrd(len(self.A[0]),n)
        for i in range(n):
            for j in range(len(self.A[0])):
                for q in range(len(self.A[1])):
                    c = phiAlpha(self.A[0],self.A[1],v,J[i][j],q)+0*Symbol('x')
                    c = c.subs(self.master_str_to_symbol)
                    a.append(c)
        return a

    def invariantization(self,f):
        f = f.xreplace(self.master_function_to_symbol).xreplace(self.transformed_subs_backward)
        return simplify(f.xreplace(self.frame))

    # Return the Maurer-Cartan invariants
    def rec_Relations(self):
        B = []
        for w in self.Phantoms:
            s = w.subs(self.transformed_subs_forward)
            C1 = [self.apply_vect(self.vectors[i],s.subs(self.transformed_subs_forward)).xreplace(self.master_function_to_symbol).xreplace(self.normals_substitution) for i in range(self.r)]
            C1.insert(0,diff(s.xreplace(self.master_symbol_to_function),x).subs(self.master_function_to_symbol).xreplace(self.normals_substitution))
            C2 = self.Ri[:]
            C2.insert(0,1)
            expression = sum([a*b for (a,b) in zip(C1,C2)])
            B.append(expression)
        self.mc_invariants = solve(B, self.Ri)

    def DI_x(self,f):
        s = f.subs(self.transformed_subs_forward)
        C1 = [self.apply_vect(self.vectors[i],s.subs(self.transformed_subs_forward)).xreplace(self.master_function_to_symbol).xreplace(self.normals_substitution) for i in range(self.r)]
        temp = diff(s.xreplace(self.master_symbol_to_function),x)
        temp = temp.xreplace(self.master_function_to_symbol).xreplace(self.normals_substitution)
        C1.insert(0, temp)
        C2 = [self.mc_invariants[self.Ri[i]] for i in range(self.r)]
        C2.insert(0,1)
        expression = sum([a*b for (a,b) in zip(C1,C2)])
        return expression

    def exterior_diff(self,f):
        C = []
        f = f.subs(self.master_symbol_to_function)
        n = max([ode_order(f,Function(var)) for var in self.A[1]])
        f = f.subs(self.master_function_to_symbol)
        var = fullJet(self.A[0],self.A[1],n)
        for w in var:
            C.append(diff(f,self.master_str_to_symbol[w]))
        return C

    def add_Diff_Forms(self,C,D):
        s = max(len(C),len(D))
        C = C+[0]*(s-len(C))
        D = D+[0]*(s-len(D))
        return [a+b for a,b in zip(C,D)]

    def total_diff(self,f):
        var = copy.copy(self.master_str_to_symbol.keys())
        var.remove('x')
        var.sort()
        expr = diff(f,x)+0*x
        for i in range(len(var)-1):
            expr = expr + self.master_str_to_symbol[var[i+1]]*diff(f,self.master_str_to_symbol[var[i]])
        return expr.xreplace(self.master_function_to_symbol)

    def vertical_diff(self,f):
        C = []
        f = f.subs(self.master_symbol_to_function)
        n = max([ode_order(f,Function(var)) for var in self.A[1]])
        var = fullJet(self.A[0],self.A[1],n)[1:]
        f = f.subs(self.master_function_to_symbol)
        for w in var:
            C.append(diff(f,self.master_str_to_symbol[w]))
        expr = 0*x
        for i in range(len(C)):
            expr = expr + C[i]*self.contact_symbol[i+1]
        return expr

    def horizontal_diff(f):
        return total_diff(f)*dx

    def rec_Relations_Forms(self):
        B = []
        C = [self.contact_symbol[i] for i in self.cross_section_indx]
        j = 0
        for w in self.Phantoms:
            s = w.subs(self.transformed_subs_forward)
            C1 = [self.apply_vect(self.vectors[i],s.subs(self.transformed_subs_forward)).xreplace(self.master_function_to_symbol).xreplace(self.normals_substitution) for i in range(self.r)]
            C1.insert(0,C[j])
            C2 = self.ep[:]
            C2.insert(0,1)
            expression = sum([a*b for (a,b) in zip(C1,C2)])
            B.append(expression)
            j += 1
        self.rec_forms = solve(B,self.ep)

    def Lie_contact_diff(self, v, var_index, i):
        n = i+1
        v_p = self.Prolong(v,n)
        var = fullJet(self.A[0],self.A[1],n)
        C = self.exterior_diff(v_p[1+var_index+2*i])
        D = self.exterior_diff(v_p[0])
        U_xi = [self.master_str_to_function[var[var_index+2*i+3]]]*len(D)
        D = [-a*b for a,b in zip(D,U_xi)]
        s = max(len(C),len(D))
        C = C + [0]*(s-len(C))
        D = D + [0]*(s-len(D))
        F = [-v_p[var_index+2*i+3]]+[0]*(s-1)
        Contact = [sum(t).subs(self.master_function_to_str) for t in zip(C,D,F)]
        expr = 0
        for i in range(len(Contact)-1):
            expr = expr + Contact[i+1]*self.contact_symbol[i+1]
        return expr.subs(self.master_str_to_symbol)

    def inv_D_x_contact(self,var_index,i):
        expr = 0
        for j in range(self.r):
            expr = expr + self.Lie_contact_diff(self.vectors[j], var_index, i)*self.mc_invariants[self.Ri[j]]
        return (self.contact_symbol[var_index + 3] + expr).subs(self.normals_substitution).subs(self.curvature_subs)

    def inv_vert_diff(self,var_index,i):
        expr = self.contact_symbol[var_index+2*i+1]
        C = [(self.rec_forms[self.ep[j]])*(self.Prolong(self.vectors[j],i)[1+var_index+2*i]).xreplace(self.master_function_to_symbol).xreplace(self.normals_substitution).xreplace(self.curvature_subs) for j in range(self.r)]
        return expr + sum(C)

    def inv_Euler(self):
        C = []
        for i in range(len(self.A[1])):
            temp = []
            for j in range(len(self.A[1])):
                temp.append(self.inv_vert_diff(i,1).subs(self.contact_reduction).subs({self.contact_symbol[j+1]:1}).subs(self.contact_zeros).subs({Dx:-Dx}))
            C.append(temp)
        return Matrix(C).T

    def invariant_Hamilton(self):
        expr = 0*x
        B = []
        expr = 0*x
        for i in range(self.r):
            expr = expr + self.total_diff(self.vectors[i][0])*self.rec_forms[self.ep[i]]
            C = 0*x
            for j in range(len(self.A[1])):
                C = C - self.mc_invariants[self.Ri[i]]*diff(self.vectors[i][0].xreplace(self.master_function_to_symbol),self.master_str_to_symbol[self.A[1][j]])*self.contact_symbol[j+1]
            expr = expr + C
        B = [expr.subs({self.contact_symbol[j+1]:1}).subs(self.contact_zeros).subs({Dx:-Dx}).xreplace(self.normals_substitution).xreplace(self.curvature_subs) for j in range(len(self.A[1]))]
        return Matrix(B)
        

        # r = [R1,R2,R3]
        # ep = [epsilon_1,epsilon_2,epsilon_3]
        # vectors = [v1,v2,v3]
        # w1 = 0*x
        # w2 = 0*x
        # for i in range(3):
        #     w1 = w1-mc_invariants[r[i]]*diff(vectors[i][0],u)*contact_symbol[0]+total_diff(vectors[i][0])*rec_forms[ep[i]]
        #     w2 = w2-mc_invariants[r[i]]*diff(vectors[i][0],P)*contact_symbol[1]
        # b1 = (w1+w2).subs({contact_symbol[0]:1, contact_symbol[1]:0}).subs({D_x:-D_x})
        # b2 = (w1+w2).subs({contact_symbol[0]:0, contact_symbol[1]:1}).subs({D_x:-D_x})
        # return Matrix([[b1],[b2]])

    def Euler(L,var_index):
        var = ['u','P']
        var_sub = [var[var_index]+'x'*k for k in range(1,3)]
        expr = 0*x
        for i in range(len(var_sub)):
            expr = expr + (-1)**i*diff(diff(L,self.master_str_to_symbol[var_sub[i]]).subs(self.master_symbol_to_function),x,i)
        return expr.subs(master_function_to_symbol)

    def inv_Euler_Lagrange(L):
        inv_A = invariant_Euler()
        inv_B = invariant_Hamilton()
        expr1 = (inv_A[0,0]*Euler(L,0)+inv_A[0,1]*Euler(L,1)-L*inv_B[0,0]).subs({D_x:0})
        expr2 = (inv_A[1,0]*Euler(L,0)+inv_A[1,1]*Euler(L,1)-L*inv_B[1,0]).subs({D_x:0})
        return solve(Matrix([[expr1],[expr2]]))