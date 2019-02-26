import numpy as np
import timeit
from sympy import *
from itertools import *
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib

def nJet(X,U,n):
    marker = '%s'*n
    combs = [marker % x for x in combinations_with_replacement(X,n)]
    return [dep+ind for dep in U for ind in combs]
    
def fullJet(X,U,n):
    return X+[x for i in range(n+1) for x in nJet(X,U,i)]
    
def nJetAll(X,U,n):
    marker = '%s'*n
    combs = [marker % x for x in product(X, repeat = n)]
    return [dep+ind for dep in U for ind in combs]
    
def fullJetAll(X,U,n):
    return X+[x for i in range(n+1) for x in nJetAll(X,U,i)]

def nJetAllStr(X,U,n):
    marker = '%s'*n
    combs = [marker % x for x in product(X, repeat = n)]
    return [str(dep)+"_{"+str(ind)+"}" for dep in U for ind in combs]
    
def fullJetStr(X,U,n):
    return [str(x) for x in X+U]+[x for i in range(1,n+1) for x in nJetAllStr(X,U,i)]


def uJalpha(u,x,X,U):
    J = nJet(X,U,len(str(u)))
    for i in range(len(J)):
        if sorted(str(u)+str(x)) == sorted(J[i]):
            return var(J[i])
            break
            
def divUJi(X,U,u,J,i):
    f = u
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
    phi = v[len(X):]
    xi = v[:len(X)]
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
    a = v
    J = diffOrd(len(X),n)
    for i in range(n):
        for j in range(len(X)):
            for q in range(len(U)):
                a.append(phiAlpha(X,U,v,J[i][j],q))
    return a
    
def ProlongOut(X,U,v,n,tol):
    P = Prolong(X,U,v,n)
    I = fullJetAll(X,U,n)
    J = fullJetStr(X,U,n)
    a = len(I)
    w = [str(P[i])+r")\frac{\partial}{\partial "+ str(J[i]) + "}"  for i in range(a-1) if P[i]!=0]
    pq = len(X)+len(U)
    tex = '$'
    d = "("+'+('.join(w)
    d = d.replace('**','^',10)
    d = d.replace('*','\cdot ',10)
    for i in range(a):
        d=d.replace(str(I[a-i-1]),str(J[a-i-1]),tol)
    tex = tex+d+'$'
    return tex
    
def ProlongPrint(X,U,v,n):
    tmptext = ProlongOut(X,U,v,n,15)
    matplotlib.rc('image', origin='upper')
    parser = mathtext.MathTextParser("Bitmap")
    parser.to_png('test2.png',
                r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} '
                r'y\right)\right]$', color='green', fontsize=14, dpi=100)
    code, depth2 = parser.to_rgba(
        'Output: '+tmptext, color='black', fontsize=10, dpi=200)
    
    fig = plt.figure()
    fig.figimage(code.astype(float)/255., xo=100, yo=220)
    plt.show()
    
if __name__ == "__main__":

    # # Example 1
    # X = ['x']
    # U = ['u']
    # var(fullJetAll(X,U,3))
    # print Prolong(X,U,[-u,x],2)
    
    # Example 2
    X = ['x']
    U = ['u']
    var(fullJetAll(X,U,3))