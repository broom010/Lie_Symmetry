#################################
# List Based Dictionary Creation
#################################

def str_sym(A,
n):
    B = {}
    for i in range(len(A[1])):
        for j in range(len(A[0])):
            B = merge_two_dicts(B, dict((p, symbols(str(p[0]) + '_'*(1 - int(len(p)==1)) + str(p[1:]))) for p in fullJet(A[0], A[1], n)))
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

def initialize_normals(B):
    A = B[0]
    n = B[1]

    C = ['H']
    for i in range(n + 1):
        for p in A[1]:
            C.append('I^' + str(p) + '_' + str(i))

def create_normals(B, E, T):
    A_jet = E[1]
    cross_section_indx = T[3]

    C = initialize_normals(B)
    normals_substitution = dict((symbols(A_jet[i]), symbols(C[i])) for i in range(len(A_jet)))
        for i in cross_section_indx:
            normals_substitution[symbols(A_jet[i])] = 0

#############################################
# Method - Based Dictionary and List Creation
#############################################

def gen_invs(B, T):
    A = B[0]
    n = B[1]
    normals_substitution = T[4]

    t = len(A[1])
    C1 = []
    C2 = []
    for i in range(t):
        C3 = [str(p[0]) + '_'*(1 - int(len(p)==1)) + str(p[1:]) for p in fullJet(A[0], A[1][i], n)]
        y = np.trim_zeros([symbols(C3[i]).subs(normals_substitution) for i in range(len(C3))])[0]
        C1.append(y)
        C2.append(symbols('kappa^' + str(y)[2]))
    return dict((k, v) for k, v in zip(C1, C2))

def contact_Reduction(B):
    A = B[0]
    contact_symbol = B[7]
    contact_zeros = B[8]

    contact_reduction = dict((contact_symbol[i + 3], solve(Dx*contact_symbol[i + 1] - inv_D_x_contact(i, 0), contact_symbol[i + 3])[0]) for i in range(len(A[1])))
    contact_zeros = dict((p, 0) for p in contact_symbol)
    return [contact_reduction, [contact_zeros]]

def create_vectors(B, S, T):
    params = B[2]
    r = B[3]
    m = B[6]
    identity = T[0]
    transform = T[1]

    vectors = []
    for i in range(r):
        vectors.append(vect_Field(B, S, T, params[i]))


def moving_frame(B, S, T):
        params = B[2]
        transform = T[1]
        cross_section_index = T[3]
        K = T[2]

        C = [transform[cross_section_indx[i]]  -  K[i] for i in range(len(K))]
        C = [c.xreplace(S[5]) for c in C]
        return solve(C, params, dict=True)[0]

def rec_Relations(B, S, T):
    A = B[0]
    r = B[3]
    Ri = B[4]
    Phantoms = T[0]
    vectors = T[1]
    normals_substitution = T[4]
    transformed_subs_forward = T[8]

    C = []
    for w in Phantoms:
        s = w.subs(transformed_subs_forward)
        C1 = [apply_vect(B, S, vectors[i], s.subs(transformed_subs_forward)).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
        C1.insert(0, diff(s.xreplace(S[3]), x).subs(S[5]).xreplace(normals_substitution))
        C2 = Ri[:]
        C2.insert(0, 1)
        expression = sum([a*b for (a, b) in zip(C1, C2)])
        C.append(expression)
    return solve(C, Ri)

def rec_Relations_Forms(B, S, T, E, A, cross_section_indx, transformed_subs_forward):
    A = B[0]
    r = B[3]
    ep = B[5]
    contact_symbol = B[7]
    Phantoms = T[0]
    vectors = T[1]
    normals_substitution = T[4]


    Z = []
    C = [contact_symbol[i] for i in cross_section_indx]
    j = 0
    for w in Phantoms:
        s = w.subs(transformed_subs_forward)
        C1 = [apply_vect(A, vectors[i], s.subs(transformed_subs_forward), S[0], S[5], n).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
        C1.insert(0, C[j])
        C2 = ep[:]
        C2.insert(0, 1)
        expression = sum([a*b for (a, b) in zip(C1, C2)])
        Z.append(expression)
        j  + = 1
    return solve(Z, ep)

###############################
# Function Definitions
###############################

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
    transform = T[9]
    identity = T[11]

    v = []
    C = transform[:m]
    for i in range(len(C)):
        v.append((diff(C[i], y).subs(dict((params[j], identity[j]) for j in range(len(identity)))) + 0*x).xreplace(S[5]))
    return v

def Dnx(B, E, S, f):
    A_flat = E[0]
    n = B[1]
    transformed_subs_backward = T[7]
    transform = T[9]

    for i in range(n):
        f = f.xreplace(transformed_subs_backward).subs(S[3])
        f = (1/(diff(transform[0], A_flat[0]))*diff(f, A_flat[0])).xreplace(S[5])
    return simplify(f.xreplace(S[5]))

def invariantization(S, T, f):
    transformed_subs_backward = T[4]
    frame = T[]

    f = f.xreplace(S[5]).xreplace(transformed_subs_backward)
    return simplify(f.xreplace(frame))

def DI_x(B, S, T, f):
    Ri = B[4]
    vectors = T[1]
    mc_invariants = T[2]
    normals_substitution = T[4]
    transformed_subs_forward = T[8]
    
    s = f.subs(transformed_subs_forward)
    C1 = [apply_vect(B, S, vectors[i], s.subs(transformed_subs_forward)).xreplace(S[5]).xreplace(normals_substitution) for i in range(r)]
    temp = diff(s.xreplace(S[3]), x)
    temp = temp.xreplace(S[5]).xreplace(normals_substitution)
    C1.insert(0, temp)
    C2 = [mc_invariants[Ri[i]] for i in range(r)]
    C2.insert(0, 1)
    expression = sum([a*b for (a, b) in zip(C1, C2)])
    return expression

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
    return expr.xreplace(S[5]).subs(S[0])

def inv_D_x_contact(B, S, T, var_index, i):
    r = B[3]
    Ri = B[4]
    contact_symbol = B[7]
    vectors = T[1]
    mc_invariants = T[2]
    rec_forms = T[3]
    normals_substitution = T[4]
    curvature_subs = T[5]

    expr = 0
    for j in range(r):
        expr = expr + Lie_contact_diff(B, S, vectors[j], var_index, i)*mc_invariants[Ri[j]]
    return (contact_symbol[var_index + 3] + expr).subs(normals_substitution).subs(curvature_subs)

def inv_vert_diff(B, S, T, var_index, i):
    A = B[0]
    r = B[3]
    ep = B[5]
    contact_symbol = B[7]
    vectors = T[1]
    rec_forms = T[3]
    normals_substitution = T[4]
    curvature_subs = T[5]

    expr = contact_symbol[var_index + 2*i + 1]
    C = [(rec_forms[ep[j]])*(Prolong(A, vectors[j], i)[1 + var_index + 2*i]).xreplace(S[5]).xreplace(normals_substitution).xreplace(curvature_subs) for j in range(r)]
    return expr + sum(C)

def inv_Euler(B, S):
    contact_symbol = B[7]
    contact_zeros = B[8]
    curvature_subs = T[5]

    C = []
    for i in range(len(A[1])):
        temp = []
        for j in range(len(A[1])):
            temp.append(inv_vert_diff(B, S, T i, 1).subs(curvature_subs).subs({contact_symbol[j + 1]:1}).subs(contact_zeros[8]).subs({Dx: - Dx}))
        C.append(temp)
    return Matrix(C).T

def invariant_Hamilton(B, S, T):
    A = B[0]
    r = B[3]
    Ri = B[4]
    ep = B[5]
    contact_symbol = B[7]
    contact_zeros = B[8]
    vectors = T[1]
    mc_invariants = T[2]
    rec_forms = T[3]
    normals_substitution = T[4]
    curvature_subs = T[5]

    expr = 0*x
    for i in range(r):
        expr = expr + total_diff(S, vectors[i][0])*rec_forms[ep[i]]
        C = 0*x
        for j in range(len(A[1])):
            C = C  -  mc_invariants[Ri[i]]*diff(vectors[i][0].xreplace(S[5]), S[0][A[1][j]])*contact_symbol[j + 1]
        expr = expr + C
    return Matrix([expr.subs({contact_symbol[j + 1]:1}).subs(contact_zeros).subs({Dx: - Dx}).xreplace(normals_substitution).xreplace(curvature_subs) for j in range(len(A[1]))])

###############################
# Test/Validation Methods
###############################

def Cross_section_validate(A, n, vectors, S[5]):
    vects = []
    vects = [Prolong(A, v, n) for v in vectors]
    temp = []
    for i in range(len(vects)):
        temp.append([vects[i][j] for j in A])
    M = Matrix(temp)
    print(latex(simplify(M.det().xreplace(S[5]))))
    print('')
    return M



for p in cross[1]:
    self.cross_section_indx.append(self.A_jet_no_underscore.index((p.xreplace(self.master_function_to_symbol)).xreplace(self.master_symbol_to_str)))
self.transformed_subs_backward = {symbols(self.A_flat[i]): expression[i].xreplace(self.master_function_to_symbol)  for i in range(self.m)}
self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}
for i in range(1, self.n + 1):
    for p in self.A[1]:
        self.transform.append(self.Dnx(symbols(p), i))
self.transformed_subs_backward = {symbols(self.A_jet[i]): self.transform[i].xreplace(self.master_function_to_symbol)  for i in range(len(self.A_jet))}
self.transformed_subs_forward = {v: k for k, v in self.transformed_subs_backward.iteritems()}