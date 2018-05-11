# coding=utf-8
import mpmath as mmm

def hyp1f1(a,b,z,**arg):
    if 'tol' in arg.keys() and arg['tol']:
        __tol = arg['tol']
    else:
        __tol = 1e-5
    # tol is the accuracy of the 
    if 'nmin' in arg.keys() and arg['nmin']:
        __nmin = arg['nmin']
    else:
        __nmin = 10

    __term = z*a/b
    __output = 1 + __term
    __n = 1
    __an = a
    __bn = b
    
    while(__n < __nmin) or (abs(__term) > __tol):
        __n = __n + 1
        __an = __an + 1
        __bn = __bn + 1
        __term = z * __term * __an / __bn / __n
        __output = __output + __term

    return __output

def hyperU(a,b,z,**arg):
    from math import gamma
    return (gamma(1-b)/gamma(a-b+1)) * hyp1f1(a,b,z,**arg)+(gamma(b-1)/gamma(a)) * z**(1-b) * hyp1f1(a-b+1,2-b,z,**arg)

def main():
    dicounted_factor = 0.015
    kappa = 0.06
    theta = 1.1
    sigma = 0.181
    cs = 0.015
    cb = 0.025

    a = dicounted_factor/kappa
    b = (2*kappa*theta)/sigma**2
    z = 2*kappa/sigma**2
    
    psi = lambda x: hyp1f1(a,b,z*x)
    phi = lambda x: hyperU(a,b,z*x)
    dpsi = lambda x: (a*z/b)*hyp1f1(a+1,b+1,z*x)
    dphi = lambda x: (-a*z)*hyperU(a+1,b+1,z*x)

    mmm.mp.dps = 9
    mmm.mp.pretty = True

    __result = mmm.findroot([
        lambda __x_1,__x_2,A,B: A*psi(__x_1) - (B * phi(__x_1) + __x_1 - cs),
        lambda __x_1,__x_2,A,B: B*phi(__x_2) - (A*psi(__x_2) - __x_2 - cb),
        lambda __x_1,__x_2,A,B: A*dpsi(__x_1) - (B * dphi(__x_1) + 1),
        lambda __x_1,__x_2,A,B: B*dphi(__x_2) - (A*dpsi(__x_2) - 1)
        ],(1.5, 0.5, 0.001, 0.001),solver='secant', verbose=True)

    print "A = {A}, B = {B}, x_1 = {x_1}, x_2 = {x_2} ".format(A = __result[2], B = __result[3], x_1 = __result[0], x_2 = __result[1])

if __name__ == "__main__":
    main()