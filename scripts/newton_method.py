import numpy as np
import scipy.sparse.linalg as splin
# from memory_profiler import profile

def nr_step(f, df, H, x, fold, alpha=1.0): #, tolx=.0001, stpmax=1000):
    """
    Performs a single Newton-Raphson step
    :param f: incoming function, receives x as input
    :param x: arguments of function f
    :param df: derivative of f at point x
    :param H: hessian of f at point x
    :param fold: past iteration value of f
    :return: xnew, fnew: updated version of x, f(xnew)
    """
    newton_dir = splin.spsolve(H, -df)
    xnew = x + alpha * newton_dir
    fnew = f(xnew)
    if np.abs(fold - fnew) < .00001 and np.sum(np.abs(newton_dir)) < .00001:
        return xnew, fnew, True
    else:
        return xnew, fnew, False


def nr_step_with_backtracking(f, df, H, x, fold, alpha=.25, beta=.7):
    newton_dir = splin.spsolve(H, -df)
    t = 1.0
    xnew = x + t * newton_dir
    fnew = f(xnew)
    keepsearching = (fnew > fold + alpha * t * df @ newton_dir)
    while keepsearching:
        t *= beta
        xnew = x + t * newton_dir
        fnew = f(xnew)
        keepsearching = (fnew > fold + alpha * t * df @ newton_dir)

    if np.abs(fold - fnew) < .00001 and np.sum(np.abs(newton_dir)) < .00001:
        return xnew, fnew, True
    else:
        return xnew, fnew, False


def nr_algo(f, df, h, x, maxiter=5, thresholditer=10, mode='simple'):
    """
    Performs the Newton-Raphson optimization method
    :param f: function to be optimized
    :param x: starting location
    :param df: derivative at point x
    :param H: Hessian at point x
    :param threshold: cutoff value, improvements smaller that this value are inconsequential
    :return: location of optimum
    """
    fold = f(x)
    cont = True
    iter = 0
    if mode == 'simple':
        al = 1.0
        while cont:
            x, fnew, check = nr_step(f, df(x), h(x), x.copy(), fold, alpha=al)
            print('fold = {} diff = {}'.format(fnew, fold - fnew))
            iter += 1
            if iter > thresholditer:
                al *= .5
            if check or iter > maxiter:
                cont = False
            else:
                fold = 1. * fnew
        return x
    elif mode == 'backtracking':
        while cont:
            x, fnew, check = nr_step_with_backtracking(f, df(x), h(x), x.copy(), fold)
            print('fold = {} diff = {}'.format(fnew, fold - fnew))
            iter += 1
            if check or iter > maxiter:
                cont = False
            else:
                fold = 1. * fnew
        return x


if __name__ == "__main__":
    def testf(x):
        return (1 - x[0])**2 + 10*(x[1] - x[0]**2)**2

    def testdf(x):
        return np.array([2*(1-x[0]) - 40*x[0]*(x[1]-x[0]**2), 20*(x[1] - x[0]**2)])

    def testhf(x):
        return np.array([[-2-40*x[1] - 120*x[0]**2, -40*x[0]], [-40*x[0], 20]])

    y = np.array([10, 10])
    print("initial guess = ", y)
    y = nr_algo(testf, testdf, testhf, y)
    print("final estimate = ", y)
    print("target optimum = (1, 1)")
    print("value at inferred optimum = ", testf(y))
    print("value at target optimum = ", testf([1., 1.]))
    print('rsquared = {}'.format(1-np.mean((y-np.ones(2))**2/np.ones(2)**2)))

