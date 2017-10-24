import numpy as np
import scipy.sparse.linalg as splin


def nr_step(f, df, H, x, fold):
    """
    Performs a single Newton-Raphson step
    :param f: incoming function, receives x as input
    :param x: arguments of function f
    :param df: derivative of f at point x
    :param H: hessian of f at point x
    :param fold: past iteration value of f
    :return: xnew, fnew: updated version of x, f(xnew)
    """
    print('computing newton direction')
    #newton_dir = splin.spsolve(H, -df)
    newton_dir = np.linalg.solve(H, -df)
    xnew = x + newton_dir
    alpha = .0001
    fnew = f(xnew)  # this can be optimized by saving this value as fold for the next step
    # First test
    test = fnew >= fold + alpha * np.dot(df, xnew - x)
    if test:
        return xnew, fnew

    # Next test
    gamma1 = 1
    g0 = 1. * fold
    gprime = np.dot(df, newton_dir)
    g1 = 1. * fnew  # f(x + gamma1 * newton_dir)

    gamma2 = - .5 * gprime / (g1 - g0 - gprime)
    if gamma2 < .1:
        gamma2 = .1
    if gamma2 > 1:
        gamma2 = .5

    xnew = x + gamma2 * newton_dir
    fnew = f(xnew)
    test = fnew >= fold + alpha * df @ (xnew - x)
    if test:
        return xnew, fnew

    # Finally go to modeling g as cubic
    while not test:
        gl1 = f(x + gamma1 * newton_dir)
        gl2 = f(x + gamma2 * newton_dir)
        ab = (1 / (gamma1 - gamma2)) * np.dot(np.array([[1/gamma1**2, -1/gamma2**2], [-gamma2/gamma1**2, gamma1/gamma2**2]]),
                                            np.array([gl1 - gprime*gamma1 - g0, gl2 - gprime*gamma2 - g0]))

        gamma3 = (-ab[1] + np.sqrt(ab[1]**2 - 3*ab[0]*gprime))/(3 * ab[0])

        if gamma3 < .1 * gamma1:
            gamma3 = .1 * gamma1
        if gamma3 > .5 * gamma1:
            gamma3 = .5 * gamma1
        xnew = x + gamma3 * newton_dir
        fnew = f(xnew)

        gamma1 = 1. * gamma2
        gamma2 = 1. * gamma3

        test = fnew <= fold + alpha * df @ (xnew - x)
    print('returning updated locations')
    return xnew, fnew

def nr_algo(f, df, H, x, threshold=.0001):
    """
    Performs the Newton-Raphson optimization method
    :param f: function to be optimized
    :param x: starting location
    :param df: derivative at point x
    :param H: Hessian at point x
    :param threshold: cutoff value, improvements smaller that this value are inconsequential
    :return: location of optimum
    """
    print(x)
    fold = f(x)
    cont = True
    while cont:
        x, fnew = nr_step(f, df(x), H(x), x, fold)
        if fnew - fold < threshold:
            cont = False
        fold = 1. * fnew
        print(x)
    return x

if __name__ == "__main__":
    def testf(x):
        w = np.array([2, 3])
        return np.sum(-(x-w)**2)

    def testdf(x):
        w = np.array([2, 3])
        return np.array(-2*x+2*w)

    def testhf(x):
        return np.array([[-2, 0], [0, -2]])

    y = np.random.randn(2)
    y = nr_algo(testf, testdf, testhf, y)
