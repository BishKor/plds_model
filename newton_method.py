import numpy as np
import numdifftools as nd


def nr_step(f, x, fold):
    """
    Performs a single Newton-Raphson step
    :param f: incoming function, receives x as input
    :param x: arguments of function f
    :param fold: past iteration value of f
    :return: xnew: updated version of x
    """

    grad = nd.Gradient(f)(x)
    hess = nd.Hessian(f)(x)
    newton_dir = np.linalg.solve(hess, -grad)
    # newton_dir = -np.dot(np.linalg.inv(hess), grad)
    xnew = x + newton_dir
    alpha = .0001
    fnew = f(xnew)  # this can be optimized by saving this value as fold for the next step


    # First test
    test = fnew <= fold + alpha * np.dot(grad, xnew - x)
    if test:
        return xnew

    # Next test
    gamma1 = 1
    g0 = 1. * fold
    gprime = np.dot(grad, newton_dir)
    g1 = f(x + gamma1 * newton_dir)

    gamma2 = -.5 * gprime / (g1 - g0 - gprime)
    if gamma2 < .1:
        gamma2 = .1
    if gamma2 > 1:
        gamma2 = .5

    xnew = x + gamma2 * newton_dir
    fnew = f(xnew)
    test = fnew <= fold + alpha * grad @ (xnew - x)
    if test:
        return xnew

    # Finally go to modeling g as cubic
    while not test:
        gl1 = f(x + gamma1 * newton_dir)
        gl2 = f(x + gamma2 * newton_dir)
        ab = 1 / (gamma1 - gamma2) * np.dot(np.array([[1/gamma1**2, -1/gamma2**2], [-gamma2/gamma1**2, gamma1/gamma2**2]]),
                                            np.array([gl1 - gprime*gamma1 - g0, gl2 - gprime*gamma2 - g0]))

        gamma3 = (-ab[1] + np.sqrt(ab[1]**2 - 3*ab[0]*gprime))/(3 * ab[0])

        if gamma3 < .1 * gamma1:
            gamma3 = .1 * gamma1
        if gamma3 > .5 * gamma1:
            gamma3 = .5 * gamma1
        xnew = x + gamma3 * newton_dir
        fnew = f(xnew)
        test = fnew <= fold + alpha * grad @ (xnew - x)
        gamma1 = 1. * gamma2
        gamma2 = 1. * gamma3

    return xnew, fnew


def nr_algo(f, x, threshold=.00001):
    """
    Performs the Newton-Raphson optimization method
    :param f: function to be optimized
    :param x: starting location
    :param threshold: cutoff value, improvements smaller that this value are inconsequential
    :return: location of optimum
    """

    fold = f(x)
    while cont:
        x, fnew = nr_step(f, x, fold)
        if fnew - fold < threshold:
            cont = False
        fold = 1. * fnew
    return x
