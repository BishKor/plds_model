import numpy as np
import numdifftools as nd


def nr_step(f, theta):
    """
    Performs a single Newton-Raphson step
    :param f: incoming function, receives theta as input
    :param theta: parameters in function f
    :return: thetanew: updated version of theta (new coordinate in parameter space)
    """

    theta = np.zeros(len(theta))
    grad = nd.Gradient(f)(theta)
    hess = nd.Hessian(f)(theta)
    newton_dir = -np.dot(np.linalg.inv(hess), grad)
    thetanew = theta + newton_dir
    alpha = .0001
    fnew = f(thetanew)  # this can be optimized by saving this value as fold for the next step
    fold = f(theta)


    # First test
    test = fnew <= fold + alpha * np.dot(grad, thetanew - theta)
    if test:
        return thetanew

    # Next test
    gamma1 = 1
    g0 = 1. * fold
    gprime = np.dot(grad, newton_dir)
    g1 = f(theta + gamma1 * newton_dir)

    gamma2 = -.5 * gprime / (g1 - g0 - gprime)
    if gamma2 < .1:
        gamma2 = .1
    if gamma2 > 1:
        gamma2 = .5

    thetanew = theta + gamma2 * newton_dir
    fnew = f(thetanew)
    test = fnew <= fold + alpha * np.dot(grad, thetanew - theta)
    if test:
        return thetanew

    # Finally go to modeling g as cubic
    while not test:
        gl1 = f(theta + gamma1 * newton_dir)
        gl2 = f(theta + gamma2 * newton_dir)
        ab = 1 / (gamma1 - gamma2) * np.dot(np.array([[1/gamma1**2, -1/gamma2**2], [-gamma2/gamma1**2, gamma1/gamma2**2]]),
                                            np.array([gl1 - gprime*gamma1 - g0, gl2 - gprime*gamma2 - g0]))

        gamma3 = (-ab[1] + np.sqrt(ab[1]**2 - 3*ab[0]*gprime))/(3 * ab[0])

        if gamma3 < .1 * gamma1:
            gamma3 = .1 * gamma1
        if gamma3 > .5 * gamma1:
            gamma3 = .5 * gamma1
        thetanew = theta + gamma3 * newton_dir
        fnew = f(thetanew)
        test = fnew <= fold + alpha * np.dot(grad, thetanew - theta)
        gamma1 = 1. * gamma2
        gamma2 = 1. * gamma3

    return thetanew



