import numpy as np
import scipy.sparse.linalg as splin


def nr_step(f, df, H, x, fold, mode='sparse', alpha=.0001, tolx=.0001, stpmax=1000):
    """
    Performs a single Newton-Raphson step
    :param f: incoming function, receives x as input
    :param x: arguments of function f
    :param df: derivative of f at point x
    :param H: hessian of f at point x
    :param fold: past iteration value of f
    :return: xnew, fnew: updated version of x, f(xnew)
    """

    if mode == 'sparse':
        newton_dir = splin.spsolve(H, -df)
    elif mode == 'dense':
        newton_dir = np.linalg.solve(H, -df)

    if np.linalg.norm(newton_dir) > stpmax:
        print("rescaling newton direction")
        newton_dir = stpmax * newton_dir / np.linalg.norm(newton_dir)

    slope = df @ newton_dir
    # if slope > 0.0:
    #     print('Roundoff problem in nr_step')
    #     sys.exit()

    test = 0.
    for d in range(len(x)):
        tmp = np.abs(newton_dir[d])/max(np.abs(x[d]), 1)
        if tmp > test:
            test = tmp

    if test < 1e-12:
        return x, fold, True

    alamin = tolx/test

    lam = 1.0

    while True:
        xnew = x + lam * newton_dir
        fnew = f(xnew)

        if lam < alamin:
            return x, fold, True
        elif fnew <= fold + alpha * lam * slope:
            return xnew, fnew, False
        else:
            if lam == 1.0:
                tmplam = - .5 * slope / (fnew - fold - slope)
            else:
                rhs1 = fnew - fold - lam * slope
                rhs2 = f2 - fold - lam2 * slope
                a = (rhs1/(lam*lam)-rhs2/(lam2*lam2))/(lam-lam2)
                b = (-lam2*rhs1/(lam*lam) + lam*rhs2/(lam2*lam2))/(lam-lam2)
                if a == 0.0:
                    tmplam = -slope/(2.0*b)
                else:
                    disc = b*b-3.0*a*slope
                    if disc < 0.0:
                        tmplam = .5*lam
                    elif b <= 0.0:
                        tmplam = (-b + np.sqrt(disc))/(3.0 * a)
                    else:
                        tmplam = -slope/(b+np.sqrt(disc))
                if tmplam > .5 * lam:
                    tmplam = .5 * lam
        lam2 = 1. * lam
        f2 = 1. * fnew
        lam = max(tmplam, .1*lam)


def nr_algo(f, df, h, x, mode='sparse'):
    """
    Performs the Newton-Raphson optimization method
    :param f: function to be optimized
    :param x: starting location
    :param df: derivative at point x
    :param H: Hessian at point x
    :param threshold: cutoff value, improvements smaller that this value are inconsequential
    :return: location of optimum
    """
    TOLMIN = 1e-12
    fold = f(x)
    cont = True
    iter = 0
    while cont:
        iter += 1
        x, fnew, check = nr_step(f, df(x), h(x), x, fold, mode=mode)
        # check for convergence on function values
        # I'm not sure how this should be implemented
        # check for grad of f zero , i.e., spurious convergence
        if check:
            test = np.max(np.abs(df(x)) * np.maximum(np.abs(x), 1)/max(f(x), 1))
            check = test < TOLMIN
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

    y = np.array([1, 1])
    print("initial guess = ", y)
    y = nr_algo(testf, testdf, testhf, y)
    print("final estimate = ", y)
    print("target optimum = (1, 1)")
    print("value at inferred optimum = ", testf(y))
    print("value at target optimum = ", testf([1., 1.]))
    print('rsquared = {}'.format(1-np.mean((y-np.ones(2))**2/np.ones(2)**2)))

