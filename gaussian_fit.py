# Gaussian curve fitting
# https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting

from scipy.odr import odrpack as odr
from scipy.odr import models
from scipy import optimize
import math
import numpy as np


def gaussian(B, x):
    ''' Returns the gaussian function for B=m,stdev,max,offset '''
    return B[3] + B[2] / (B[1] * math.sqrt(2 * math.pi)) * np.exp(-((x - B[0])**2 / (2 * B[1]**2)))


def gaussian_lsq(x, y, verbose=False, itmax=200, iparams=[]):
    ''' Performs a gaussian least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.'''

    def _gauss_fjd(B, x):
        # Analytical derivative of gaussian with respect to x
        return 2 * (x - B[0]) * gaussian(B, x)

    # these derivatives need to be fixed!
    def _gauss_fjb(B, x):
        # Analytical derivatives of gaussian with respect to parameters
        _ret = np.concatenate((-2 * (x - B[0]) * gaussian(B, x),
                               ((x - B[0])**2 / (2 * B[1]**2) - 1) /
                               B[1] * gaussian(B, x),
                               gaussian(B, x) / B[2],
                               np.ones(x.shape, float)))
        _ret.shape = (4,) + x.shape
        return _ret

    # Centre data in mean(x) (makes better conditioned matrix)
    mx = np.mean(x)
    x2 = x - mx

    if not any(iparams):
        # automatic guessing of gaussian's initial parameters (saves
        # iterations)
        iparams = np.array([x2[np.argmax(y)], np.std(y), math.sqrt(
            2 * math.pi) * np.std(y) * np.max(y), 1.])

    gauss = odr.Model(gaussian, fjacd=_gauss_fjd, fjacb=_gauss_fjb)

    mydata = odr.Data(x2, y)
    myodr = odr.ODR(mydata, gauss, beta0=iparams, maxit=itmax)

    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)

    fit = myodr.run()

    # Display results:
    if verbose:
        fit.pprint()
        print('Re-centered Beta: [%f  %f  %f %f]' %
              (fit.beta[0] + mx, fit.beta[1], fit.beta[2], fit.beta[3]))

    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        print("(WWW) gauss_lsq: Iteration limit reached, result not reliable!")

    # Results and errors
    coeff = fit.beta
    coeff[0] += mx  # Recentre in original axis
    err = fit.sd_beta

    return coeff, err, itlim


def gaussian_fit(y, verbose=False, itmax=200, iparams=[]):
    x = np.arange(len(y))
    return gaussian_lsq(x, y, verbose, itmax, iparams)


def errfunc(P, x, y):
    return y - gaussian(P, x)


def gaussian_sfit(data):
    x = np.arange(len(data))
    # initial estimate of parameters
    p0 = [np.argmax(data), 1., np.max(data), 0]
    fit = optimize.leastsq(errfunc, p0, args=(x, data))
    return fit[0]
