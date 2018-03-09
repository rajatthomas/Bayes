"""
Package to perform Variational Bayesian Inference for Linear and Logistic Regression.

Based on:
https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/arxiv2014.pdf

Programmed by:
Rajat Mani Thomas
University of Amsterdam
09042018
"""
import numpy as np
import numpy.linalg as lin
import sys
import scipy.special as sp
from warnings import warn

def logdet(A):
    """

    This function is more accurate than log(det(A))
    Copyright(c) 2013, Jan, Drugowitsch

    :param A: Positive definite matrix
    :return: log(det(A))
    """

    return 2 * sum(np.log(np.diag(lin.cholesky(A))), 1)


def vb_linear_fit(X, y, a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4):
    """
    Estimates w such that y = Xw, using Bayesian regularisation.

    The underlying generative model assumes:
    p(y | x, w, tau) = N(y | w'x, tau^-1),

   x and y -> rows of the given X and y.
   w and tau ->  conjugate normal inverse-gamma prior
   p(w, tau | alpha) = N(w | 0, (tau alpha)^-1 I) Gam(tau | a0, b0),

    with the hyper-prior
    p(alpha) = p(alpha | c0, d0).

    The prior parameters a0, b0, c0, and d0 can be set by calling the script
    with the additional parameters vb_linear_fit(X, y, a0, b0, c0, d0). If
    not given, they default to values a0 = 1e-2, b0 = 1e-4, c0 = 1e-2, and
    d0 = 1e-4, such that the prior is uninformative.

    The returned posterior parameters (computed by variational Bayesian
    inference) determine a posterior of the form

    N(w1 | w, tau^-1 V) Gam(tau | an, bn).

    Also, the mean E_a = E(alpha) is returned, together with the inverse of V,
    and its log determinant. L is the variational bound of the model, and is a
    lower bound on the log-model evidence ln p(y | X).

    Copyright (c) 2013, 2014, Jan Drugowitsch
    All rights reserved.

    :param X:
    :param y:
    :param a0:
    :param b0:
    :param c0:
    :param d0:
    :return:
    """
    # Pre-process data
    N, D = X.shape
    X_corr = X.T.dot(X)
    Xy_corr = X.T.dot(y)

    an = a0 + N/2
    gammaln_an = sp.gammaln(an)
    cn = c0 + D/2
    gammaln_cn = sp.gammaln(cn)

    # Iterate to find hyperparameters
    L_last = -sys.float_info.max  # 1.7976931348623157e+308
    max_iter = 500
    E_a = c0 / d0

    for iter in range(max_iter):
        # covariance and weight of linear model
        invV = E_a * np.eye(D) + X_corr
        V = lin.inv(invV)
        logdetV = - logdet(invV)
        w = V.dot(Xy_corr)

        # parameters of noise model(an remains constant)
        sse = ((X.dot(w) - y)**2).sum()
        bn = b0 + 0.5 * (sse + E_a * (w.T.dot(w)))
        E_t = an / bn

        # hyperparameters of covariance prior(cn remains constant)
        dn = d0 + 0.5 * (E_t * (w.T.dot(w)) + np.trace(V))
        E_a = cn / dn

        #import pdb; pdb.set_trace()
        # variational bound, ignoring constant terms for now
        L = - 0.5 * (E_t * sse + (X * (X.dot(V))).sum()) + \
            0.5 * logdetV - b0 * E_t + gammaln_an -\
            an * np.log(bn) + an + gammaln_cn - cn * np.log(dn)

        # variational bound must grow!
        if L_last > L:
            print('Last bound {}, current bound {}'.format(L_last, L))
            print('Variational bound should not reduce')
            exit()

        # stop if change in variation bound is < 0.001 %
        if abs(L_last - L) < abs(0.00001 * L):
            break

        L_last = L

    if iter == max_iter:
        warn('Bayes:maxIter, '
             'Bayesian linear regression reached MAX # iterations.');


    # augment variational bound with constant terms
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - \
        sp.gammaln(a0) + a0 * np.log(b0) - sp.gammaln(c0) \
        + c0 * np.log(d0)

    return w, V, invV, logdetV, an, bn, E_a, L

if __name__ == "__main__":

    X = np.hstack((np.ones((100, 1)), np.random.rand(100, 3)))
    y = X.dot(np.array([1, 2, 3, 5]))

    w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit(X, y)

    print(w)