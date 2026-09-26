"""The mean-neutral location of a row under a Gaussian pool effect, and its Gauss-Hermite rule."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit


@lru_cache(None)
def quadrature(order):
    x,w=hermgauss(order)
    return x,w/np.sqrt(np.pi)


def mean_neutral(eta,sd,order=40):
    """Conditional location A and its first three derivatives w.r.t. mean logit."""
    eta,sd=np.broadcast_arrays(np.asarray(eta,float),np.asarray(sd,float))
    # At zero variance A(eta)=eta exactly, even when expit(eta) underflows.
    # Exclude these rows from quadrature and derivative ratios (otherwise 0/0).
    zero=sd==0
    if np.any(zero):
        result=[eta.copy(),np.ones_like(eta),np.zeros_like(eta),np.zeros_like(eta)]
        if np.any(~zero):
            active=mean_neutral(eta[~zero],sd[~zero],order)
            for output,value in zip(result,active):
                output[~zero]=value
        return tuple(result)
    x,w=quadrature(order);u=np.sqrt(2)*sd[:,None]*x
    # Solve on the rare-event side to avoid subtracting probabilities near one.
    sign=np.where(eta>0,-1.,1.);work=eta*sign
    target=expit(work);A=work-.5*sd*sd
    for _ in range(40):
        p=expit(A[:,None]+u);v=p*(1-p);m1=v@w
        step=(p@w-target)/m1;A-=np.clip(step,-2,2)
        if np.max(abs(step))<2e-12:break
    else:raise RuntimeError('mean-neutral location failed')
    p=expit(A[:,None]+u);v=p*(1-p)
    m1=v@w;m2=(v*(1-2*p))@w
    t1=target*(1-target);t2=t1*(1-2*target)
    a1=t1/m1;a2=(t2-m2*a1*a1)/m1
    return sign*A,a1,sign*a2
