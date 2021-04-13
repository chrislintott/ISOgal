from astropy.table import Table
from astropy.io import ascii
import pandas
import astropy.units as u
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy.stats as stats
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

BBmodel = ascii.read("../sav/Combineddata.csv",data_start=2)
BBmodel.rename_column('\ufeffFe_H', 'Fe_H')

# Now interpolate between points

def piecewise_poly(x):
    '''this function allows us to set the extrapolation to the limits of the data in x'''
    model_poly = models.Polynomial1D(degree=3)
    fitter_poly = fitting.LinearLSQFitter()
    best_fit_poly = fitter_poly(model_poly, BBmodel['Fe_H'],BBmodel['H20'])
    minx, maxx = np.min(BBmodel['Fe_H']), np.max(BBmodel['Fe_H'])
    minxy = best_fit_poly(minx)
    maxxy = best_fit_poly(maxx)
    if not hasattr(x, '__iter__'):
        if x < minx:
            return minxy
        elif x > maxx:
            return maxxy
        else:
            return best_fit_poly(x)
    else:
        out = np.zeros(len(x))
        out[x < minx] = minxy
        out[x > maxx] = maxxy
        out[(x >= minx) & (x <= maxx)] = best_fit_poly(x[(x >= minx) & (x <= maxx)])
        return out
    
def get_iso_water_fraction(fehs):
    return piecewise_poly(fehs)
