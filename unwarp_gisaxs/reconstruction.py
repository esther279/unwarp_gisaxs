# construct pattern of direct beam and reflect from a model
# build a one-d model
import glob
import os
import numpy as np
import matplotlib.pyplot as plt # for showing image
from pylab import * # for multiple figure window
from skimage import io
import re
import statsmodels.api as sm
from scipy.optimize import minimize

import time

t = time.time()

def SAXS_recons(qx_dimension=None,skip_qx=None,alpha_incident=None,\
		GISAXS_im=None,x0=None,fitting_range_model=None,\
		qz=None,qz_r=None,qz_d=None,qz_f=None,\
		reflc_params=None,trans_params=None,r_f=None,t_f=None,\
		qz_min=None,qz_max=None,range_index_min=None,\
		range_index_max=None,initial = np.empty((0,0)), iterations=3000):
    """
    Unwarp GISAXS pattern to SAXS pattern
    See paper: Liu, Jiliang and Kevin Yager. " Unwarp GISAXS data", IUCrJ (2018)

    This function will iteratively reconstruct SAXS for each qx

    "alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,\
    fitting_range_model,qz_min,qz_max,range_index_min,range_index_max"
    will be precalculated by module coefficient_calculation.
    example.py shows how to run coefficient_calculation and this function.

    Parameters
    -----------
    qx_dimension: 1D numpy.array
    		  this is range(len(qx)) correlates to dimension of qx for GISAXS.
    		  function parallel unwarp GISAXS for each qx independently.
    skip_qx : 1D numpy.array
    		  this array contains index of qx, which has NO GISAXS data,
    		  usually masked by gap or beam stop.
    alpha_incident: 1D numpy.array
    		  incident angle for corresponded GISAXS
    GISAXS_im: 3D numpy.array
    		  stack of GISAXS pattern. each GISAXS is a 2D numpy.array, multiple
    		  GISAXS patterns with different incident angle pack together.
    x0      : 1D numpy.array
     		  linear spaced 1D array with customer defined length, minimum is 0,
     		  maximum should be less than maximum two theta of detector space.
    fitting_range_model: 2D numpy.array
    		  reinterpolated multiple incident angle related Q (1D array) to
    		  same size for requirements of opitimization of function.
    qz      : 1D numpy.array
    		  q of detectorspace
    qz_r    : 1D numpy.array
    		  refraction corrected q for reflect channel
    qz_d    : 1D numpy.array
    		  refraction corrected q for direct channel
    reflc_params: 1D numpy.array
    		  reflectivity coefficient
    trans_index: 1D numpy.array
    		  transmission coefficient
    r_f     : floats
    		  reflectivity coefficient for refraction corrected q
    t_f     : floats
    		  transmission coefficient for refrection corrected q
    qz_min  : floats
    		  minimum refrection corrected q
    qz_max  : floats
    		  maximum refraction corrected q
    range_index_min: int
    		  index of qz_min
    range_index_max: int
    		  index of qz_max
    initial : 2D numpy.array
    		  initial guess other than default neighbor pattern
    iterations: int
    		  maximum function evaluation iterations

    Returns
    ---------
    im      : 2D numpy.array
    		  SAXS pattern unwarped from GISAXS data
    """
    fitting_portion_model = np.zeros((len(x0),len(alpha_incident)))
    im_recons = np.zeros((len(x0),GISAXS_im.shape[1]))
    for j in qx_dimension:
        if j in skip_qx:
            pass
        else:
            for i in range(len(alpha_incident)):
                I1 = (GISAXS_im[:,:,i])[:,j]
                I1[np.abs(I1)==inf] = np.nan
                I1 = flipud(I1)
                I1[np.abs(log(I1))==inf] = np.nan
				if np.size(I1[np.isnan(I1)==0]) == 0:
                    pass
                    return np.zeros((len(x0)))
                I1 = np.interp(np.arange(0,len(I1),1),np.arange(0,len(I1),1)[isnan(I1)==0],I1[isnan(I1)==0])
                #figure(1),plot(qz,log(I1),qz_r,log(I1),qz_d,log(I1))
                fitting_range = qz[int(range_index_min[i]):int(range_index_max[i])]
                fitting_portion = I1[int(range_index_min[i]):int(range_index_max[i])]
                fitting_portion_model[:,i] = np.interp(fitting_range_model[:,i],fitting_range[isnan(log(fitting_portion))==0],fitting_portion[isnan(log(fitting_portion))==0])
            if np.size(initial)!=0:
                y0 = initial[:,j]
            else:
                y0 = np.log(fitting_portion_model[:,0])#np.interp(np.linspace(np.min(qz[:,1]),np.max(qz[:,1]),x0_length),qz[:,1],log(model)[:,1])
            if np.size(y0[isnan(y0)==1])==0:
                pass
            else:
                y0[isnan(y0)==1] = np.interp(x0[isnan(y0)==1],x0[isnan(y0)==0],y0[isnan(y0)==0])
            def fun(y,x=x0,I1=log(fitting_portion_model),qz1=fitting_range_model,alpha_incident=alpha_incident,
                qz_d=qz_d,qz_r=qz_r,t_f=t_f,r_f=r_f,reflc_params=reflc_params,trans_params=trans_params):
                norm_judge = np.zeros((len(alpha_incident),))
                for i in range(len(alpha_incident)):
                    I = I1[:,i]
                    qz = qz1[:,i]
                    if np.size(I[isnan(I)==1])==0:
                       pass
                    else:
                       I[isnan(I)==1] = np.interp(qz[isnan(I)==1],qz[isnan(I)==0],I[isnan(I)==0])

                    qz_max=np.nanmax(qz_d[:,i])
                    qz_min=np.nanmin(qz_r[:,i])
                    I_direct = np.interp(qz,
                            qz_d[np.nanargmin(np.abs(qz_d[:,i]-qz_min)):np.nanargmin(np.abs(qz_d[:,i]-qz_max)),i],
                            y[np.nanargmin(np.abs(qz_d[:,i]-qz_min)):np.nanargmin(np.abs(qz_d[:,i]-qz_max))])
                    I_reflect = np.interp(qz,
                            qz_r[np.nanargmin(np.abs(qz_r[:,i]-qz_min)):np.nanargmin(np.abs(qz_r[:,i]-qz_max)),i],
                            y[np.nanargmin(np.abs(qz_r[:,i]-qz_min)):np.nanargmin(np.abs(qz_r[:,i]-qz_max))])
                    #print I_reflect.shape,qz.shape,qz_r.shape,y.shape,
                    #return I_reflect,I_direct,qz_max,qz_min
                    fitting_portion =I#I[np.nanargmin(np.abs(qz-qz_min)):np.nanargmin(np.abs(qz-qz_max))]
                    norm_judge[i] = norm(fitting_portion-log(trans_params[:,i]**2*t_f[i]**2*exp(I_direct)+
                         t_f[i]**2*reflc_params[:,i]**2*exp(I_reflect)+trans_params[:,i]**2*r_f[i]**2*exp(I_reflect)+r_f[i]**2*reflc_params[:,i]**2*exp(I_direct)))

                return np.sum(norm_judge)
            ret = minimize(fun, y0,method="L-BFGS-B",options={'maxfun':iterations})
            im_recons[:,j]=exp(ret.x)
    return im_recons

#print time.time()-t
