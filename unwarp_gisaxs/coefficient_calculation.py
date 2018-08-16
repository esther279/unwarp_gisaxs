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


'''
The meaning of input:

	x0 correlates to a downsampled qz coordinate
	alpha_incident correlates to the incident angle recorded from experiment
	ambient_n correlates to 
	film_n correlates to 
	qz is reciprocal coordinate calculate from detector space
	reflc correlate to the reflectivity of sample, could be measured or estimated
	q_reflc reciprocal coordinate correlate to the reflectivity curve
	trans_index is transmission coefficient which could be fitted or estimated basing on reflectivity of sample, the reciprocal coordinate is similar to the reflc
	
coefficient is precalculate the effect of distortion of refraction. The refraction only effects the scattering along the z direction
'''


def coefficient_calculation(x0,alpha_incident,ambient_n,film_n,qz,q_reflc,reflc,trans_index,k0):
    x0_length = len(x0)
    qz_r = np.zeros((len(x0),len(alpha_incident)))
    qz_d = np.zeros((len(x0),len(alpha_incident)))
    qz_f = np.zeros((len(x0),len(alpha_incident)))
    reflc_params = np.zeros((len(x0),len(alpha_incident)))
    trans_params = np.zeros((len(x0),len(alpha_incident)))
    r_f = np.zeros((len(alpha_incident),))
    t_f = np.zeros((len(alpha_incident),))
    alpha_incident_eff = np.zeros((len(alpha_incident),))
    qz_min = np.zeros((len(alpha_incident),)) 
    qz_max = np.zeros((len(alpha_incident),))
    range_index_min = np.zeros((len(alpha_incident),))
    range_index_max = np.zeros((len(alpha_incident),))
    fitting_range_model = np.zeros((len(x0),len(alpha_incident)))
    for i in range(len(alpha_incident)):
            
        # Breiby paper a=only consider two term for DWBA
        #direct beam
        alpha_incident_eff[i] = np.arccos(cos(alpha_incident[i])*ambient_n/film_n)
        two_theta = 2*arcsin(x0/2/k0)
        exit_angle_r = two_theta - alpha_incident_eff[i]
        exit_angle_r[exit_angle_r<0] = np.nan
        exit_angle_r_real = np.arccos(cos(exit_angle_r)*film_n/ambient_n)
        two_theta_r_final = exit_angle_r_real + alpha_incident[i]
        qz_d[:,i] = 2*k0*sin(two_theta_r_final/2.)
        
        # reflect neam
        #alpha_incident_eff = np.arccos(cos(alpha_incident[i])*ambient_n/film_n)
        #two_theta = 2*arcsin(x/2/k0)
        exit_angle_d = two_theta + alpha_incident_eff[i]
        exit_angle_d[exit_angle_d<0] = np.nan
        exit_angle_d_real = np.arccos(cos(exit_angle_d)*film_n/ambient_n)
        two_theta_d_final = exit_angle_d_real + alpha_incident[i]
        qz_r[:,i] = 2*k0*sin(two_theta_d_final/2.)
        
        
        qz_max[i] = np.nanmax(qz_d[:,i])
        qz_min[i] = np.nanmin(qz_r[:,i])
        
        fitting_range = qz[np.nanargmin(np.abs(qz-qz_min[i])):np.nanargmin(np.abs(qz-qz_max[i]))]
        fitting_range_model[:,i] = np.interp(np.linspace(np.min(fitting_range),np.max(fitting_range),num=x0_length),fitting_range,fitting_range)
        
        qz_f[:,i] = 2*k0*sin(np.arccos(cos(2*arcsin(fitting_range_model[:,i]/2/k0)-alpha_incident[i])*ambient_n/film_n))
        #print qz_f.shape,fitting_range_model.shape
        reflc_params[:,i] = np.interp(qz_f[:,i],
                                q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))],
                                reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))])
        trans_params[:,i] = np.interp(qz_f[:,i] ,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))],
                            trans_index[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))])
        r_f[i] = reflc[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff[i])))]
        t_f[i] = trans_index[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff[i])))]
        range_index_min[i] = np.nanargmin(np.abs(qz-qz_min[i]))
        range_index_max[i] = np.nanargmin(np.abs(qz-qz_max[i]))
    return alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,fitting_range_model,qz_min,qz_max,range_index_min,range_index_max


'''
The meaning of output:
	alpha_incident_eff: real incident angle after refraction correction
	qz_r: refraction corrected qz for reflect beam channel
	qz_d: refraction corrected qz for direct beam channel
	qz_f: refraction corrected q_reflc
	reflc_params: reflectivity correlate to qz_f
	trans_params: transmission correlate to qz_f
	r_f: reflectivity at specific qz correlated to alpha_incident_eff
	t_f: transmission at specific qz correlated to alpha_incident_eff
	fitting_range_model: re-interpolated qz enable the same array length for late fitting process
	qz_min: minimum of qz_r determine the lower bound of fitting_range_model
	qz_max: maximum of qz_f determine the high bound of fitting_range_model
	range_index_min: array index of qz_min help to re-interpolate fitting_range_model
	range_index_max: array index of qz_max help to re-interpolate fitting_range_model
'''

'''
alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,fitting_range_model,\
qz_min,qz_max,range_index_min,range_index_max =  coefficient_calculation(x0,alpha_incident,
                                                                ambient_n,film_n,qz,q_reflc,reflc,trans_index)
'''
