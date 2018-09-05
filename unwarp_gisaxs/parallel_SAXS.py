# construct pattern of direct beam and reflect from a model
# build a one-d model
import glob
import os
import numpy as np
import matplotlib.pyplot as plt # for showing image
#from pylab import * # for multiple figure window
from skimage import io
import re
import statsmodels.api as sm
from scipy.optimize import minimize
from numpy.linalg import norm

#import time

#t = time.time()
# qx_dimension determined by the input image qx dimension
#qx_dimension = 980
# skip_qx correlate the certain qx position completely masked along qz direction which enable to reconstrcut.
#skip_qx =  np.concatenate([np.arange(180,245),np.arange(485,496)])

def SAXS_para_recons(qx_dimension=None,skip_qx=np.empty((0,0)),\
        alpha_incident=None,\
		GISAXS_im=None,x0=None,fitting_range_model=None,\
		qz=None,qz_r=None,qz_d=None,qz_f=None,\
		reflc_params=None,trans_params=None,r_f=None,t_f=None,\
		qz_min=None,qz_max=None,range_index_min=None,\
		range_index_max=None,initial = np.empty((0,0)), iterations=3000):
        fitting_portion_model = np.zeros((len(x0),len(alpha_incident)))
        j = qx_dimension
        if j in skip_qx:#,np.arange(550,580)]):
            pass
            return np.zeros((len(x0),))
        else:
            for i in range(len(alpha_incident)):
                I1 = (GISAXS_im[:,:,i])[:,j]
                I1[np.abs(I1)==np.inf] = np.nan
                I1 = np.flipud(I1)
                I1[np.abs(np.log(I1))==np.inf] = np.nan
                if np.size(I1[np.isnan(I1)==0]) == 0:
                    pass
                    return np.zeros((len(x0)))
                I1 = np.interp(np.arange(0,len(I1),1),np.arange(0,len(I1),1)[np.isnan(I1)==0],I1[np.isnan(I1)==0])
                #figure(1),plot(qz,log(I1),qz_r,log(I1),qz_d,log(I1))
                fitting_range = qz[int(range_index_min[i]):int(range_index_max[i])]
                fitting_portion = I1[int(range_index_min[i]):int(range_index_max[i])]
                fitting_portion_model[:,i] = np.interp(fitting_range_model[:,i],\
                                  fitting_range[np.isnan(np.log(fitting_portion))==0],\
                                  fitting_portion[np.isnan(np.log(fitting_portion))==0])
            #sys.exit()
            #model = fitting_portion_model
            #qz = fitting_range_model
            if np.size(initial)!=0:
                y0 = initial[:,j]
            else:
                y0 = np.log(fitting_portion_model[:,0])#np.interp(np.linspace(np.min(qz[:,1]),np.max(qz[:,1]),x0_length),qz[:,1],log(model)[:,1])
                #sys.exit()
                if np.size(y0[np.isnan(y0)==1])==0:
                    pass
                else:
                    y0[np.isnan(y0)==1] = np.interp(x0[np.isnan(y0)==1],x0[np.isnan(y0)==0],y0[np.isnan(y0)==0])
            def fun(y,x=x0,I1=np.log(fitting_portion_model),qz1=fitting_range_model,alpha_incident=alpha_incident,
                    qz_d=qz_d,qz_r=qz_r,t_f=t_f,r_f=r_f,reflc_params=reflc_params,trans_params=trans_params):
                norm_judge = np.zeros((len(alpha_incident),))
                for i in range(len(alpha_incident)):
                    I = I1[:,i]
                    qz = qz1[:,i]
                    if np.size(I[np.isnan(I)==1])==0:
                        pass
                    else:
                        I[np.isnan(I)==1] = np.interp(qz[np.isnan(I)==1],qz[np.isnan(I)==0],I[np.isnan(I)==0])
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
                    norm_judge[i] = norm(fitting_portion-np.log(trans_params[:,i]**2*t_f[i]**2*np.exp(I_direct)+
                                           t_f[i]**2*reflc_params[:,i]**2*np.exp(I_reflect)+trans_params[:,i]**2*r_f[i]**2*\
                                           np.exp(I_reflect)+r_f[i]**2*reflc_params[:,i]**2*np.exp(I_direct)))
                return np.sum(norm_judge)
            ret = minimize(fun, y0,method="L-BFGS-B",options={'maxfun':iterations})#,'maxiter':5,'maxls':5})
            #else:
            #if (j%100)==0:
            #print time.time()-t
        return ret.x

#skip_qx = np.empty((0,0))
from functools import partial
import multiprocessing

def parallel_SAXS_para_recons(qx_array=None,skip_qx = np.empty((0,0)),\
        alpha_incident=None,\
        GISAXS_im=None,x0=None,fitting_range_model=None,\
        qz=None,qz_r=None,qz_d=None,qz_f=None,\
        reflc_params=None,trans_params=None,r_f=None,t_f=None,\
        qz_min=None,qz_max=None,range_index_min=None,\
        range_index_max=None,initial = np.empty((0,0)), iterations=3000):
	para_func = partial(SAXS_para_recons,skip_qx=skip_qx,\
            alpha_incident=alpha_incident,GISAXS_im=GISAXS_im,\
            x0=x0,fitting_range_model=fitting_range_model,qz_r=qz_r,\
            qz=qz,qz_d=qz_d,qz_f=qz_f,reflc_params=reflc_params,\
            trans_params=trans_params,r_f=r_f,t_f=t_f,qz_min=qz_min,\
            qz_max=qz_max,range_index_min=range_index_min,\
            range_index_max=range_index_max,initial = initial, iterations=iterations)

	pool = multiprocessing.Pool()
	result = pool.map(para_func,qx_array)
	pool.close()
	im = np.zeros((len(x0),len(qx_array)))
	for i in range(len(result)):
		im[:,i] = np.exp(result[i])
	return im

'''
# the code below is test on local repository for checking validation of parameter and funtions
os.chdir('/Users/jiliangliu/Dropbox/GISAXS_code/example/')
q_reflc = np.load('reflc_n_trans_coef.npz')['q_reflc']
trans_index = (np.load('reflc_n_trans_coef.npz')['T01'])**.5
reflc = (np.load('reflc_n_trans_coef.npz')['R01'])**.5

detector_distance = 4.937849
wavelength = 0.9184
ratioDw = 29.27
ct_f =  0.0928039405254*0.9
ct_si = 0.135
k0 = 2*np.pi/wavelength

film_n = 1-(np.radians(ct_f)/2**.5)**2
ambient_n = 1.
alpha_incident = np.array([.12,.14,.16])
alpha_incident = np.radians(alpha_incident)
x0_length=200
#alpha_incident = np.radians(.15)
fitting_portion_model = np.zeros((x0_length,len(alpha_incident)))
x0 = np.linspace(0,.16,x0_length)

import time
t = time.time()

list1 = glob.glob('GISAXS_*')
shape_index = (io.imread(list1[0])).shape
ycenter = 686

qz = 2*np.pi*2*np.sin(np.arcsin((ycenter-np.arange(0,1043,1))*172*1e-6/detector_distance)/2)/wavelength
qz = np.flipud(qz)

from coefficient_calculation import coefficient_calculation
alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,\
fitting_range_model,qz_min,qz_max,range_index_min,range_index_max = \
		coefficient_calculation(x0,alpha_incident,ambient_n,\
					film_n,qz,q_reflc,reflc,\
					trans_index,k0)

#print np.degrees(alpha_incident),np.degrees(alpha_incident_eff)

qx_dimension = range(20) #range(shape_index[1])
skip_qx = np.concatenate([np.arange(180,245),np.arange(485,496)])

def GISAXS_concatenate(alpha_incident,list1,shape_index):
    im = np.zeros((shape_index[0],shape_index[1],len(alpha_incident)))
    for i in range(len(alpha_incident)):
        im[:,:,i] = io.imread(list1[i])
    return im

GISAXS_im = GISAXS_concatenate(alpha_incident,list1,shape_index)

w_initial = np.log(np.load('w_initial.npz')['init'])
if w_initial.shape[0] != x0_length:
    from skimage.transform import resize
    w_initial = resize(w_initial,(int(x0_length),int(w_initial.shape[1])))
from functools import partial
#from unwarp_gisaxs.parallel_SAXS import SAXS_para_recons,parallel_SAXS_para_recons

#skip_qx = np.empty((0,0))
para_func = partial(SAXS_para_recons,#skip_qx=skip_qx,\
		alpha_incident=alpha_incident,GISAXS_im=GISAXS_im,\
		x0=x0,fitting_range_model=fitting_range_model,qz_r=qz_r,\
		qz=qz,qz_d=qz_d,qz_f=qz_f,reflc_params=reflc_params,\
		trans_params=trans_params,r_f=r_f,t_f=t_f,qz_min=qz_min,\
		qz_max=qz_max,range_index_min=range_index_min,\
		range_index_max=range_index_max,initial = w_initial, iterations=1500)
import multiprocessing

pool = multiprocessing.Pool(multiprocessing.cpu_count())
result = pool.map(para_func,range(shape_index[1]))
pool.close()

im = np.zeros((len(x0),shape_index[1]))
for i in range(len(result)):
    im[:,i] = np.exp(result[i])
print(time.time()-t)

fig,ax = plt.subplots()
ax.imshow(np.log(im),vmin=0)
plt.show()
#print time.time()-t
'''
