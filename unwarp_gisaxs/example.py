# construct pattern of direct beam and reflect from a model
# build a one-d model
import glob
import os
import sys
#sys.path.append('/Users/jiliangliu/Dropbox/GISAXS_code/')
import numpy as np
import matplotlib.pyplot as plt # for showing image
from skimage import io
import re
import statsmodels.api as sm
#from pkg_resource import resource_filename

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
x0_length=300
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
'''
from unwarp_gisaxs.reconstruction import SAXS_recons
im = SAXS_recons(qx_dimension=range(10),skip_qx=skip_qx,\
		alpha_incident=alpha_incident,GISAXS_im=GISAXS_im,\
		x0=x0,fitting_range_model=fitting_range_model,qz_r=qz_r,\
		qz=qz,qz_d=qz_d,qz_f=qz_f,reflc_params=reflc_params,\
		trans_params=trans_params,r_f=r_f,t_f=t_f,qz_min=qz_min,\
		qz_max=qz_max,range_index_min=range_index_min,\
		range_index_max=range_index_max)
'''
os.chdir('/Users/jiliangliu/unwarp_gisaxs/unwarp_gisaxs')
'''
from functools import partial
from parallel_SAXS import SAXS_para_recons

#skip_qx = np.empty((0,0))
para_func = partial(SAXS_para_recons,skip_qx=skip_qx,\
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

'''
from parallel_SAXS import parallel_SAXS_para_recons
im = parallel_SAXS_para_recons(qx_array=range(shape_index[1]),skip_qx=skip_qx,\
		alpha_incident=alpha_incident,GISAXS_im=GISAXS_im,\
		x0=x0,fitting_range_model=fitting_range_model,qz_r=qz_r,\
		qz=qz,qz_d=qz_d,qz_f=qz_f,reflc_params=reflc_params,\
		trans_params=trans_params,r_f=r_f,t_f=t_f,qz_min=qz_min,\
		qz_max=qz_max,range_index_min=range_index_min,\
		range_index_max=range_index_max)#,initial = w_initial, iterations=1500)

print(time.time()-t)

fig,ax = plt.subplots()
ax.imshow(np.log(im),vmin=0)
plt.show()
