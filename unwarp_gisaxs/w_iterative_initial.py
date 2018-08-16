# construct pattern of direct beam and reflect from a model
# build a one-d model
import glob
import os
import sys
sys.path.append('/Users/jiliangliu/Dropbox/GISAXS_code/')
import numpy as np
import matplotlib.pyplot as plt # for showing image
from pylab import * # for multiple figure window
from skimage import io
import re
import statsmodels.api as sm

os.chdir('/Users/jiliangliu/Dropbox/GISAXS_code/example')
#reflc = np.load('reflecvtivity_coefficient.npz')['R01_am']
#trans_index = np.load('reflecvtivity_coefficient.npz')['T01_am']
#q_reflc = np.load('reflecvtivity_coefficient.npz')['q_reflc']

q_reflc = np.load('reflc_n_trans_coef.npz')['q_reflc']
trans_index = (np.load('reflc_n_trans_coef.npz')['T01'])**.5
reflc = (np.load('reflc_n_trans_coef.npz')['R01'])**.5

detector_distance = 4.937849
wavelength = 0.9184
ratioDw = 29.27
ct_f =  0.0928039405254*0.9
ct_si = 0.135
k0 = 2*pi/wavelength

list1 = glob.glob('GISAXS_*')
film_n = 1-(np.radians(ct_f)/2**.5)**2
ambient_n = 1.
#alpha_incident = np.array([.14,.16,.18])
alpha_incident = np.array([.14])
alpha_incident = np.radians(alpha_incident)
x0_length=300
#alpha_incident = np.radians(.15)
fitting_portion_model = np.zeros((x0_length,len(alpha_incident)))

import time
t = time.time()

shape_index =(io.imread(list1[0])).shape
ycenter = 686

qz = 2*pi*2*np.sin(np.arcsin((ycenter-np.arange(0,1043,1))*172*1e-6/detector_distance)/2)/wavelength
qz = flipud(qz)

x0 = np.linspace(0,qz[-1]-k0*np.sin(2*alpha_incident),x0_length)#np.linspace(0,k0*sin(np.radians(4)),x0_length) 
#os.chdir('/Users/jiliangliu/Dropbox/GISAXS_code/')
from coefficient_calculation import coefficient_calculation
alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,\
fitting_range_model,qz_min,qz_max,range_index_min,range_index_max = \
		coefficient_calculation(x0,alpha_incident,ambient_n,\
					film_n,qz,q_reflc,reflc,\
					trans_index,k0)

min_indx = np.argmin(np.abs(qz-qz_min))
max_indx = np.argmin(np.abs(qz-qz_max))
overlap_range = qz[min_indx:max_indx]
qz_usable = np.linspace(qz[min_indx],qz[max_indx],x0_length)
qx = 2*pi*2*np.sin(np.arcsin((589-np.arange(0,shape_index[1],1))*172*1e-6/detector_distance)/2)/wavelength
qx = np.flipud(qx) 
#print np.degrees(alpha_incident),np.degrees(alpha_incident_eff)

#qx_dimension = range(shape_index[1])
#skip_qx = np.concatenate([np.arange(180,245),np.arange(485,496)])

gisaxs = np.flipud(io.imread(list1[0])).astype(float)+1.
mask = (gisaxs>np.inf)
mask[193:212,:] = True
mask[406:424,:] = True
mask[617:636,:] = True
mask[830:848,:] = True
mask[:,486:494] = True
#mask[210:670,580:600] = True
gisaxs[mask] = np.nan
xv,yv = np.meshgrid(np.arange(0,np.shape(gisaxs)[1]),np.arange(0,np.shape(gisaxs)[0]))

delete_row = np.concatenate([np.arange(193,213),np.arange(406,425),np.arange(617,637),np.arange(830,849)])
delete_col = np.arange(486,495)

gisaxs_interpolated_1 = np.delete(gisaxs,delete_row,0)
gisaxs_interpolated_1 = np.delete(gisaxs_interpolated_1,delete_col,1)
xv_interpolated_1 = np.delete(xv,delete_row,0)
xv_interpolated_1 = np.delete(xv_interpolated_1,delete_col,1)
yv_interpolated_1 = np.delete(yv,delete_row,0)
yv_interpolated_1 = np.delete(yv_interpolated_1,delete_col,1)

from scipy import interpolate
gisaxs_interp_mask = interpolate.interp2d(xv_interpolated_1[0,:],yv_interpolated_1[:,0],gisaxs_interpolated_1,kind='cubic')
gisaxs_no_mask = np.abs(gisaxs_interp_mask(xv[0,:],yv[:,0]))

gisaxs_usable = gisaxs_no_mask[min_indx:max_indx,:]

gisaxs_interp = interpolate.interp2d(qx,overlap_range,gisaxs_usable,kind='cubic')
incident_angle = np.radians(0.14)
gisaxs_new = gisaxs_interp(qx,qz_usable)
Ri = np.ones(shape(gisaxs_new))*reflc[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff)))]
Ti = np.ones(shape(gisaxs_new))*trans_index[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff)))]
Rf = np.interp(np.linspace(np.min(qz_f),np.max(qz_f),x0_length),\
               q_reflc[np.argmin(np.abs(q_reflc-np.min(qz_f))):np.argmin(np.abs(q_reflc-np.max(qz_f)))],\
               reflc[np.argmin(np.abs(q_reflc-np.min(qz_f))):np.argmin(np.abs(q_reflc-np.max(qz_f)))])
Rf = np.tile(Rf,(981,1)).T
Tf = np.interp(np.linspace(np.min(qz_f),np.max(qz_f),x0_length),\
               q_reflc[np.argmin(np.abs(q_reflc-np.min(qz_f))):np.argmin(np.abs(q_reflc-np.max(qz_f)))],\
               trans_index[np.argmin(np.abs(q_reflc-np.min(qz_f))):np.argmin(np.abs(q_reflc-np.max(qz_f)))])
Tf = np.tile(Tf,(981,1)).T

Tc = Ti*Tf+Ri*Rf
Rc = Ri*Tf+Ti*Rf

w_guess = np.ones(shape(gisaxs_new))*0.44

for i in range(1):
    It = gisaxs_new/( Tc + Rc/w_guess - Rc )
    Ir = (gisaxs_new - Tc*It)/Rc
    
    Qz_r = k0*sin(np.arccos(cos(2*arcsin(qz_usable/2/k0)-alpha_incident)*\
                    ambient_n/film_n)-alpha_incident_eff)
    
    Qz_d = k0*sin(np.arccos(cos(2*arcsin(qz_usable/2/k0)-alpha_incident)*\
                    ambient_n/film_n)+alpha_incident_eff)
    
    shift_indx = np.argmin(np.abs(Qz_r-np.min(Qz_d)))
    It_R = np.append(np.ones((shift_indx,981))*np.nan,It,axis=0)
    #It_R = np.tile(It_R,(1,256)).T
    Ir_R = np.append(Ir,np.ones((shift_indx,981))*np.nan,axis=0)
    #Ir_R = np.tile(Ir_R,(1,256)).T
    delta = It_R - Ir_R
    delta_max_current = np.nanmax(np.abs(delta),axis=0)
    #i=0
    spread = np.tile(delta_max_current*0.5/(1+i),(x0_length+shift_indx,1))
    I_R = np.zeros((shape(delta)))
    I_R[:shift_indx,:] = Ir_R[:shift_indx,:]
    I_R[-shift_indx:,:] = It_R[-shift_indx:,:]
    def m_response(delta, spread=10):
        
        return 1.0/( 1.0 + np.exp( -delta/spread ) )
    ms = m_response(delta, spread=spread)  
    I_R[shift_indx:-shift_indx,:] = ms[shift_indx:-shift_indx,:]*\
                                    Ir_R[shift_indx:-shift_indx,:]+\
                                    (1-ms[shift_indx:-shift_indx,:])*\
                                    It_R[shift_indx:-shift_indx,:] 
    Ir_new = I_R[:-shift_indx]
    It_new = I_R[shift_indx:]
    
    w_guess = It_new/(It_new+Ir_new)


from skimage.transform import resize
saxs_guess = resize(I_R,(300,981))
#saxs = np.load('simulate_gisaxs_14_18.npz')['saxs']
#saxs_resize = resize(np.flipud(saxs)[:216,:],(150,256))
#w_guess = np.ones(len(qz_useable))*0.44

fig,ax =plt.subplots()
ax.imshow(np.flipud(np.log(saxs_guess)),vmin=1,vmax=10)
plt.xticks([11,69,128,186,244],[r'$\rm{-}$0.10',r'$\rm{-}$0.05','0.00','0.05','0.10'],fontsize=24)
plt.yticks([149,110,69,29],['0.00','0.05','0.10','0.15'],fontsize=24)
plt.xlabel(r'$\rm{Q_{x}}\,\,\rm{\AA^{-1}}$',{'fontname':'crusive','fontsize':36})
plt.ylabel(r'$\rm{Q_{y}}$',fontsize=36)
plt.title(r'$\rm{iter\,\,1}$',fontsize=32)
plt.tick_params(pad=3)
plt.tight_layout()
plt.show()
