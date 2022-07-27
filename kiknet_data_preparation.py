from numba import jit
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pt
import csv
from math import sqrt
import functools
import operator
import time


@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result


@jit(nopython=True)
def nije_rs(acceleration, time_step, periods, damping ,units="cm/s/s"):
    """
    Define the response spectrum
    """
    #global num_steps, num_per, acceleration_2, d_t
    
    
    periods = periods
    num_per = len(periods)
    acceleration_2 = acceleration
    damping = damping
    d_t = time_step
    num_steps = len(acceleration)
    
    omega = (2. * np.pi) / periods
    omega2 = omega ** 2.
    omega3 = omega ** 3.
    omega_d = omega * sqrt(1.0 - (damping ** 2.))
    const = {'f1': (2.0 * damping) / (omega3 * d_t),
            'f2': 1.0 / omega2,
            'f3': damping * omega,
            'f4': 1.0 / omega_d}
    const['f5'] = const['f3'] * const['f4']
    const['f6'] = 2.0 * const['f3']
    const['e'] = np.exp(-const['f3'] * d_t)
    const['s'] = np.sin(omega_d * d_t)
    const['c'] = np.cos(omega_d * d_t)
    const['g1'] = const['e'] * const['s']
    const['g2'] = const['e'] * const['c']
    const['h1'] = (omega_d * const['g2']) - (const['f3'] * const['g1'])
    const['h2'] = (omega_d * const['g1']) + (const['f3'] * const['g2'])
    
    x_d = np.zeros((num_steps - 1, num_per))
    x_v = np.zeros((num_steps - 1, num_per))
    x_a = np.zeros((num_steps - 1, num_per))
    
    for k in range(0, num_steps - 1):
        yval = k - 1
        dug = acceleration_2[k + 1] - acceleration_2[k]
        z_1 = const['f2'] * dug
        z_2 = const['f2'] * acceleration_2[k]
        z_3 = const['f1'] * dug
        z_4 = z_1 / d_t
        if k == 0:
            b_val = z_2 - z_3
            a_val = (const['f5'] * b_val) + (const['f4'] * z_4)
        else:    
            b_val = x_d[k - 1, :] + z_2 - z_3
            a_val = (const['f4'] * x_v[k - 1, :]) +\
                (const['f5'] * b_val) + (const['f4'] * z_4)

        x_d[k, :] = (a_val * const['g1']) + (b_val * const['g2']) +\
            z_3 - z_2 - z_1
        x_v[k, :] = (a_val * const['h1']) - (b_val * const['h2']) - z_4
        x_a[k, :] = (-const['f6'] * x_v[k, :]) - (omega2 * x_d[k, :])    
        
    
    response_spectrum =  (omega ** 2.) * np_apply_along_axis(np.max, 0,np.abs(x_d)) 
    
    return response_spectrum







def interp_vs(x,depth=[],vs=[]):
    return vs[np.argmax(depth>x)-1]
#print(pt.Path(__file__).parent.absolute())

kiknet_soil_folder = "./sitedat_kiknet/"
knet_soil_folder = "./sitedat_knet/"

kiknet_soil_file_list = os.listdir(kiknet_soil_folder)
kiknet_soil_file_list[0]

knet_soil_file_list = os.listdir(knet_soil_folder)
knet_soil_file_list[0]

df=pd.read_csv('sitepub_all_en_fixed.csv', sep=',',header=None)
sites = df.values
sites.shape
sites[0]
"""  x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
             dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]) """

 
all_st = []
kiknet_st_names =[]
i=0     
for each_st in kiknet_soil_file_list:
    try:
        #print(i)
        each_st_arr = np.array([tuple(x.replace('\n','').replace('-----','-1.0').split(',')) for x in 
                open(kiknet_soil_folder + each_st).readlines()[2:]],
                dtype=[('No',np.int32),('Thickness (m)',np.float32),('Depth (m)',np.float32)
                ,('Vp (m/s)',np.float32),('Vs (m/s)',np.float32)])
        all_st.append(each_st_arr)
        kiknet_st_names.append(each_st.replace('.dat',''))
    except:
        i=i+1
    
 

all_st_arr = np.array(all_st)
all_st_arr.shape
all_st_arr[0]
all_st_arr[1]['Vs (m/s)']

kiknet_st_arr = []

for i in np.arange(all_st_arr.shape[0]):
    vs = np.zeros(all_st_arr[i]['Vs (m/s)'].shape,dtype=np.float32)
    depth = np.zeros(all_st_arr[i]['Depth (m)'].shape,dtype=np.float32)
    np.copyto(vs,all_st_arr[i]['Vs (m/s)'])
    np.copyto(depth,all_st_arr[i]['Depth (m)'])
    depth[-1]=depth[-2]*1.1  # last value of depth is '-1' making it 1.1 times of last acceptable value
    kiknet_st_arr.append([kiknet_st_names[i],
        sites[np.where(kiknet_st_names[i]==sites[:,0:1].reshape((-1,)))[0]][0][2],
        sites[np.where(kiknet_st_names[i]==sites[:,0:1].reshape((-1,)))[0]][0][3],        
        interp_vs(10,depth,vs),interp_vs(20,depth,vs),interp_vs(30,depth,vs)])

kiknet_st_arr = np.array(kiknet_st_arr)
kiknet_st_arr.shape
kiknet_st_arr[5]
#np.save('kiknet_st_arr.npy',kiknet_st_arr)


my_sample_file = open('D:/00-GENEL/08-MASTER/00-TEZ/04_Data_Jap/kiknet_eq/2020_01_28-2020_03_31_1200/ABSH062002131934.EW1').readlines()
my_sample_file[13]
#scale factor 1
float(my_sample_file[13].split()[2][0:my_sample_file[13].split()[2].find('(')])
#scale factor 2
float(my_sample_file[13].split()[2][my_sample_file[13].split()[2].find(')')+2:-1])
sample_acc = [x.split() for x in my_sample_file[17:]]
sample_acc

sample_acc = np.array(sample_acc,dtype=np.float32).reshape((-1,))
sample_acc
sample_acc = sample_acc * 2940. / 6170270. 
sample_acc_off = sample_acc - np.mean(sample_acc)
np.max(sample_acc_off)
np.min(sample_acc_off)

plt.plot(sample_acc_off)



#response_spectrum = nije_rs(acceleration = sample_acc_off.reshape((-1,1)),time_step= dt, periods = periods, damping=dmpng)

#plt.plot(periods,response_spectrum)
#(sample_acc-np.mean(sample_acc))[0:10]
os.listdir('D:/00-GENEL/08-MASTER/00-TEZ/04_Data_Jap/kiknet_eq/')

file_lst = [x.replace('.EW1','') for x in os.listdir('D:/00-GENEL/08-MASTER/00-TEZ/04_Data_Jap/kiknet_eq/') if 'EW1' in x]
len(file_lst)
file_lst

"""
The strong-motion data for KiK-net sites have six channel numbers;
 the first three correspond to three components of a borehole 
 seismograph and the other three correspond to those of a surface 
 seismograph (see table below). Please note that channels 1 and 2 
 may be rotated from NS and EW directions due to the difficulty in
 adjusting the azimuth of sensors in the borehole. 
  Orientations of Hi-net sensor, which are considered to be the same 
  as those of KiK-net sensor installed in the same container, 
  are available from the Hi-net website (in Japanese) and serve 
  as a reference for KiK-net data.

Channel number	Component	Location	Extension of file
1	NS	Borehole	NS1
2	EW	Borehole	EW1
3	UD	Borehole	UD1
4	NS	Surface	    NS2
5	EW	Surface	    EW2
6	UD	Surface	    UD2
 """

""" 
Origin Time       2020/02/13 19:34:00
Lat.              45.055
Long.             149.162
Depth. (km)       155
Mag.              7.2
Station Code      ABSH06
Station Lat.      44.2146
Station Long.     143.6202
Station Height(m) 7
Record Time       2020/02/13 19:34:49
Sampling Freq(Hz) 100Hz
Duration Time(s)  290
Dir.              2
Scale Factor      2940(gal)/6170270
Max. Acc. (gal)   0.797
Last Correction   2020/02/13 19:34:34
 """

#file_lst[0][:6]




def create_data_arr(each_file, ch):
    each_file_longname = each_file + '.' + ch
    acc_file = open('D:/00-GENEL/08-MASTER/00-TEZ/04_Data_Jap/kiknet_eq/'+each_file_longname).readlines()
    eq_name = each_file[6:]
    st_code = each_file[:6]
    channel = ch
    origin_time = acc_file[0].replace('\n','').split()[2]+' '+acc_file[0].replace('\n','').split()[3]
    eq_lat = float(acc_file[1].replace('\n','').split()[1])
    eq_long = float(acc_file[2].replace('\n','').split()[1])
    eq_depth = float(acc_file[3].replace('\n','').split()[2])
    eq_mag = float(acc_file[4].replace('\n','').split()[1])
    st_lat = float(acc_file[6].replace('\n','').split()[2])
    st_long = float(acc_file[7].replace('\n','').split()[2])
    st_height = float(acc_file[8].replace('\n','').split()[2])
    record_time = acc_file[9].replace('\n','').split()[2]+' '+acc_file[9].replace('\n','').split()[3]
    sampling_freq = int(acc_file[10].replace('\n','').replace('Hz','').split()[2])
    duration_time = float(acc_file[11].replace('\n','').split()[2])
    direction = int(acc_file[12].replace('\n','').split()[1])
    scale_factor = float(acc_file[13].split()[2][0:acc_file[13].split()[2].find('(')])/float(acc_file[13].split()[2][acc_file[13].split()[2].find(')')+2:-1])
    max_acc = float(acc_file[14].replace('\n','').split()[3])
    last_correction = acc_file[15].replace('\n','').split()[2]+' '+ acc_file[15].replace('\n','').split()[3]

    my_acc = [x.split() for x in acc_file[17:]]
    my_acc = functools.reduce(operator.iconcat, my_acc, [])
    my_acc = np.array(my_acc,dtype=np.float32).reshape((-1,))    
    my_acc = my_acc * scale_factor
    acc_data = my_acc - np.mean(my_acc)

    dt=1/sampling_freq
    periods = np.concatenate((np.linspace(0.01, 1, 500),np.linspace(1.05, 10, 100)))
    dmpng = 0.001

    response_spectrum = nije_rs(acceleration = acc_data.reshape((-1,1)),time_step= dt, periods = periods, damping=dmpng)

    each_file_arr = np.array(tuple([each_file_longname,eq_name,st_code,channel,
                                origin_time,eq_lat,eq_long,eq_depth,eq_mag,st_lat,
                                st_long,st_height,record_time,sampling_freq,
                                duration_time,direction,scale_factor,max_acc,last_correction,
                                acc_data,response_spectrum]),
        dtype=[('File Name',np.dtype('U25')),('Eq name',np.dtype('U25')),('Station Code',np.dtype('U25'))
        ,('Channel',np.dtype('U25')),('Origin Time',np.dtype('U25')),('Eq Lat.',np.float32)
        ,('Eq Long.',np.float32),('Eq Depth. (km)',np.float32),('Eq Mag.',np.float32)
        ,('Station Lat.',np.float32),('Station Long.',np.float32),('Station Height(m)',np.float32)
        ,('Record Time',np.dtype('U25')),('Sampling Freq(Hz)',np.int16),('Duration Time(s)',np.float32)
        ,('Dir.',np.int16),('Scale Factor',np.float32),('Max. Acc. (gal)',np.float32)
        ,('Last Correction',np.dtype('U25')),('Acc Data',np.object),('Response Spectrum',np.object)])
    return each_file_arr

data_jp_all_acc_arr = []
file_lst[0]

i=0
for each_file in file_lst:
    
    ew1 = create_data_arr(each_file,'EW1')
    data_jp_all_acc_arr.append(ew1)
    ew2 = create_data_arr(each_file,'EW2')
    data_jp_all_acc_arr.append(ew2)
    ns1 = create_data_arr(each_file,'NS1')
    data_jp_all_acc_arr.append(ns1)
    ns2 = create_data_arr(each_file,'NS2')
    data_jp_all_acc_arr.append(ns2)
    ud1 = create_data_arr(each_file,'UD1')
    data_jp_all_acc_arr.append(ud1)
    ud2 = create_data_arr(each_file,'UD2')
    data_jp_all_acc_arr.append(ud2)
    i=i+1
    print(i)


len(my_acc)

len(data_jp_all_acc_arr)
data_jp_all_acc_arr[3:5]['File Name']
arr = np.array(data_jp_all_acc_arr)
arr['File Name'].shape
plt.plot(periods,data_jp_all_acc_arr[1000]['Response Spectrum'][0])
