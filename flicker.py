import numpy as np
from scipy import signal

import pandas as pd
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
##Subfunction: get_percentile
def get_percentile(limit):
    idx = np.argmin(abs(cpf_cum_probability - limit))
    return cpf_magnitude[idx]


import matplotlib.pyplot as plt

#df=pd.read_excel(r'C:\Users\sha79576\Desktop\Amrit Plots\June_With_USVI_II_flicker_Mon_8b_vi.xlsx',parse_cols='B,C')

x = 24.170 /100     #  Voltage Deviation   delta V/V
ff = 40.0    #  Fluctuating frequency
fs = 4000.0
f_line = 60.0

time = np.arange(0,600,1/fs)
voltage = 1.0 * np.sin(np.pi *2* f_line* time)*(1.0 + x*0.5*np.sign(np.sin(np.pi*2*ff*time)))

#plt.plot(time, voltage)


####################BLOCK-1###########################################

## Removing DC Component

voltage = voltage - np.mean(voltage)

## Normalizing input

voltage_rms = np.sqrt(np.mean(voltage**2))

voltage = voltage / (voltage_rms * np.sqrt(2))

#plt.plot(time, voltage)

###################BLOCK-2###########################################

voltage_block2 = voltage**2

#plt.plot(time, voltage)


########### BLOCK-3 #######################
# Block 3: Bandpass and weighting filter
HIGHPASS_ORDER  = 1.0
HIGHPASS_CUTOFF = 0.05

LOWPASS_ORDER = 6.0
if (f_line == 50.0):
    LOWPASS_CUTOFF = 35.0

if (f_line == 60.0):
    LOWPASS_CUTOFF = 42.0

#subtract DC component to limit filter transients at start of simulation

voltage_block2_ac = voltage_block2 - np.mean(voltage_block2)

#plt.plot(time, voltage_block2_ac)

##################################################################
b_hp, a_hp = signal.butter(HIGHPASS_ORDER, HIGHPASS_CUTOFF/ (fs / 2), 'high', analog =0)
voltage_block2_ac_hp = signal.lfilter(b_hp, a_hp, voltage_block2_ac)

#plt.plot(time, voltage_block2_ac_hp)

#####################################################################
# smooth start of signal to avoid filter transient at start of simulation
smooth_limit = int(min(np.round(fs / 10.0), len(voltage_block2_ac_hp)))
voltage_block2_ac_hp[0 : smooth_limit] = voltage_block2_ac_hp[0 : smooth_limit] * np.linspace(0, 1, smooth_limit)

b_bw, a_bw = signal.butter(LOWPASS_ORDER, LOWPASS_CUTOFF / (fs / 2), 'low', analog=0)
voltage_bw = signal.lfilter(b_bw, a_bw, voltage_block2_ac_hp)

#plt.plot(time, voltage_bw)


####weighting filter

if (f_line == 50.0):

    K = 1.74802
    LAMBDA = 2 * np.pi * 4.05981
    OMEGA1 = 2 * np.pi * 9.15494
    OMEGA2 = 2 * np.pi * 2.27979
    OMEGA3 = 2 * np.pi * 1.22535
    OMEGA4 = 2 * np.pi * 21.9

if (f_line == 60.0):
    K = 1.6357
    LAMBDA = 2 * np.pi * 4.167375
    OMEGA1 = 2 * np.pi * 9.077169
    OMEGA2 = 2 * np.pi * 2.939902
    OMEGA3 = 2 * np.pi * 1.394468
    OMEGA4 = 2 * np.pi * 17.31512

num1 = (K * OMEGA1, 0.0)
den1 = (1.0, 2.0 * LAMBDA, OMEGA1**2)
num2 = (1.0 / OMEGA2, 1.0)
den2 = (1.0 / (OMEGA3 * OMEGA4), (1.0 / OMEGA3) + (1.0 / OMEGA4), 1.0)

b_w, a_w = signal.bilinear(np.convolve(num1, num2), np.convolve(den1, den2), fs)

voltage_w = signal.lfilter(b_w, a_w, voltage_bw)

#plt.plot(time, voltage_w)

#########################################################################
# Block 4: Squaring and smoothing

LOWPASS_2_ORDER  = 1.0
LOWPASS_2_CUTOFF = 1.0 / (2.0 * np.pi * 300e-3)  # time constant 300 msec
SCALING_FACTOR = 1238400.0  # scaling of output to perceptibility scale  

voltage_q = voltage_w**2;

b_lp, a_lp = signal.butter(LOWPASS_2_ORDER, LOWPASS_2_CUTOFF / (fs / 2), 'low', analog = 0)
s = SCALING_FACTOR * signal.lfilter(b_lp, a_lp, voltage_q)

#plt.plot(time, s)

# Block 5: Statistical evaluation

NUMOF_CLASSES = 10000

bin_cnt, cpf_magnitude = np.histogram(s, NUMOF_CLASSES)
cpf_magnitude = cpf_magnitude[1:]
cpf_cum_probability = 100.0 * (1 - np.cumsum(bin_cnt) / np.sum(bin_cnt))

#
p_50s = np.mean([get_percentile(30), get_percentile( 50), get_percentile( 80)])
p_10s = np.mean([get_percentile( 6), get_percentile( 8), get_percentile( 10), get_percentile( 13), get_percentile(17)])
p_3s = np.mean([get_percentile( 2.2), get_percentile(3), get_percentile(4)])
p_1s = np.mean([get_percentile( 0.7), get_percentile( 1), get_percentile(1.5)])
p_0_1 = get_percentile(0.1)

P_st = np.sqrt(0.0314 * p_0_1 + 0.0525 * p_1s + 0.0657 * p_3s + 0.28 * p_10s + 0.08 * p_50s)
print(P_st)
#plt.plot(time,P_st)

plt.plot(cpf_magnitude, cpf_cum_probability)
#plt.plot( cpf_cum_probability, cpf_magnitude)

