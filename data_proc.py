#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:43:17 2018

@author: smrak
"""

import xarray
import numpy as np
from pandas import Timestamp
from datetime import timedelta
from pyGnss import gnssUtils as gu
from pyGnss import pyGnss

import matplotlib.pyplot as plt

#

obs = '/media/smrak/Eclipse2/mahali/2015-10-07/mah82800.15o.nc'
nav = '/media/smrak/Eclipse2/mahali/2015-10-07/brdc2800.15n'

sv = 'G23'
arg = 'L1'
el_mask = 30
skip = 10
porder = 8
forder = 6
fc = 0.1
fs = 1

D = xarray.open_dataset(obs, group='OBS')
# Dummy arrays
T = np.nan * np.ones(D.sizes['time'] - skip)
Ls = np.copy(T)
Ss = np.copy(T)
#
rx_xyz = D.position
leap_seconds = gu.getLeapSeconds(nav)
obstimes64 = D.time[skip:].values
obstimes_dt = np.array([Timestamp(t).to_pydatetime() for t in obstimes64]) \
                    - timedelta(seconds = leap_seconds)
Ds = D.sel(sv=sv)
L1 = Ds['L1'][skip:].values
S1 = Ds['S1'][skip:].values

aer = pyGnss.getSatellitePosition(rx_xyz,sv,obstimes_dt,nav,cs='aer',dtype='georinex')

idel = (aer[1] >= el_mask)
obstimes = obstimes64[idel]
L1 = L1[idel]
S1 = S1[idel]

L1d = pyGnss.phaseDetrend(L1, order=porder)
L1s = gu.hpf(L1d, order=forder, fc=fc, plot=False, fs=fs)
L1s[:20] = np.nan

S1d = pyGnss.phaseDetrend(S1, order=3)
S1dd, Td = gu.lpf(S1d,fc=0.005,group_delay=True)
S1hp = gu.hpf(S1d, order=forder, fc=0.01, plot=False, fs=fs)

idin = np.isin(obstimes64,obstimes)
T[idin] = obstimes
T = np.asarray(T, dtype='datetime64[ns]')
Ls[idin] = L1s
Ss[idin] = S1dd

plt.plot(obstimes, S1d)
plt.plot(obstimes - int(Td*1e9), S1dd)
plt.plot(obstimes, S1hp,'r')