#%% ==============================================================
# import libraries
# ================================================================
import numpy as np                    # mathematics
import matplotlib.pyplot as plt       # plot
import pandas as pd                   # pandas dataframe
import os                             # files sort
import glob                           # glob
import datetime                       # process timestamp
import time                           # time                       
from scipy.fft import   fft, fftfreq  # function fo fast fourier
from scipy.constants import g         # gravity constant
from scipy.signal import find_peaks, windows  # signar processsing

#%% ===============================================================
# functions
# =================================================================
def dft(X, fr):
  ''' get a fft from a matrix of data X, 
  row measurement, columns sensors
  X = amplitude matrix on time domain
  dt = sampling frequency
  '''
  n = len(X)                          # quanttity of data
  fourier = fft(X, axis=0)[0:n//2, :] # amplitude fourier transform
  Xf = 2.0/n*np.abs(fourier)          # amplitude
  Xf[:5] = 0                          # take off first values
  f = fftfreq(n, 1/fr)[0:n//2]        # frequency values
  return f, Xf
def plot_vib(df, haxis,  name = '', start = time.time(),
    xlabel ='time [s]', case='time'):
  ''' Function to plot vibration in time or frequency '''
  # get time -----------------------------------------------------
  dt_object = datetime.datetime.fromtimestamp(start)
  date_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

  # change the y limits based on case ----------------------------
  if case=='time':
    max_df = np.max(np.array(np.abs(df)))*1.05
    min_df = - max_df
  else:
    max_df = np.array(df).max()*1
    min_df = 0

  # plot ---------------------------------------------------------
  fig, axs = plt.subplots(df.shape[1], 1, sharex=True)
  fig.suptitle('sensor %s, on %s'%(name, date_time))
  for ax, col in zip(axs, df.columns):
    vib = np.array(df[col])
    ax.plot(haxis, vib, color='k', )
    ax.set_ylabel('%s [$\\mathrm{m^2/s}$]'%(col))
    ax.set_xlim(haxis.min(), haxis.max())
    ax.set_ylim(min_df, max_df)
  ax.set_xlabel(xlabel)
  fig.tight_layout()
  return fig, axs
RMS = lambda x: (np.sum(x**2, axis=0)/x.shape[0])**0.5

#%% ===============================================================
# import data
# =================================================================
data_paths = glob.glob(os.sep.join(['data', 'part1', '*.csv']))
sensors = {}   # dictionary to save all data
for path in data_paths:
  file_name = path.split(os.sep)[-1] 
  start, interval, name = file_name.split('-') # get data inf
  name = name[:-4]                             # sensor name
  sensors[name] = {
    'start' : int(start),
    'interval': int(interval),
    'data': pd.read_csv(path)
  }

#%% ===============================================================
# data in time domain
# =================================================================
for name in sensors:
  data = sensors[name]['data']
  max_time = sensors[name]['interval']/1000 # time in seconds
  start = sensors[name]['start']            # start time

  # get the time and save frequency sampling ----------------------
  dt = max_time/data.shape[0]             # time sampling
  t = np.arange(data.shape[0])*dt         # time vector
  sensors[name]['sampling'] = 1/dt        # frequency sampling Hz

  # remove gravity offset from vertical sensor and change units ---
  print(np.round(data.mean()))
  data_rem = data - np.round(data.mean()) # remove g offset
  data_rem = data_rem*g                   # change to m2/s

  # plot ---------------------------------------------------------
  fig, axs = plot_vib(data_rem, t, name=name, start=start)

#%% ===============================================================
# data in frequency domain
# =================================================================
for name in sensors:
  data = sensors[name]['data']
  start = sensors[name]['start']
  sampling = sensors[name]['sampling']
  # remove gravity offset from vertical sensor and change units ---
  data_rem = data - np.round(data.mean()) # remove g offset
  data_rem = data_rem*g                   # change to m2/s
  # apply hanning window to the vibration data -------------------
  win = windows.hann(data.shape[0], np.pi*0.5) 
  win = win[:, np.newaxis]                # hanning window
  X = np.array(data_rem)*win              # vibration matrix
  # apply the dft ------------------------------------------------
  f, Y = dft(X, sampling)                           #
  data_vib = pd.DataFrame(Y, columns=data.columns)
  sensors[name]['f'] = f
  sensors[name]['data_freq'] = data_vib
  # plot ---------------------------------------------------------
  fig, axs = plot_vib(data_vib, f, name=name, start=start,
    xlabel='frequency [Hz]',case='freq')

#%% ===============================================================
# find harmonics
# =================================================================
def find_peaks_harmonics(y, threshold, f):
  '''function to find the peaks index in frequency vector
    over the selected threshold and the corresponding 
    harmonics and mean frequency index
  '''
  peaks = find_peaks(y, threshold)[0]   # find the peaks over threshold
  harmonics = np.array([], dtype=int)   # store the harmonics
  main_peaks = np.array([], dtype=int)  # store mean frequencies 
  # after the middle of spectrum there is no possible harmonics
  peaks_eval = peaks[peaks<(len(f)//2 + 1)] 
  for peak in peaks_eval:
    if np.isin(peak, harmonics): 
      pass        # pass frequencies on the harmonics 
    elif sum(peaks%peak==0)>1:
      harm = peaks[peaks%peak==0]
      print('main freq = %.2f  [Hz] presents %s harmonics'%(
        f[peak], len(harm)
      ))
      harmonics = np.r_[harmonics, harm]
      main_peaks = np.r_[main_peaks, peak]
  return peaks, harmonics, main_peaks
# -------------------------------------------------------------------
for name in sensors:
  data = sensors[name]['data']
  start = sensors[name]['start']
  f = sensors[name]['f']
  data_vib = sensors[name]['data_freq']
  # plot ---------------------------------------------------------
  fig, axs = plot_vib(data_vib, f, name=name, start=start,
    xlabel='frequency [Hz]',case='freq')
  Y = np.array(data_vib)
  # finding peaks in spectrum ------------------------------------
  rms = RMS(Y)
  for i, ax in enumerate(axs):
    threshold = rms[i]
    ax.hlines(threshold, f.min(), f.max(), color='b', label='rms',
    linestyle='--', alpha=0.7)
    peaks, harmonics, main_peaks = find_peaks_harmonics(
      Y[:, i], threshold, f)
    print('sensor %s in direction %s presents %s main frequencies\
       with harmonics over rms value'%(
      name, data_vib.columns[i], len(main_peaks)
    ))
    ax.vlines(f[peaks], 0, Y.max()*1.05, color='gray', zorder=0, 
      label='peaks', linestyle='--', alpha=0.5)
    if len(harmonics)>0:
      ax.plot(f[harmonics], Y[harmonics, i], 'x', color='r', 
        label='harmonics', alpha=0.6)
    ax.legend()

# %%
