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
import seaborn as sns

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
def find_peaks_harmonics(y, threshold, f):
  '''function to find the peaks over the selected threshold
  and find the harmonics in this peaks
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

#%% ===============================================================
# import data
# =================================================================
collects_path = os.sep.join(['data', 'part2', 'collects.csv'])
assets_path = os.sep.join(['data', 'part2', 'assets.csv'])
collects = pd.read_csv(collects_path)
assets = pd.read_csv(assets_path)
assets.index = [assets['sensors'][i][2:-2] for i in assets.index]


#%% ===============================================================
# data preprocessing
# =================================================================
# removing nan value ----------------------------------------------
collects = collects[~collects['params.accelRMS.x'].isna()]
collects.index = range(len(collects))
sensors = np.unique(collects['sensorId'])
# ------------------------------------------------------------------
'''some equipemnts the conventional horizontal, vertical axial 
directions not apply. However convenience for analysis will 
standardized.
 '''
sel_col = ['specifications.axisX', 'specifications.axisY',
  'specifications.axisZ']
assets.loc[:,sel_col] = assets.loc[:,sel_col].fillna('other')
assets = assets.fillna(0)
assets.loc[['MUR8453', 'MZU6388'],  'specifications.axisZ'] = 'axial'
assets.loc['MYS2071', 'specifications.axisY'] = 'vertical'
assets.loc['NAH4736', sel_col] = ['horizontal', 'vertical', 'axial']
assets.loc['NEW4797', sel_col] = ['horizontal', 'vertical', 'axial']
model_names =  np.unique(assets['modelType'])
assets.loc[:,sel_col]

#%% ===============================================================
# Create a dataframe for analyses
# =================================================================
df = pd.DataFrame()
df['duration'] = collects['params.duration']
df['sampling_rate'] = collects['params.sampRate']
df['time_start'] = collects['params.timeStart']
df['sensorId'] = collects['sensorId']
df['temp'] = collects['temp']
for sensor in sensors:
  dfi =  collects[collects['sensorId']==sensor]
  i = dfi.index
  df.loc[i, 'temp_max'] = assets.loc[sensor, 'specifications.maxTemp' ]
  df.loc[i, 'Downtime_max'] = assets.loc[sensor, 'specifications.maxDowntime' ]
  df.loc[i, 'power'] = assets.loc[sensor, 'specifications.power' ]
  df.loc[i, 'rpm'] = assets.loc[sensor, 'specifications.rpm' ]

  model_name = assets.loc[sensor, 'modelType' ]
  df.loc[i, 'model_type'] = np.where(model_name==model_names)[0][0]
  specs = ['X', 'Y', 'Z']
  directions = [assets.loc[sensor, ('specifications.axis' + j)][0] 
    for j in specs]
  for d, s in zip(directions, specs): 
    df.loc[i, ('accel_RMS_' + d)] = dfi['params.accelRMS.' + s.lower()]
    df.loc[i, ('vel_RMS_' + d)] = dfi['params.velRMS.' + s.lower()] 
df['vel_RMS'] = df.loc[:, 
  ['vel_RMS_' + d for d in directions]].sum(axis=1)/3
df['accel_RMS'] = df.loc[:,
  ['accel_RMS_' + d for d in directions]].sum(axis=1)/3

#%% ===============================================================
# Present data
# =================================================================
for sensor in sensors:
  dfi = df[df['sensorId']==sensor]
  t =np.array(dfi['time_start'])
  lab1 = [ 'acceleration', 'velocity']
  lab2 = [ 'horizontal', 'vertical', 'axial']
  fig, axs = plt.subplots(4, 1, sharex=True)
  ax2 = [0, 0, 0]
  fig.suptitle(assets['name'][sensor])
  #for i, kind in enumerate(['accel_RMS_', 'vel_RMS_']):
  for j, d in enumerate(directions):
    axs[j].plot(t, dfi['accel_RMS_' + d], color='g')
    ax2[j] = axs[j].twinx()
    ax2[j].plot(t, dfi['vel_RMS_'+ d], color='b', alpha=0.7)
    axs[j].set_ylabel(lab1[0] + '\n' + lab2[j], color='g')
    ax2[j].set_ylabel(lab1[1] + '\n' + lab2[j], color='b')
  axs[3].plot(t, dfi['temp'], color='gray', label='measure')
  if np.mean(dfi.temp_max)>0:
    axs[3].plot(t, dfi['temp_max'], label='limite', color='r')
  axs[3].set_ylabel('temperature')
  axs[3].legend()
  fig.tight_layout()
  
#%% ===============================================================
# labeling data
# =================================================================
df['class'] = 1
# =================================================================
for j, sensor in enumerate(sensors):
  fig, ax = plt.subplots(3, 1)
  dfi = df[df['sensorId']==sensor]
  index = dfi.index
  fig.suptitle(('%s, %s'%(
    assets['name'][sensor], assets['modelType'][sensor]
  )))
  # axis 0
  ax0 = ax[0].twinx()
  th = 2.1e-2 if j!=3 else 1e-2
  th = th if j!=9 else 3e-2
  th = th if j!=1 else 1.2e-2
  th = th if j!=2 else 1.9e-2
  dfi['class'] = 0*(dfi['accel_RMS']<th) + 1*(dfi['accel_RMS']>th)
  df.loc[index, 'class'] = np.array(dfi['class'])
  sns.scatterplot(dfi, x='accel_RMS', y='time_start', hue='class', ax =ax0)
  sns.kdeplot(dfi['accel_RMS'], ax =ax[0])
  ax[0].set_xscale('log')
  
  # axis 1
  ax1 = ax[1].twinx()
  
  sns.scatterplot(dfi, x='vel_RMS', y='time_start', hue='class', ax =ax1)
  ax[1].set_xscale('log')
  
  # axis 2
  ax2 = ax[2].twinx()
  dtime = np.array(dfi.time_start[1:]) - np.array(dfi.time_start[:-1])
  sns.kdeplot(dtime, ax =ax[2])
  sns.scatterplot(x=dtime,y=dfi.time_start[:-1])
  ax2.vlines(1e4,dfi['time_start'].min(), 
    dfi['time_start'].max(), color='r')


  fig, ax = plt.subplots()
  ax.plot(dfi.time_start[dfi['class']==0], 
    dfi.accel_RMS[dfi['class']==0],'.', color='r')
  ax.plot(dfi.time_start[dfi['class']==1], 
    dfi.accel_RMS[dfi['class']==1], color='k')
  fig.tight_layout()
  


#%% ===============================================================
# classification decision tree
# =================================================================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
columns = [
  'vel_RMS', 'vel_RMS_h', 'vel_RMS_v', 'vel_RMS_a',
  'accel_RMS', 'accel_RMS_h', 'accel_RMS_v', 'accel_RMS_a', 
  'model_type',
  'rpm', 'power'
]
X = np.array(df.loc[:, columns])
y = np.array(df.loc[:, 'class'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=21)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=21,
  max_leaf_nodes=4)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export the Decision Tree to a text-based representation
tree_text = export_text(clf, feature_names=columns)

# Print the Decision Tree structure on the screen
print(tree_text)
#%% ===============================================================
# add some statistics for next analisys
# =================================================================
windows_size
df['mean'] = df['accel_RMS_vertical'].rolling(window=20, min_periods=1).mean()
df['var'] = df['accel_RMS_vertical'].rolling(window=20, min_periods=1).var()
df['accel_RMS_vertical'].rolling(window=20, min_periods=1).cov(df['accel_RMS_vertical'])
df['accel_RMS_vertical'].rolling(window=20).corr(df['accel_RMS_vertical'])


#%%
for column in collects.columns:
  l = len(np.unique(collects.loc[:,column])) 
  print(column, l)
print('assets ===============================================')
for column in assets.columns[1:]:
  try:
    l = len(np.unique(assets.loc[:,column]) )
    print(column, l)
  except:
    print(column, 'fail')


  #print(assets[column])
#%% ===============================================================
# first view
# =================================================================
sensors = np.unique(collects['sensorId'])
for sensor in sensors:
  df = collects[collects['sensorId']==sensor]
  #df = collects
  t = np.array(df['params.timeStart'])
  x = np.array(df['params.accelRMS.x'])
  y = np.array(df['params.accelRMS.y'])
  z = np.array(df['params.accelRMS.z'])
  acel = x + y + z
  xvel = np.array(df['params.velRMS.x'])
  yvel = np.array(df['params.velRMS.y'])
  zvel = np.array(df['params.velRMS.z'])
  vel = xvel + yvel + zvel
  temp = np.array(df['temp'])
  i = np.argsort(t)
  t = (t - np.nanmin(t))
  plt.title((assets['name'][sensor], 
    assets['specifications.maxDowntime'][sensor],
    assets['modelType'][sensor],
    assets['specifications.rpm'][sensor]
    ))
  #plt.plot(t[i], acel[i]/3, '.')
  #plt.hlines(1.3e-2, np.nanmin(t), np.nanmax(t), color='r')
  j = (acel[i]/3)>(1.3e-2)
  #plt.plot(t[i], temp[i], '-')
  plt.plot(t[i][j], x[i][j], '.')
  plt.plot(t[i][j], y[i][j], '.')
  plt.plot(t[i][j], z[i][j], '.')
  #plt.plot(t[i][j], yvel[i][j], '.')
  #plt.plot(t[i][j], zvel[i][j], '.')
  #plt.loglog(t[i][j], z[i][j], '.')

  #plt.hlines(2e-3, np.nanmin(t), np.nanmax(t), color='r')
  plt.yscale('log')
  #plt.plot(sorted(t), '.')
  plt.show()






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
  data_rem = data - np.round(data.mean()) # remove g offset
  data_rem = data_rem*g                   # change to m2/s

  # plot ---------------------------------------------------------
  fig, axs = plot_vib(data_rem, t, name=name, start=start)
