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
'''some equipments the conventional horizontal, vertical axial 
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
df = df.sort_values(by=['sensorId', 'time_start'])
df.index = range(len(df))

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
  for j, d in enumerate(directions):
    axs[j].plot(t, dfi['accel_RMS_' + d], color='g')
    ax2[j] = axs[j].twinx()
    ax2[j].plot(t, dfi['vel_RMS_'+ d], color='b', alpha=0.7)
    axs[j].set_ylabel(lab1[0] + '\n' + lab2[j], color='g')
    ax2[j].set_ylabel(lab1[1] + '\n' + lab2[j], color='b')
  axs[3].plot(t, dfi['temp'], color='gray', label='measure')
  if np.mean(dfi.temp_max)>0:
    axs[3].plot(t, dfi['temp_max'], label='limit', color='r')
  axs[3].set_ylabel('temperature')
  axs[3].legend()
  fig.tight_layout()
  
#%% ===============================================================
# labeling data 0 for downtime and 1 for uptime
# =================================================================
for j, sensor in enumerate(sensors):
  # define data of each sensor ------------------------------------
  dfi = df[df['sensorId']==sensor].copy()
  index = dfi.index
  title = '%s, %s'%(
    assets['name'][sensor], assets['modelType'][sensor])
  dtime = np.array(dfi.time_start[1:]
    ) - np.array(dfi.time_start[:-1])
  dfi['dt'] = np.r_[dtime, dtime[-1]]

  # labeling the data of corresponding sensor threshold  -----------
  th_values = {1: 1.2e-2, 2: 1.9e-2, 3: 1e-2, 9: 3e-2}
  th = th_values.get(j, 2.1e-2) # threshold

  vib_acel = np.array(dfi['accel_RMS'])
  dfi['class']           = 0*(vib_acel<th) + 1*(vib_acel>th)
  df.loc[index, 'class'] = 0*(vib_acel<th) + 1*(vib_acel>th)

  # plot the view of classification -------------------------------
  fig, axs = plt.subplots(4, 1, figsize=(6, 10))
  ax_0 = list(axs)
  ax_1 = [0, 0, 0]
  fig.suptitle(title)

  # axis 0 a 2 
  labels = ['accel_RMS', 'vel_RMS', 'dt']
  for i in range(3):
    ax_1[i] = ax_0[i].twinx()
    sns.scatterplot(dfi, alpha=0.4,
      x=labels[i], y='time_start', hue='class', ax =ax_0[i])
    sns.kdeplot(dfi[labels[i]], ax =ax_1[i], color='k')
    #if i<2:
    ax_1[i].set_xscale('log')
  ax_0[2].vlines(1e4,dfi['time_start'].min(), 
    dfi['time_start'].max(), color='r')
  
  # axis 3
  for Class, color  in enumerate(['r', 'k']):
    ax_0[3].plot(dfi.time_start[dfi['class']==Class], 
      dfi.accel_RMS[dfi['class']==Class],'--', color=color)
  fig.tight_layout()
  
#%% ===============================================================
# classification decision tree for downtime and uptime asset
# =================================================================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
# select features
columns = ['vel_RMS', 'accel_RMS', 'model_type','rpm']
X = np.array(df.loc[:, columns])
y = np.array(df.loc[:, 'class'])

# Split the data into training and testing sets -------------------
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.5, random_state=21)

# Create and train a Decision Tree Classifier ---------------------
clf = DecisionTreeClassifier(random_state=21)
clf.fit(X_train, y_train)

# Make predictions on the train and test data set -----------------
y_pred_tr = clf.predict(X_train)
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy -----------------------------------
accuracy_tr = accuracy_score(y_train, y_pred_tr)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy train:", accuracy_tr)
print("Accuracy test:", accuracy)
print("Number of leaf nodes:", clf.get_n_leaves())

# Export the Decision Tree to a text-based representation
tree_text = export_text(clf, feature_names=columns)
print(tree_text)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
lab = ['downtime', 'uptime']
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', 
  cbar=False, xticklabels=lab, yticklabels=lab)
plt.xlabel('Predicted');  plt.ylabel('Actual'); plt.show()


#%% ===============================================================
# function to predict downtime and uptime
# =================================================================
def calculate_dw_up(df, sensor):
  '''calculate the downtime and uptime in dataframe df for the 
  equipment measured by the sensor  
  '''
  # get delta times between vibration measurement 
  dfi = df[df['sensorId']==sensor].copy() # select sensor data
  dfi = dfi.sort_values(by='time_start').copy() # ensure order
  dtimes = np.array(dfi.time_start[1:]    # get delta times
    ) - np.array(dfi.time_start[:-1])
  last_dt = df['duration'].iloc[-1]       # add dtime last 
  dtimes = np.r_[dtimes,  last_dt]        #     measurement

  # predict downtime or uptime in each measurement 
  columns = ['vel_RMS', 'accel_RMS', 'model_type','rpm']
  class_pred = clf.predict(np.array(dfi.loc[:, columns]))

  # calculate the operation time, downtime, and uptime in seconds 
  total_time = dfi.time_start.max() - dfi.time_start.min() + last_dt
  uptime   = np.sum(dtimes*(class_pred==1))
  downtime = np.sum(dtimes*(class_pred==0))
  return downtime, uptime, total_time
# -----------------------------------------------------------------
# predict times for each machine
# -----------------------------------------------------------------
msgs = ['total operation time = ', 'downtime = ', 'uptime = ']
for sensor in sensors:
  # define data of each sensor 
  machine = assets['name'][sensor], 
  mtype = assets['modelType'][sensor]
  maxdw = assets['specifications.maxDowntime'][sensor]

  # calculate downtime and uptime
  downtime, uptime, total_time = calculate_dw_up(df, sensor)
  print((' %s '%(machine)).center(65, '█'))
  print('machine type : ', mtype)
  print('maximum downtime : ', maxdw)
  times = [total_time, downtime, uptime]
  for msg, time in zip(msgs, times):
    time_formated = str(datetime.timedelta(seconds=time))
    print(msg, time_formated)

#%% ===============================================================
# functions to identify changes in vibration patterns
# =================================================================
for sensor in sensors:
  #def changes_patterns(df, sensor):
  # get data
  dfi = df[df['sensorId']==sensor].copy()
  index = dfi.index

  # filter only for uptime measurements
  columns = ['vel_RMS', 'accel_RMS', 'model_type','rpm']
  class_pred = clf.predict(np.array(dfi.loc[:, columns]))
  dfi = dfi.iloc[class_pred==1, :]

  # signals to analyse 
  signals_a = ['accel_RMS_' + i for i in ['h', 'v', 'a']]
  signals_v = ['vel_RMS_' + i for i in ['h', 'v', 'a']] 
  signals = signals_a + signals_v

  # add the moving average and standard deviation 
  for signal in signals:
    dfi.loc[:, signal + '_mean'] = dfi[signal].rolling(
      window=20, min_periods=0).mean()
    dfi.loc[:, signal + '_std'] = dfi[signal].rolling(
      window=20, min_periods=0).std()

  # add pearson correlation between acceleration and velocity 
  for i in ['h', 'v', 'a']:
    dfi.loc[:, 'cor_accel_vel_' + i] = np.abs(
      dfi['accel_RMS_' + i].rolling(
        window=20, min_periods=1).corr(dfi['vel_RMS_'+ i]))

  # identify soft and hard changes in behavior
  dfi.loc[:, ['soft', 'hard', 'outliers']] = 0
  for signal in signals:
    mean = dfi.loc[:, signal + '_mean']
    std = dfi.loc[:, signal + '_std']
    x = dfi.loc[:, signal] 
    dfi.loc[:, 'soft'] += 1*(x > (mean + 2*std))  # soft changes in behavior
    dfi.loc[:, 'hard'] += 1*(x > (mean + 3*std))  # hard changes in behavior

  # identify outliers in transformer, motors and centrifugal pumps
  if np.isin(dfi['model_type'].mean(), [1, 4, 5]):
    for i in ['h', 'v', 'a']:   
      if dfi['model_type'].mean()==4 and dfi['power'].mean()>0:
        pass
      else:
        dfi.loc[:, 'outliers'] += 1*(dfi.loc[:,'cor_accel_vel_' + i]  <0.9)

  plt.plot(dfi['outliers']); plt.show()

#%% ===============================================================
# add moving to predict changes in vibration patterns
# =================================================================
# add mean average for each vibration signal:
signals_a = ['accel_RMS_' + i for i in ['h', 'v', 'a']]
signals_v = ['vel_RMS_' + i for i in ['h', 'v', 'a']]
signals = signals_a + signals_v
signals
df_an = df[df['class']==1].copy()
#df_an = df.copy()
for j, sensor in enumerate(sensors):
  # define data of each sensor ------------------------------------
  dfi = df_an[df_an['sensorId']==sensor].copy()
  index = dfi.index
  title = '%s, %s, %s rpm, %s kw'%(
    assets['name'][sensor], assets['modelType'][sensor],
    assets['specifications.rpm'][sensor],
    assets['specifications.power'][sensor])

  # ----
  for signal in signals:
    df_an.loc[index, signal + '_mean'] = dfi[signal].rolling(
      window=20, min_periods=1).mean()
    df_an.loc[index, signal + '_std'] = dfi[signal].rolling(
      window=20, min_periods=1).std()
    df_an.loc[index, signal + '_min'] = dfi[signal].rolling(
      window=20, min_periods=1).min()
    df_an.loc[index, signal + '_max'] = dfi[signal].rolling(
      window=20, min_periods=1).max()

  for i in ['h', 'v', 'a']:
    df_an.loc[index, 'cov_accel_vel_' + i] = dfi['accel_RMS_' + i
      ].rolling(window=20, min_periods=1).cov(dfi['vel_RMS_' + i])
    df_an.loc[index, 'cor_accel_vel_' + i] = np.abs(dfi['accel_RMS_' + i
    ].rolling(window=20, min_periods=1).corr(dfi['vel_RMS_'+ i]))



  #plt.plot(df.loc[index, 'accel_RMS_h'])
  fig, ax = plt.subplots(3, 1)
  fig.suptitle(title)
  x = np.arange(0, len(dfi)) 
  var =      np.array(df_an.loc[index, 'vel_RMS_a'])
  var2 =     np.array(df_an.loc[index, 'accel_RMS_a'])
  var_mean = np.array(df_an.loc[index, 'vel_RMS_a_mean'])
  var_std =  np.array(df_an.loc[index, 'vel_RMS_a_std'])
  var_min =  np.array(df_an.loc[index, 'vel_RMS_a_min'])
  var_max =  np.array(df_an.loc[index, 'vel_RMS_a_max'])
  var_corr = np.array(df_an.loc[index, 'cor_accel_vel_a'])
  var_cov =  np.array(df_an.loc[index, 'cov_accel_vel_a'])
  z_score = (var - var_min)/(var_max - var_min)
  #ax[0].plot(df_an.loc[index, 'accel_RMS_h'],df_an.loc[index, 'vel_RMS_h'], '.')
  ax[0].plot(x, var, '.-')
  ax[0].plot(x, var_mean)
  sel = var_corr<0.9
  ax[0].plot(x, var_mean + 3*var_std)
  ax[0].plot(x[sel], var[sel] , 'x', color='r' )
  #ax[0].set_yscale('log')
  #ax[1].set_yscale('log')
  #ax[1].plot(var_cov, '.')
  #ax[1].hlines([2e-6], index.min(), index.max(), color='r')
  #ax[1].plot(x, np.gradient(var), '.')
  ax[1].plot(var/var.max(), var2/var2.max(), '.')
  ax[1].plot(var[sel]/var.max(), var2[sel]/var2.max(), 'x', color='r')
  ax[2].plot(var_corr)
  ax[2].hlines([0.9], 0, x.max(), color='r', linestyle='--')
  #ax[1].plot(var_std*var_std, '.')
  plt.show()


# %%
