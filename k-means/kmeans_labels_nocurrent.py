#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:55:19 2020

@author: stephey
"""

#try maxim's summary files in kmeans

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(5)


#I processed quench #3 and #103 for now, "bot" acoustic channel only.
#Signals there thresholded at 60 mV to extract individual events. For
#each individual event a spectrogram was calculated and recorded as 2D
#array in a comma-separated csv file (where filename is the event
#number). The length of each waveform is ~5 ms (5000 points). For events
#that were shorter than 5000 pts rest of the waveform was "padded" with
#small random numbers before spectrogram calculation to avoid processing
#artifacts. Events that were longer than 5000 points were truncated to
#5000 points. A pre-trigger window in front of all events is 100 points.
#Spectrogram arrays are 512 rows (frequencies) x 625 columns (duration).
#The physical frequency range corresponding to the 512 levels is 0 -
#485000 Hz. The physical duration corresponding to the 625 points is 5.15 ms.

#In addition to spectrograms, a file named "summary.csv" was recorded for
#both quench ramps containing all events found in that particular ramp.
#Each event is represented with a row of 6 comma-separated numbers, and
#the column labels are the following:
#
## of zero crossings,  Duration of the event (pts),  Square of max.
#amplitude (Umax^2), Energy (Urms^2), Mean frequency (Hz), Magnet current
#(A), Absolute event starting point (#)

#col 0 - num of zero crossings
#col 1 - Duration of the event (pts)
#col 2 - Square of max amplitude (Umax^2)
#col 3 - Energy (Urms^2)
#col 4 - Mean frequency (Hz)
#col 5 - Magnet current (A)
#col 6 - Absolute event starting point (#)

#For quench ramp #3 magnet current was saturated at ~900A, but events
#were collected in the window where current is >500A, so post-quench
#events are not included.

#indicies are the same


df003 = pd.read_csv("/Users/stephey/Dropbox/NERSC/Work/Dates/20200409/summary_q003.csv",
                    names=["Num Zero Crossings", "Duration (pts)", "Umax^2",
                           "Urms^2", "Mean Freq (Hz)", "Mag Current (A)", "Abs Start Point"])

df103 = pd.read_csv("/Users/stephey/Dropbox/NERSC/Work/Dates/20200409/summary_q103.csv",
                    names=["Num Zero Crossings", "Duration (pts)", "Umax^2",
                           "Urms^2", "Mean Freq (Hz)", "Mag Current (A)", "Abs Start Point"])

#maxim would like to exclude amplitude and current (anything that correlates to ramp time)

#Would it be possible to see if clustering occurs when only quantities not 
#obviously dependent on the magnet current are considered? Such as mean 
#frequency, or ratio of (N zero crossings)/(U maX^2), or maybe  of 
#(N zero crossings)/ (duration)?

#add some other columns that maxim suggests
#won't add zero crossings/duration since we already have both of those orig columns

ratio_umax_003 = df003['Num Zero Crossings']/df003['Umax^2']
ratio_umax_103 = df103['Num Zero Crossings']/df103['Umax^2']

df003['Ratio Umax'] = ratio_umax_003
df103['Ratio Umax'] = ratio_umax_103

#delete other columns maxim suggests
del df003['Umax^2']
del df003['Urms^2']
del df003['Mag Current (A)']
del df003['Abs Start Point']

del df103['Umax^2']
del df103['Urms^2']
del df103['Mag Current (A)']
del df103['Abs Start Point']

#get the mean and std for each quench dataset
q003_mean = df003.mean()
q003_std = df003.std()

q103_mean = df103.mean()
q103_std = df103.std()

#subtract the mean and then divide by the standard deviation
q003_norm = (df003 - q003_mean)/q003_std 
q103_norm = (df103 - q103_mean)/q103_std 

#merge into one large dataframe
norm_all = pd.concat([q003_norm, q103_norm],axis=0)
#4 columns in this dataset
pca = PCA(n_components=4)
pca_data = pca.fit(norm_all)
print(pca.explained_variance_ratio_)
#[n_components, n_features]
#pc1 is first row, pc2 is second row,...
comps = abs(pca.components_)
print(comps)


reduced_data = PCA(n_components=3).fit_transform(norm_all)

#now put pca data into kmeans clustering algorithm
#with pca reduced data
X = reduced_data

estimators = [('2_clusters', KMeans(n_clusters=2))]

#with pca reduced data
X = reduced_data

fignum = 1
titles = ['2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('PCA component 0')
    ax.set_ylabel('PCA component 1')
    ax.set_zlabel('PCA component 2')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1


##also plot in 2d
#for name, est in estimators:
#    est.fit(X)
#    
#    fig, (ax_l, ax_c, ax_r) = plt.subplots(nrows=1, ncols=3,
#                                       sharex=True, figsize=(12, 3.5))
#
#    labels = est.labels_
#    ax_l.scatter(X[:, 0], X[:, 1],
#               c=labels.astype(np.float), edgecolor='k')
#    ax_l.set_title(name + " 0,1 PCA components")
#    ax_c.scatter(X[:, 1], X[:, 2],
#               c=labels.astype(np.float), edgecolor='k')
#    ax_c.set_title(name + " 1,2 PCA components")
#    ax_r.scatter(X[:, 0], X[:, 2],
#               c=labels.astype(np.float), edgecolor='k')
#    ax_r.set_title(name + " 0,2 PCA components")
#    
#    fig.suptitle(name)
#    
#    plt.show()
    
#labels contains the info we want-- sorts the events into 2 clusters
#let's use the 3d pca analysis so we don't throw out important information

#break into labels for each quench
q003_labels = labels[0:3151]
q103_labels = labels[3151::]
#    
#fig = plt.figure()
#plt.plot(df003['Abs Start Point'],q003_labels,'.')    
#    
#fig = plt.figure()
#plt.plot(df103['Abs Start Point'],q103_labels,'.')    

#write label data to file so we can use later

np.save('q003_kmeans_labels_nocurrent.npy',q003_labels)
np.save('q103_kmeans_labels_nocurrent.npy',q103_labels)    

#also save to human readable csv files

np.savetxt("q003_kmeans_labels_nocurrent.csv", q003_labels, delimiter="\n")   
np.savetxt("q103_kmeans_labels_nocurrent.csv", q103_labels, delimiter="\n") 





