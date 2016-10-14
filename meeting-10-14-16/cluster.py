# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:34:06 2016

@author: acbart
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.cluster import KMeans, AffinityPropagation, Birch, SpectralClustering

np.random.seed(42)

instructors = pd.read_csv('simplified-filtered.csv', sep='\t')
instructors = instructors.set_index('Name')

# Logarithm # of courses
instructors['Courses Taught'] = instructors['Courses Taught'].apply(math.log)

# Clustering
columns = ['As', 'Bs', 'Cs', 'Ds', 'Fs', 'Courses Taught']
interested_columns = ["Courses Taught", "Upper Levels"]
#interested_columns=interested_columns
color_index = ['r', 'g', 'b', 'y', 'm', 'c']

CLUSTERS = 3
k_means = KMeans(n_clusters=CLUSTERS, n_init=10)
#k_means = AffinityPropagation()
#k_means = Birch(n_clusters=CLUSTERS)
#k_means = SpectralClustering(n_clusters=CLUSTERS)
fitted_means = k_means.fit(instructors[interested_columns])
colors = [color_index[l % len(color_index)] for l in fitted_means.labels_]
ax = instructors.plot(kind='scatter', 
                      x=interested_columns[0],
                      y=interested_columns[1],
                      c=colors)
#bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
instructors['Group'] = fitted_means.labels_
instructors = instructors.sort_values(['Group'])
instructors['Color'] = instructors['Group'].map(lambda x: color_index[x% len(color_index)])

# Show each color's average values
print(instructors.groupby('Color').mean().sort_values(['GPA']))
print(instructors.groupby('Color').count()['Group'])
print(instructors)