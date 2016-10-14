# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:34:06 2016

@author: acbart
"""

# Great library for mathy stuff
import numpy as np
# Great library for making graphs
import matplotlib.pyplot as plt
# Great library for processing data
import pandas as pd

# SkLearn has a number of clustering algorithms
from sklearn.cluster import KMeans, AffinityPropagation, Birch, SpectralClustering

# We will use a Data Standardizer from the preprocessing module
from sklearn import preprocessing

# For dev purposes, we'll use the same clustering randomness each time
np.random.seed(42)

# Read in the CSV dataset
instructors = pd.read_csv('vt-cs-grades.csv', sep='\t')

# First column (named "Name") becomes the DataFrame index
instructors = instructors.set_index('Name')

# Preprocessing: 
# Standardize all the columns so that we're working on the same scales
for column in instructors.columns:
    scaler = preprocessing.StandardScaler()
    instructors[column] = scaler.fit_transform(instructors[column])

# Let's focus on two particular columns. This algorithm generalizes just fine
#   to higher dimensional feature sets, though.
interested_columns = ["GPA", "Upper Levels"]
data = instructors[interested_columns]

# Useful list of colors for our eventual clusters
color_index = ['r', 'g', 'b', 'y', 'm', 'c']

# Set the K of our algorithm, or number of clusters
CLUSTERS = 5

# Build up the KMeans obect
clusterer = KMeans(n_clusters=CLUSTERS, n_init=10)
#clusterer = AffinityPropagation()
#clusterer = Birch(n_clusters=CLUSTERS)
#clusterer = SpectralClustering(n_clusters=CLUSTERS)

# Perform the clustering algorithm, only on the relevant columns
fitted_means = clusterer.fit(data)

# Snippet to convert numeric cluster labels to colors
colors = [color_index[l % len(color_index)] for l in fitted_means.labels_]
                      
# Make a scatter plot of the clustered data, with colors!
ax = instructors.plot(kind='scatter', 
                      x=interested_columns[0],
                      y=interested_columns[1],
                      c=colors)

# Put the resulting clusters back into the DataFrame
instructors['Group'] = fitted_means.labels_

# And we can put the color back in there too. Why not.
instructors['Color'] = colors

# Sort the instructors by their cluster, for reporting purposes
instructors = instructors.sort_values(['Group'])

# Show each color's average value for each feature
print(instructors.groupby('Color').mean().sort_values(['GPA']))

# Show each color's frequency in the dataset
print(instructors.groupby('Color').count()['Group'])

# Print all the instructors and their groups
print(instructors['Color'])