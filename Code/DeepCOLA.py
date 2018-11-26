import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from sklearn.cluster import MeanShift
import pylab as pl
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, euclidean 
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # to smooth the plot
import timeit

class DeepCOLA:
    
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim
        
        self.data = self.readData()
        self.x = StandardScaler().fit_transform(self.data)

    def readData(self):
        datasetFile = '../Dataset/UK-Dale/Home1_1Day.csv'
        data = np.genfromtxt(datasetFile, delimiter=",", skip_header=1)
        data = np.where(np.isnan(data), 0, data) 
        
        return data
    
    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(20, activation='relu')(inputs)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(20)(inputs)
        decoded = Dense(43)(decoded)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        
        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='adadelta', loss='mse')
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tbCallBack])

    def DaviesBouldin(self, X, labels):
        n_cluster = len(np.bincount(labels))
        cluster_k = [X[labels == k] for k in range(n_cluster)]
        centroids = [np.mean(k, axis = 0) for k in cluster_k]
        variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
        db = []
        
        centroids.remove(centroids[0])
        
        variances.remove(variances[0])
        
        for i in range(n_cluster-1):
            for j in range(n_cluster-1):
                if j != i:
                    db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))
        
        return  (np.max(db) / n_cluster)    
    

    def findDunn(self, x, y):
  
        dunk = self.dunn(y, euclidean_distances(x))
            
        return dunk
    
    def dunn(self, labels, distances):
    
        labels = self.normalize_to_smallest_integers(labels)
    
        unique_cluster_distances = np.unique(self.min_cluster_distances(labels, distances))
        max_diameter = max(self.diameter(labels, distances))
    
        if np.size(unique_cluster_distances) > 1:
            return unique_cluster_distances[1] / max_diameter
        else:
            return unique_cluster_distances[0] / max_diameter
        
    def normalize_to_smallest_integers(self, labels):

        max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
        sorted_labels = np.sort(np.unique(labels))
        unique_labels = range(max_v)
        new_c = np.zeros(len(labels), dtype=np.int32)
    
        for i, clust in enumerate(sorted_labels):
            new_c[labels == clust] = unique_labels[i]
    
        return new_c    
    
    def min_cluster_distances(self, labels, distances):
    
        labels = self.normalize_to_smallest_integers(labels)
        n_unique_labels = len(np.unique(labels))
    
        min_distances = np.zeros((n_unique_labels, n_unique_labels))
        for i in np.arange(0, len(labels) - 1):
            for ii in np.arange(i + 1, len(labels)):
                if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                    min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
        return min_distances

    def diameter(self, labels, distances):
     
        labels = self.normalize_to_smallest_integers(labels)
        n_clusters = len(np.unique(labels))
        diameters = np.zeros(n_clusters)
    
        for i in np.arange(0, len(labels) - 1):
            for ii in np.arange(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
        return diameters
    
    def evaluateClusters(self, x, y):
        
        dunni = self.findDunn(x, y)
      
        sili = silhouette_score(x, y, metric='euclidean', sample_size=None, random_state=None)
        
        cali = metrics.calinski_harabaz_score(x, y)
        
        print("dunni ", dunni)
        print("sili  ", sili)
        print("cali  ", cali)

if __name__ == '__main__':
 
    ''' Clustering the appliances consumption (Cell 1)'''
    
    # %% Cell 1
    
    print("Part 1: Encoding the input data")
    
    ''' Initialize seed '''

    np.random.seed(2)
    set_random_seed(2)
    
    ''' Create object of DeepCOLA and initialize '''
    
    dc = DeepCOLA(encoding_dim=5)
    dc.encoder_decoder()
    
    ''' Start the timer '''
    start = timeit.default_timer()
    
    ''' Initialize batch size and epochs '''
    dc.fit(batch_size=10, epochs=50) 
    
    encoder = dc.encoder 
    
    inputs = dc.x
    
    encoded = encoder.predict(inputs) 
    
    print("\n\n")
    print("Part 2: Finding the clusters")
    
    ''' Apply mean shift to find clusters '''
    ms = MeanShift()
    ms.fit(encoded)
    
    ''' Stop the timer '''
    stop = timeit.default_timer()
    
    print('Deep COLA: Finds clusters         '+str(len(set(ms.labels_))))
    print("Deep COLA: Time taken (seconds) {}".format(stop - start))
    
    
    print("\n\n")
    print("Part 3: Evaluating the clusters")
    
    ''' Evaluate the clusters '''
    x1 = inputs
    y1 = ms.labels_
    dc.evaluateClusters(x1, y1.astype(int))
    
    df = pd.DataFrame(dc.data)
    df["c"] = y1
    
    
    print("\n\n")
    print("Part 4: Visualing the clusters")
    ''' Visualizing the clusters '''
    
    # %% Cell 2
    df0 = df.copy()
    df0[df0["c"]!=0] = 0
    yhat = savgol_filter(df0.iloc[:,42], 2201, 3)
    plt.plot(yhat, color='k', marker='o', markersize=4)
    plt.ylim(-50, 1500) 
    plt.show()
    
    df1 = df.copy()
    df1[df1["c"]!=1] = 0
    yhat = savgol_filter(df1.iloc[:,42], 201, 3)
    plt.plot(yhat, color='k', marker='o', markersize=4)
    plt.ylim(-50, 1500) 
    plt.show()
    
    df2 = df.copy()
    df2[df2["c"]!=2] = 0
    yhat = savgol_filter(df2.iloc[:,42], 501, 3)
    plt.plot(yhat, color='k', marker='o', markersize=4)
    plt.ylim(-50, 1500) 
    plt.show()
    
    df3 = df.copy()
    df3[df3["c"]!=3] = 0
    yhat = savgol_filter(df3.iloc[:,42], 201, 3)
    plt.plot(yhat, color='k', marker='o', markersize=4)
    plt.ylim(-50, 1500) 
    plt.show()
    
    df4 = df.copy()
    df4[df4["c"]!=4] = 0
    yhat = savgol_filter(df4.iloc[:,42], 801, 3)
    plt.plot(yhat, color='k', marker='o', markersize=4)
    plt.ylim(-50, 1500) 
    plt.show()
    