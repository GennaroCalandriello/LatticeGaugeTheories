from math import isnan
import numpy as np
import os
import statistics
from numba import njit, float64, int64
from pandas import isna


def bootstrap(N, N_boot, source_collection):

    bootstrap_collection = np.zeros((N_boot, N), np.float)
    bootstrap_means = np.zeros((N_boot), np.float)

    for k in range(N_boot):
        for i in range(0, N):
            bootstrap_collection[k, i] = source_collection[int(np.random.uniform(0, N))]
        bootstrap_means[k] = np.means(bootstrap_collection[k])

    boot_estimator = np.mean(bootstrap_means)
    boot_std = np.std(bootstrap_means)

    return boot_estimator, boot_std


def JackKnife(N, original_sample, biased_estimator):
    jack_var = 0.0
    bias = 0.0

    for i in range(N):
        partial_mean = np.mean(np.delete(original_sample, i))
        jack_var += (partial_mean - biased_estimator) ** 2
        bias += partial_mean

    jack_var *= (N - 1) / N
    jack_std = np.sqrt(jack_var)

    bias = bias / N
    unbiased_estimator = biased_estimator - (N - 1) * (bias - biased_estimator)

    return unbiased_estimator, jack_std


def bin_data(data):
    data = data[:]
    variance = []
    bins = []
    power = int(np.log2(len(data) + 1))

    for i in range(power):
        new_data = []
        N = len(data)
        variance.append((np.std(data) ** 2) / N)
        while data != []:
            x = data.pop()
            y = data.pop()
            new_data.append((x + y) / 2)
        bins.append(i)
        data = new_data[:]

    return np.power(2, bins), variance


def bin_lepage(data):
    def bin_var(G, binsize):
        G_binned = []
        for i in range(0, len(G), binsize):
            G_avg = 0
            for j in range(0, binsize):
                G_avg += G[i + j]
            G_binned.append(G_avg / binsize)

        return G_binned

    bins = []
    var = []
    for binsize in range(1, int(len(data) / 2) + 1):
        if len(data) % binsize == 0:
            binned_data = bin_var(data, binsize)
            bins.append(binsize)
            var.append((np.std(binned_data) ** 2) / len(binned_data))

    return bins, var

@njit(float64[:](float64[:],float64), fastmath=True)
def ricampionamento(array_osservabile, bin):
    sample=[]
    for _ in range(round(len(array_osservabile)/bin)):
        ii=np.random.randint(0, len(array_osservabile))
        sample.extend(array_osservabile[ii:min(ii+bin, len(array_osservabile))]) 

    return np.array(sample)

def bootstrap_(array_osservabile, bin):
    mean_array=[]
    
    for _ in range(100):

        sample=ricampionamento(array_osservabile, bin)
        mean_array.append(np.mean(sample))
    
    sigma=statistics.stdev(mean_array) 
    return sigma

if __name__=='__main__':

    bin_array=[i+5 for i in range(0, 1000, 5)]
    print(bin_array)
    path = 'njit/data/polyakov/err'
     
    sigma_array=[]

    for files in os.listdir(path):
        sigma_temp=[]
        patata=os.path.join(path, files)
        L = np.loadtxt(patata, dtype=float)
        for bin in bin_array:
            sigma=bootstrap_(L, bin)
            sigma_temp.append(sigma)
        sigma_array.append(max(sigma_temp))
        
    np.savetxt(f'{path}/sigmaPoly.txt', sigma_array)