import numpy as np


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
