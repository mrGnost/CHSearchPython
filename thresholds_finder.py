import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from math import sqrt
import warnings


warnings.filterwarnings('ignore', category=UserWarning, append=True)


def get_data(path):
    my_map = fits.open(path)
    image_data = my_map[0].data
    return image_data


def get_values(data):
    arr = []
    for row in data:
        for elem in row:
            arr.append(elem)
    return arr


def get_hist_patches(arr):
    n, bins, patches = plt.hist(x=arr, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.close()
    return bins, np.array([patch.get_height() for patch in patches])


def moving_average(numbers, window_size=3):
    i = 0
    moving_averages = []
    for j in range(window_size - 1):
        this_window = numbers[0: window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def calculate_threshold(data):
    start_ind = np.argmax(data)
    ind = start_ind
    while data[ind] >= data[ind - 1] and ind > start_ind / 5:
        ind -= 1
    if ind <= start_ind / 5:
        return None
    return ind


def get_data_and_threshold(source_dir, dirname, filename):
    data = get_data(source_dir + "/brightness_maps/" + dirname + "/" + filename)
    patches = get_hist_patches(get_values(data))
    return patches, calculate_threshold(moving_average(patches[1]))


def dispersion(data, average):
    return sum((data - average) ** 2) / (len(data) - 1)


def find_statistical_error(data, average):
    return sqrt(dispersion(data, average) / len(data))


def get_thresholds_and_stat_error(source_dir, dirname, filenames):
    length = len(filenames)
    threshold_indices = [0] * length
    thresholds = [0] * length
    patches = [None] * length
    for i in range(length):
        local_patches, local_threshold = get_data_and_threshold(source_dir, dirname, filenames[i])
        patches[i] = local_patches
        threshold_indices[i] = local_threshold
    threshold_average = 0
    threshold_count = 0
    for i in range(length):
        if threshold_indices[i] is not None:
            threshold_average += patches[i][0][threshold_indices[i]]
            threshold_count += 1
    threshold_average /= threshold_count
    if all(index is None for index in threshold_indices):
        return [80] * length, None
    statistical_error = find_statistical_error(
        np.array([patches[i][0][threshold_indices[i]] for i in range(length) if threshold_indices[i] is not None]),
        threshold_average
    )
    for i in range(length):
        if threshold_indices[i] is None:
            thresholds[i] = threshold_average
        else:
            thresholds[i] = patches[i][0][threshold_indices[i]]
    return thresholds, statistical_error


def draw_distribution(arr, source_dir, dirname, result_name):
    n, bins, patches = plt.hist(x=arr, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Brightness distribution')
    max_frequency = n.max()
    plt.ylim(ymax=np.ceil(max_frequency / 10) * 10 if max_frequency % 10 else max_frequency + 10)
    plt.xlim(xmin=0, xmax=400)
    if not os.path.exists(source_dir + "/distributions/" + dirname + "/"):
        os.makedirs(source_dir + "/distributions/" + dirname + "/")
    plt.savefig(source_dir + "/distributions/" + dirname + "/" + result_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        files = list(os.walk(sys.argv[1] + "/brightness_maps/" + sys.argv[2]))[0][2]
        print(files)
        print(get_thresholds_and_stat_error(sys.argv[1], sys.argv[2], files))
    elif len(sys.argv) == 4:
        data_patches, threshold = get_data_and_threshold(sys.argv[1], sys.argv[2], sys.argv[3])
        if threshold is not None:
            threshold = data_patches[0][threshold]
        print(threshold)
    elif len(sys.argv) == 5:
        values = get_values(get_data(sys.argv[1] + "/brightness_maps/" + sys.argv[2] + "/" + sys.argv[3]))
        draw_distribution(values, sys.argv[1], sys.argv[2], sys.argv[4])
