import sys
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import warnings


warnings.filterwarnings('ignore', category=UserWarning, append=True)


def get_data(path):
    my_map = fits.open(path)
    image_data = my_map[0].data
    return image_data


def preprocess(source_map, ed_size, ed_order):
    result_map = source_map

    def erode():
        nonlocal result_map
        result_map = ndimage.binary_erosion(result_map).astype(result_map.dtype)

    def dilate():
        nonlocal result_map
        result_map = ndimage.binary_dilation(result_map).astype(result_map.dtype)

    mask = 1
    for i in range(ed_size):
        if (ed_order and mask) == 0:
            erode()
        else:
            dilate()
        mask *= 2
    return result_map


def watershed_cluster(source_map):
    shape = source_map.shape
    xm, ym = np.ogrid[0:shape[0]:10, 0:shape[1]:10]
    markers = np.zeros_like(source_map).astype(np.int16)
    markers[xm, ym] = np.arange(xm.size * ym.size).reshape((xm.size, ym.size))
    res = ndimage.watershed_ift(source_map.astype(np.uint8), markers)
    res[xm, ym] = res[xm - 1, ym - 1]
    return res


def watershed_color(clustered_map, magnetic_map):
    background = np.argmax(np.bincount(clustered_map.flat))
    signs = {}
    for i in range(clustered_map.shape[0] - 1):
        for j in range(clustered_map.shape[1] - 1):
            if clustered_map[i, j] in signs:
                signs[clustered_map[i, j]] += magnetic_map[i, j]
            else:
                signs[clustered_map[i, j]] = magnetic_map[i, j]
    holes_map = clustered_map * 0
    for i in range(clustered_map.shape[0] - 1):
        for j in range(clustered_map.shape[1] - 1):
            if clustered_map[i, j] == background:
                holes_map[i, j] = 0
            elif signs[clustered_map[i, j]] > 0:
                holes_map[i, j] = 1
            else:
                holes_map[i, j] = -1
    return holes_map


def watershed(source_map, magnetic_map, k):
    return watershed_color(watershed_cluster(source_map), magnetic_map)


def kmeans(source_map, magnetic_map, k):
    hole_points = []
    signs = []
    for i in range(source_map.shape[0] - 1):
        for j in range(source_map.shape[1] - 1):
            if magnetic_map[i][j] != 0 and source_map[i][j] == 1:
                hole_points.append([i, j])
                signs.append((magnetic_map[i][j] > 0) * 2 - 1)
    clusters = KMeans(n_clusters=k, random_state=0).fit(hole_points)
    signs_by_cluster = [[] for i in range(k)]
    for i in range(len(clusters.labels_)):
        signs_by_cluster[clusters.labels_[i]].append(signs[i])
    signs_of_clusters = []
    for i in signs_by_cluster:
        signs_of_clusters.append((sum(i) > 0) * 2 - 1)
    result = source_map * 0
    for i in range(len(hole_points)):
        result[hole_points[i][0]][hole_points[i][1]] = signs_of_clusters[clusters.labels_[i]]
    result *= (source_map == 1)
    return result


def cmeans(source_map, magnetic_map, k):
    hole_points = [[], []]
    for i in range(source_map.shape[0] - 1):
        for j in range(source_map.shape[1] - 1):
            if source_map[i][j] == 1:
                hole_points[0].append(j)
                hole_points[1].append(i)
    all_data = np.vstack((hole_points[0], hole_points[1]))
    center, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(all_data, k, 2, error=0.005, maxiter=1000)
    signs_by_cluster = [[] for i in range(k)]
    for i in range(k):
        xes = all_data[0, u_orig.argmax(axis=0) == i]
        yes = all_data[1, u_orig.argmax(axis=0) == i]
        for j in range(len(xes)):
            signs_by_cluster[i].append((magnetic_map[yes[j]][xes[j]] > 0) * 2 - 1)
    signs_of_clusters = []
    for i in signs_by_cluster:
        signs_of_clusters.append((sum(i) > 0) * 2 - 1)
    result = source_map * 0
    for i in range(k):
        xes = all_data[0, u_orig.argmax(axis=0) == i]
        yes = all_data[1, u_orig.argmax(axis=0) == i]
        for j in range(len(xes)):
            result[yes[j]][xes[j]] = signs_of_clusters[i]
    result *= (source_map == 1)
    return result


def clusterize(preprocessed_map, magnetic_map, method, k):
    methods = [kmeans, cmeans, watershed]
    return methods[method](preprocessed_map, magnetic_map, k)


def make_chmap(brightness_map, magnetic_map, threshold, ed_size, ed_order, method, k):
    holes_map = (brightness_map < threshold) * 1
    preprocessed_map = preprocess(holes_map, ed_size, ed_order)
    if method is None:
        return preprocessed_map
    return clusterize(preprocessed_map, magnetic_map, method, k)


def draw_chmap_to_files(source_dir, dir_name, brightness_name, magnetic_name, name, threshold, ed_size, ed_order, method, k):
    brightness_map = get_data(source_dir + "/brightness_maps/" + dir_name + "/" + brightness_name)
    magnetic_map = get_data(source_dir + "/magnetic_maps/" + dir_name + "/" + magnetic_name)
    chmap = make_chmap(brightness_map, magnetic_map, threshold, ed_size, ed_order, method, k)
    result_dir = "ch_maps"
    if not os.path.exists(source_dir + "/" + result_dir + "/" + dir_name + "/"):
        os.makedirs(source_dir + "/" + result_dir + "/" + dir_name + "/")
    fits.PrimaryHDU(chmap).writeto(source_dir + "/" + result_dir + "/" + dir_name + "/" + name, overwrite=True)


if __name__ == '__main__':
    draw_chmap_to_files(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        float(sys.argv[6]),
        int(sys.argv[7]),
        int(sys.argv[8]),
        int(sys.argv[9]),
        int(sys.argv[10])
    )
