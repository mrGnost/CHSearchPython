import sys
import os

from astropy.io import fits
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore', category=UserWarning, append=True)


def get_data(path):
    my_map = fits.open(path)
    image_data = my_map[0].data
    return image_data


def draw(map_type, dirname, filename, result_name):
    data = get_data("./" + map_type + "/" + dirname + "/" + filename)
    plt.imshow(data, cmap='gray')
    if not os.path.exists("./" + map_type + "/" + dirname + "/"):
        os.makedirs("./" + map_type + "/" + dirname + "/")
    plt.savefig("./" + map_type + "/" + dirname + "/" + result_name, bbox_inches='tight')


if __name__ == '__main__':
    draw(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
