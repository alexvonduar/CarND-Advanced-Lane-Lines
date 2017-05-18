import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from gaussian_fit import gaussian_sfit
from gaussian_fit import gaussian


def save_image(img, path, append=None):
    if path != "":
        head, tail = os.path.split(path)
        name, ext = os.path.splitext(tail)
        if append != None:
            # print("save name :", path, append)
            if os.path.exists(os.path.join(head, append)) == False:
                print("mkdir :", os.path.join(head, append))
                os.mkdir(os.path.join(head, append))
            # append = "_" + append
        savename = os.path.join(head, append, name + ext)
        cv2.imwrite(savename, img)
        # print("save file :", savename)


def save_hist(data, path, append=None):
    if path != "":
        head, tail = os.path.split(path)
        name, ext = os.path.splitext(tail)
        if append != None:
            # print("save name :", path, append)
            # append = "_" + append
            if os.path.exists(os.path.join(head, append)) == False:
                print("mkdir :", os.path.join(head, append))
                os.mkdir(os.path.join(head, append))
        n = int(data.shape[0] / 2)
        # varl = np.var(data[:n] / np.sum(data[:n]))
        # varr = np.var(data[n:] / np.sum(data[n:]))
        savename = os.path.join(head, append, name + ext)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

        data = data / np.sum(data)

        ax.plot(data, 'b')

        left = np.copy(data)
        left[n:] = 0
        right = np.copy(data)
        right[:n] = 0

        # left_fit = gaussian_fit(left)
        left_fit = gaussian_sfit(left)
        ax.plot(gaussian(left_fit, np.arange(len(data))), 'r')

        # right_fit = gaussian_fit(right)
        right_fit = gaussian_sfit(right)
        ax.plot(gaussian(right_fit, np.arange(len(right))), 'g')

        ax.set_title("left mean: " + "{0:f}".format(left_fit[0]) +
                     " stdev: " + "{0:f}".format(left_fit[1]) +
                     " max: " + "{0:f}\n".format(left_fit[2]) +
                     " right mean: " + "{0:f}".format(right_fit[0]) +
                     " stdev: " + "{0:f}".format(right_fit[1]) +
                     " max: " + "{0:f}".format(right_fit[2]))

        fig.savefig(savename)   # save the figure to file
        plt.close(fig)    # close the figure
