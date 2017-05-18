import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys
import os

# Define a function to return some characteristics of the dataset


def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


def supported_image(name):
    lc_name = str.lower(name)
    if 'jpg' in lc_name or 'jpeg' in lc_name or 'png' in lc_name:
        return True
    else:
        return False


def load_training_images(path):
    carList = []
    noncarList = []
    # print("traverse ", path)
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        # print('Found directory: %s' % dirName)
        if "non-vehicles" in dirName:
            for fname in fileList:
                #print('\t%s' % fname)
                if supported_image(fname):
                    noncarList.append(os.path.join(dirName, fname))
        else:
            for fname in fileList:
                #print('\t%s' % fname)
                if supported_image(fname):
                    carList.append(os.path.join(dirName, fname))
    np.random.shuffle(carList)
    np.random.shuffle(noncarList)
    return carList, noncarList


def load_images(path):
    images = []
    # print("traverse ", path)
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for fname in fileList:
            #print('\t%s' % fname)
            if supported_image(fname):
                images.append(os.path.join(dirName, fname))
    np.random.shuffle(images)
    return images


def load_training_data(path):
    cars, notcars = load_training_images(path)
    print("get ", len(cars), " vehicel images")
    print("get ", len(notcars), " non-vehicle images")

    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    return cars, notcars, data_info


def main(path):
    cars, notcars, data_info = load_training_data(path)

    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    fig.savefig("output_images/training_data.jpg")


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    main(test_dir)
