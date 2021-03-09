import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
import cv2


def generate_numpy_images(csv_file):
    """
    0=Angry
    1=Disgust
    2=Fear
    3=Happy
    4=Sad
    5=Surprise
    6=Neutral
    """

    print("Data not found. Processing now")

    # Load the data
    data = pd.read_csv(csv_file)

    # Get rid of extra data
    data = data[(data['emotion'] == 3) | (
        data['emotion'] == 5) | (data['emotion'] == 6)]

    # Reassign the labels
    data.loc[data['emotion'] == 3, ['emotion']] = 0
    data.loc[data['emotion'] == 5, ['emotion']] = 1
    data.loc[data['emotion'] == 6, ['emotion']] = 2

    # Get Labels column
    labels = data.iloc[:, 0].values

    # Get Image column
    images = data.iloc[:, 1].values

    # Process the images
    images = np.array([np.array(list(map(int, image.split()))).reshape(
        (48, 48, 1))/255 for image in tqdm(images)])

    print("Images and Labels Processed\n")

    # Train Test Splitting
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.05)

    # Save images localy
    print("Populating Training data")
    counter = 0
    for image, label in zip(train_images, train_labels):
        if label == 0:
            cv2.imwrite(
                f"./fer2013/processed/train/{label}/{counter}.jpg", image*255)
            counter += 1
        elif label == 1:
            cv2.imwrite(
                f"./fer2013/processed/train/{label}/{counter}.jpg", image*255)
            counter += 1
        else:
            cv2.imwrite(
                f"./fer2013/processed/train/{label}/{counter}.jpg", image*255)
            counter += 1

    print("Populating Testing data")
    counter = 0
    for image, label in tqdm(zip(test_images, test_labels)):
        if label == 0:
            cv2.imwrite(
                f"./fer2013/processed/test/{label}/{counter}.jpg", image*255)
            counter += 1
        elif label == 1:
            cv2.imwrite(
                f"./fer2013/processed/test/{label}/{counter}.jpg", image*255)
            counter += 1
        else:
            cv2.imwrite(
                f"./fer2013/processed/test/{label}/{counter}.jpg", image*255)
            counter += 1

    print("Data generated")
