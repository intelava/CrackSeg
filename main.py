

import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from glob import glob
from PIL import Image
import random

from model import *
from dataPull import *

run = True

while(run == True):

    print("Welcome to Crack Seg!")
    print("Press 1 to train a new model")
    print("Press 2 to load the model")
    print("Press 3 to make predictions")

    operation = input("Your choice:")

    if operation == '1':
        train, validation = getGenerator()
        unet_model = createAndTrain(train,validation)

    elif operation == '2':
        unet_model = tf.keras.models.load_model()

    elif operation == '3':
        try:
            unet_model
        except NameError:
            print("\nPlease create or load a model first!\n\n")
            continue
        else:
             image = unet_model.predict(prepareImage())
             plt.imshow(image)
