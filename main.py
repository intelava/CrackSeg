# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import datetime
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt



from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from glob import glob
from tensorflow.keras.optimizers import Adam


import scipy
import os

import keras

from PIL import Image, ImageOps
from skimage import transform
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


def iou(y_true, y_pred):
   def f(y_true, y_pred):
      intersection = (y_true * y_pred).sum()
      union = y_true.sum() + y_pred.sum() - intersection
      x = (intersection + 1e-15) / (union + 1e-15)
      x = x.astype(np.float32)
      return x

   return tf.numpy_function(f, [y_true, y_pred], tf.float32)


smooth = 1e-15


def dice_coef(y_true, y_pred):
   new = y_true.numpy()



   # Adds a subplot at the 1st position
   fig = plt.figure()
   fig.add_subplot(1, 2, 1)

   # showing image
   plt.imshow(new[0,:,:,0])
   plt.axis('off')
   plt.title("y_true")

   # Adds a subplot at the 2nd position
   fig.add_subplot(1, 2, 2)

   # showing image
   plt.imshow(y_pred[0,:,:])
   plt.axis('off')
   plt.title("pred")
   plt.show()
   y_true = tf.keras.layers.Flatten()(y_true)
   y_pred = tf.keras.layers.Flatten()(y_pred)
   intersection = tf.reduce_sum(y_true * y_pred)
   return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
   return 1.0 - dice_coef(y_true, y_pred)




def loadCV(image, mask):
   image = image.decode()
   mask = mask.decode()
   image = cv2.imread(image, cv2.IMREAD_COLOR)
   image = image.astype(np.float32)
   #image = cv2.resize(image,(160,160))

   mask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
   mask = mask.astype(np.float32)
   #mask = cv2.resize(mask, (160, 160))
   mask = np.expand_dims(mask, axis=-1)
   return image,mask

def load2(X,Y):
   newx = load(X)
   newy = load(Y)
   return newx, newy

def create_set(X,Y):
   dataset = tf.data.Dataset.from_tensor_slices((X,Y))
   dataset = dataset.map(tf_parse)
   dataset = dataset.batch(2)
   dataset = dataset.prefetch(10)
   return dataset


def tf_parse(x, y):

   x, y = tf.numpy_function(loadCV, [x, y], [tf.float32, tf.float32])
   x.set_shape([448, 448, 3])
   y.set_shape([448, 448, 1])
   return x, y


def SqueezeAndExcite(inputs, ratio=8):
   init = inputs
   filters = init.shape[-1]
   se_shape = (1, 1, filters)

   se = GlobalAveragePooling2D()(init)
   se = Reshape(se_shape)(se)
   se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
   se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
   x = init * se
   return x


def ASPP(inputs):
   """ Image Pooling """
   shape = inputs.shape
   y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
   y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
   y1 = BatchNormalization()(y1)
   y1 = Activation("relu")(y1)
   y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

   """ 1x1 conv """
   y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
   y2 = BatchNormalization()(y2)
   y2 = Activation("relu")(y2)

   """ 3x3 conv rate=6 """
   y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
   y3 = BatchNormalization()(y3)
   y3 = Activation("relu")(y3)

   """ 3x3 conv rate=12 """
   y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
   y4 = BatchNormalization()(y4)
   y4 = Activation("relu")(y4)

   """ 3x3 conv rate=18 """
   y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
   y5 = BatchNormalization()(y5)
   y5 = Activation("relu")(y5)

   y = Concatenate()([y1, y2, y3, y4, y5])
   y = Conv2D(256, 1, padding="same", use_bias=False)(y)
   y = BatchNormalization()(y)
   y = Activation("relu")(y)

   return y

def deeplabv3_plus(shape):
   """ Input """
   inputs = Input(shape)

   """ Encoder """
   encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

   image_features = encoder.get_layer("conv4_block6_out").output
   x_a = ASPP(image_features)
   x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

   x_b = encoder.get_layer("conv2_block2_out").output
   x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
   x_b = BatchNormalization()(x_b)
   x_b = Activation('relu')(x_b)

   x = Concatenate()([x_a, x_b])
   x = SqueezeAndExcite(x)

   x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
   x = BatchNormalization()(x)
   x = Activation('relu')(x)

   x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
   x = BatchNormalization()(x)
   x = Activation('relu')(x)
   x = SqueezeAndExcite(x)

   x = UpSampling2D((4, 4), interpolation="bilinear")(x)
   x = Conv2D(1, 1)(x)
   x = Activation("sigmoid")(x)

   model = Model(inputs, x)
   return model


#test ve train klasörlerini aşağı yazın 
images = sorted(glob(os.path.join("Btrain/images", "*jpg")))
maskes = sorted(glob(os.path.join("Btrain/masks", "*jpg")))

images2 = sorted(glob(os.path.join("Btest/images", "*jpg")))
masks2 = sorted(glob(os.path.join("Btest/masks", "*jpg")))

images = images[100:200]
masks = maskes[100:200]





dataset = create_set(images,masks)

datasetVal = create_set(images2,masks2)




model = deeplabv3_plus((448,448,3))


model.compile(optimizer="adam",loss=dice_loss,metrics=[dice_coef, iou,'accuracy'],run_eagerly=True)

model.fit(dataset,epochs=3,validation_data=datasetVal)

model.save('')
