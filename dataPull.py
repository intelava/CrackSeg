from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
data_gen_args = dict(rescale=1./255)
datagen = ImageDataGenerator(**data_gen_args, validation_split=0.2)
seed = 1

doShuffle = False
batchSize = 8
image_generator = datagen.flow_from_directory(
    'images',
    class_mode=None,
    color_mode='rgb',
    target_size=(224,224),
    batch_size=batchSize,
    shuffle=doShuffle,
    subset='training',
    seed=seed)


mask_generator = datagen.flow_from_directory(
    'masks',
    class_mode=None,
    color_mode='grayscale',
    target_size=(224,224),
    batch_size=batchSize,
    subset='training',
    shuffle=doShuffle,
    seed=seed)

val_image_generator = datagen.flow_from_directory(
    'images',
    class_mode=None,
    color_mode='rgb',
    target_size=(224,224),
    batch_size=batchSize,
    shuffle=doShuffle,
    subset='validation',
    seed=seed)


val_mask_generator = datagen.flow_from_directory(
    'masks',
    class_mode=None,
    color_mode='grayscale',
    target_size=(224,224),
    batch_size=batchSize,
    subset='validation',
    shuffle=doShuffle,
    seed=seed)


def getGenerator():
    train_generator = zip(image_generator, mask_generator)
    test_generator = zip(val_image_generator, val_mask_generator)
    return train_generator, test_generator

def prepareImage():
    image = cv2.imread("../predict/image.jpg", cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_expanded = np.expand_dims(image, axis=0)
    img_expanded = img_expanded.astype('float64')
    img_expanded /= 255.0
    return img_expanded