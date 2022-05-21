from keras import backend as K
from keras import layers
import keras
import tensorflow as tf


def dice_coefficient(true, pred):
    true_Flatten = K.flatten(true)
    pred_Flatten = K.flatten(pred)
    intersection = K.sum(true_Flatten * pred_Flatten)

    return (2.0 * intersection + 1e-6) / (K.sum(true_Flatten) + K.sum(pred_Flatten) - intersection + 1e-6)


def dice_loss(true, pred):
    coeff = dice_coefficient(true, pred)
    return 1 - coeff


def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model():
    inputs = layers.Input(shape=(224, 224, 3))

    encoder = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3),
                                       input_tensor=inputs)

    f1 = encoder.get_layer('block1_conv2').output
    f2 = encoder.get_layer('block2_conv2').output
    f3 = encoder.get_layer('block3_conv3').output
    f4 = encoder.get_layer('block4_conv3').output

    print(encoder.summary())

    bottleneck = encoder.get_layer('block5_conv3').output

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

def createAndTrain(train_generator,test_generator):
    model = build_unet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.1), loss=dice_loss, metrics=['accuracy'] )
    history = model.fit(train_generator,steps_per_epoch=1000,epochs=30,validation_data=test_generator,validation_steps=250)
    return history