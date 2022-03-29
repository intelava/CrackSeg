import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#print(info)

def resize(input_image,input_mask):
    input_image=tf.image.resize(input_image,(128,128),method="nearest")
    input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
    return input_image,input_mask

def augment(input_image,input_mask):
    if tf.random.uniform(())>0.5:
        input_image=tf.image.flip_left_right(input_image)
        input_mask= tf.image.flip_left_right(input_mask)
    return input_image, input_mask

# scaling the images to the range of [-1, 1] and decreasing the image mask by 1
def normalize(input_image,input_mask):
    input_image=tf.cast(input_image,tf.float32)/255
    input_mask-=1
    return input_image, input_mask

def load_image_train(datapoint):
    input_image=datapoint["image"]
    input_mask=datapoint["segmentation_mask"]
    input_image,input_mask=resize(input_image,input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_image_test(datapoint):
    input_image=datapoint["image"]
    input_mask=datapoint["segmentation_mask"]
    input_image,input_mask=resize(input_image,input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

train_dataset=dataset["train"].map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
test_dataset=dataset["test"].map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE = 4
BUFFER_SIZE = 20
train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

#print(train_dataset)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")

    plt.show()
sample_batch = next(iter(train_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
display([sample_image, sample_mask])


def double_conv_block(x,n_filters):
    x=tf.keras.layers.Conv2D(n_filters,3,padding="same",activation="relu",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def downsample_block(x,n_filters):
    f=double_conv_block(x,n_filters)
    p=tf.keras.layers.MaxPool2D(2)(f)
    p=tf.keras.layers.Dropout(0.3)(p)
    return f,p

def upsample_block(x,conv_features,n_filters):
    x=tf.keras.layers.Conv2DTranspose(n_filters,3,2,padding="same")(x)
    x=tf.keras.layers.concatenate([x,conv_features])
    x=tf.keras.layers.Dropout(0.3)(x)
    x=double_conv_block(x,n_filters)
    return x

def unitModel():
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))
    # encoder: contracting path - downsample
    # 1 -downsample
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6-upsample
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


unet_model=unitModel()

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss="binary_crossentropy",
                   metrics="accuracy")

NUM_EPOCHS=20
TRAIN_LENGTH=info.splits["train"].num_examples
STEPS_PER_EPOCH=TRAIN_LENGTH//BATCH_SIZE
VAL_SUBSPLITS=5
TEST_LENGTH=info.splits["test"].num_examples
VALIDATION_STEPS=TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

unet_model.history=unet_model.fit(train_batches,
                             epochs=NUM_EPOCHS,
                             steps_per_epoch=STEPS_PER_EPOCH,
                             validation_steps=VALIDATION_STEPS,
                             validation_data=test_batches)


# unet_model.save("learnedModel.model")
# unet_model.save()
unet_model.save("learnedModel2")
unet_model.save()

# def create_mask(pred_mask):
#     pred_mask=tf.argmax(pred_mask,axis=-1)
#     pred_mask=pred_mask[..., tf.newaxis]
#     return pred_mask[0]
#
# def show_predictions(dataset=None,num=1):
#     if dataset:
#         for image,mask in dataset.take(num):
#             pred_mask=unet_model.predict(image)
#             display([image[0],mask[0],create_mask(pred_mask)])
#
#     else:
#         display([sample_image, sample_mask,
#                  create_mask(unet_model.predict(sample_image[tf.newaxis, ...]))])


count = 0
for i in test_batches:
   count +=1
print("number of batches:", count)

# model=tf.keras.models.load_model("learnedModel2")
# model.summary()
# import cv2
# img=cv2.imread("Abyssinian_1.jpg")
# img=cv2.resize(img,(128,128))
# img=np.expand_dims(img,axis=0)
# output=model.predict(img)
# print(output.shape)
# cv2.imshow("result ",output[0,:,:,:])
# cv2.waitKeyEx()
# cv2.imshow("result ",output[0,:,:,1])
# cv2.waitKeyEx()
# cv2.imshow("result ",output[0,:,:,2])
# cv2.waitKeyEx()
# cv2.imshow()

