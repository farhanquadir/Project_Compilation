from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.__version__
#'2.1.0'
# To generate GIFs
#!pip install -q imageio
from glob import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from PIL import Image
from IPython import display
import sys, math
import keras.backend as K
#data reading

def get_image(image_path, width, height, mode):
    #print (image_path)
    image = Image.open(image_path)
    image = image.resize([width, height])
    # print("img",image)
    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    # print(image_files)
    data_batch = np.array([get_image(sample_file, width, height, mode) for sample_file in image_files])

    return data_batch

def get_all_data(image_files, width, height, mode):
    # print(image_files)
    data_batch = np.array([get_image(sample_file, width, height, mode) for sample_file in image_files])

    return data_batch


def gpu_support():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def training_details(_history):
    # Details of training
    train_loss = _history.history["loss"]
    val_loss = _history.history["val_loss"]
    train_acc = _history.history["accuracy"]
    val_acc = _history.history["val_accuracy"]

    # saving the results
    with open("gan_2" + '_' + ".txt", "w") as f:
        f.write("Epoch\t\t Train_Loss\t\t\t Val_loss\t\t\t Train_acc\t\t\t Val_acc\n")
        for ep in range(len(train_loss)):
            # f.write(str(ep) + "\t\t " + str(train_loss[ep]) + "\t\t " + str(val_loss[ep]) + "\t\t " + "\n")
            f.write(str(ep) + "\t\t " + str(train_loss[ep]) + "\t\t " + str(val_loss[ep]) + "\t\t " + str(
                train_acc[ep]) + "\t\t " + str(val_acc[ep]) + "\n")

BUFFER_SIZE = 60000
BATCH_SIZE = 256
batch_size=BATCH_SIZE
#ite=0
gpu_support()
data_dir="/data/farhan/Unsupervised_Learning/DCGAN/img_align_celeba/img_align_celeba/"
#(train_images, train_labels), (_, _) = tf.keras.datasets.CelebA.load_data()

#train_images = get_all_data(glob(data_dir+"*.jpg"),28,28,'RGB')
n_iterations = math.floor(len(glob(data_dir+"*.jpg")) / batch_size)

#train_images=get_batch(glob(os.path.join(data_dir, '*.jpg'))[ite * batch_size:(ite + 1) * batch_size], 28, 28, 'RGB')
#print (train_images.shape)
#sys.exit()
#train_images = train_images.reshape(train_images.shape[0], 28, 28, 3).astype('float32')
#print (train_images.shape)
#train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#print(type(train_dataset.size()))
#print(train_dataset.size())
#sys.exit()


#Discriminator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
# COMPUTING THE DISCRIMINATOR LOSS
def discriminator_loss(real_output, fake_output):
    real_loss = K.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = K.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
# COMPUTING THE GENERATOR LOSS
def generator_loss(fake_output):
    return K.binary_crossentropy(tf.ones_like(fake_output), fake_output)#cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(images):
    noise_dim = 100
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      with open ("generator_loss.txt","w+") as f:
          f.write(str(gen_loss)+"\n")
      with open ("discriminator_loss.txt","w+") as f:
          f.write(str(disc_loss)+"\n")
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(epochs):
  for epoch in range(epochs):
    start = time.time()
    
    for ite in range(n_iterations):
        noise_dim = 100
        num_examples_to_generate = 16
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        train_images=get_batch(glob(os.path.join(data_dir, '*.jpg'))[ite * batch_size:(ite + 1) * batch_size],
                                         28, 28, 'RGB')
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 3).astype('float32')
        train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
        for image_batch in train_dataset:
          train_step(image_batch)
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        # Save the model every 15 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
  
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(4,4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# This method returns a helper function to compute cross entropy loss
generator=make_generator_model()
discriminator=make_discriminator_model()
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#cross_entropy = tf.keras.losses.binary_crossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#loses="binary_crossentropy"
generator


#save checkpoints
checkpoint_dir = './training_checkpoints/'
if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#EPOCHS = 50
EPOCHS = 1
train(EPOCHS)