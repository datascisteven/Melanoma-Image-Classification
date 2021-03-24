import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, optimizers, applications
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class MergedGenerators(Sequence):
    def __init__(self, batch_size, generators=[], sub_batch_size=[]):
        self.generators = generators
        self.sub_batch_size = sub_batch_size
        self.batch_size = batch_size

    def __len__(self):
        return int(
            sum([(len(self.generators[idx]) * self.sub_batch_size[idx])
                 for idx in range(len(self.sub_batch_size))]) /
            self.batch_size)

    def __getitem__(self, index):
        """Getting items from the generators and packing them"""

        X_batch = []
        Y_batch = []
        for generator in self.generators:
            if generator.class_mode is None:
                x1 = generator[index % len(generator)]
                X_batch = [*X_batch, *x1]

            else:
                x1, y1 = generator[index % len(generator)]
                X_batch = [*X_batch, *x1]
                Y_batch = [*Y_batch, *y1]

        if self.generators[0].class_mode is None:
            return np.array(X_batch)
        return np.array(X_batch), np.array(Y_batch)


def build_datagenerator(dir1=None, dir2=None, batch_size=32):
    n_images_in_dir1 = sum([len(files) for r, d, files in os.walk(dir1)])
    n_images_in_dir2 = sum([len(files) for r, d, files in os.walk(dir2)])
    generator1_batch_size = int((n_images_in_dir1 * batch_size) /
                                (n_images_in_dir1 + n_images_in_dir2))
    generator2_batch_size = batch_size - generator1_batch_size
    generator1 = ImageDataGenerator(rescale=1./255).flow_from_directory(
        dir1,
        target_size=(256, 256),
        batch_size=generator1_batch_size,
        class_mode='binary'
    )
    generator2 = ImageDataGenerator(rescale=1./255).flow_from_directory(
        dir2,
        target_size=(256, 256),
        batch_size=generator2_batch_size,
        class_mode='binary'
    )
    return MergedGenerators(
        batch_size,
        generators=[generator1, generator2],
        sub_batch_size=[generator1_batch_size, generator2_batch_size])