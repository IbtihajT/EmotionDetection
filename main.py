from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import matplotlib.pyplot as plt
from data_processor import generate_numpy_images
from models import CNNArchitecture
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# To regenerate the results
seed = 7
np.random.seed(7)


# Tensorflow Stuff
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train_model(model, callbacks_list, name, train_batchs, test_batchs):

    # Start Training
    history = model.fit_generator(
        train_batches,
        validation_data=test_batches,
        epochs=200,
        verbose=1,
        steps_per_epoch=100,
        validation_steps=100,
        callbacks=callbacks_list
    )

    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title(f'{name}_model_accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{name}_model_loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f"./model_checkpoints/{name}/{name}.png")


if __name__ == "__main__":

    # Path to the dataset
    path_to_csv = "./fer2013/fer2013.csv"

    # Get the Processed Images and Labels

    if (len(os.listdir("./fer2013/processed/train/0")) == 0):
        generate_numpy_images(path_to_csv)

    # Process data for models
    train_path = "./fer2013/processed/train"
    test_path = "./fer2013/processed/test"

    train_batches = ImageDataGenerator().flow_from_directory(
        train_path, target_size=(224, 224), classes=['happy', 'neutral', 'surprise'], batch_size=8)
    test_batches = ImageDataGenerator().flow_from_directory(
        test_path, target_size=(224, 224), classes=['happy', 'neutral', 'surprise'], batch_size=4)

    # Meta Data
    input_shape = train_batches.image_shape
    num_classes = len(np.unique(train_batches.classes))
    loss = 'categorical_crossentropy'
    optimizer = Adam()

    # Get model architure
    cnn_model = CNNArchitecture(input_shape, num_classes, loss, optimizer)

    # Train VGG
    vgg_16, callbacks_list, name = cnn_model.VGG_16()
    train_model(vgg_16, callbacks_list, name, train_batches, test_batches)

    # Train Resnet
    resnet_50, callbacks_list, name = cnn_model.resnet_50()
    train_model(resnet_50, callbacks_list, name, train_batches, test_batches)

    # Train Inception
    inception_v3, callbacks_list, name = cnn_model.inception_v3()
    train_model(inception_v3, callbacks_list,
                name, train_batches, test_batches)

    # Train Inception_Resnet_V2
    inception_resnet_v2, callbacks_list, name = cnn_model.inception_resnet_v2()
    train_model(inception_resnet_v2, callbacks_list,
                name, train_batches, test_batches)

    # Train DeXpression
    deXpression, callbacks_list, name = cnn_model.DeXpression()
    train_model(deXpression, callbacks_list, name, train_batches, test_batches)
