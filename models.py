from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D, Activation, LeakyReLU, GlobalAveragePooling2D, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import Model


class CNNArchitecture:

    def __init__(self, input_shape, num_classes, loss, optimizer):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer

    def VGG_16(self):
        # Name of the Model
        name = "vgg_16"

        # Load the model with 'imagenet' weights
        vgg_model = VGG16(weights=None, include_top=False)

        # Freeze the training of the layers
        # for layer in vgg_model.layers:
        #     layer.trainable = False

        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input=vgg_model.input, outputs=predictions)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=['accuracy'])

        filepath = f"model_checkpoints/{name}/{name}" + \
            "_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list, name

    def resnet_50(self):
        # Name of the Model
        name = "resnet_50"

        # Load the model with 'imagenet' weights
        resnet_model = ResNet50(weights=None, include_top=False)

        # Freeze the training of the layers
        # for layer in resnet_model.layers:
        #     layer.trainable = False

        # Add layers
        x = resnet_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input=resnet_model.input, outputs=predictions)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=['accuracy'])

        filepath = f"model_checkpoints/{name}/{name}" + \
            "_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list, name

    def inception_v3(self):
        # Name of the Model
        name = "inception_v3"

        # Load the model with 'imagenet' weights
        inception_model = InceptionV3(weights=None, include_top=False)

        # Freeze the training of the layers
        # for layer in inception_model.layers:
        #     layer.trainable = False

        # Add Layers
        x = inception_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input=inception_model.input, outputs=predictions)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=['accuracy'])

        filepath = f"model_checkpoints/{name}/{name}" + \
            "_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list, name

    def inception_resnet_v2(self):
        # Name of the model
        name = "inception_resnet_v2"

        # Load the model with 'imagenet' weights
        inception_resnet_v2_model = InceptionResNetV2(
            weights=None, include_top=False)

        # Freeze the training of the layers
        # for layer in inception_resnet_v2_model.layers:
        #     layer.trainable = False

        # Add layers
        x = inception_resnet_v2_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input=inception_resnet_v2_model.input,
                      outputs=predictions)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=['accuracy'])

        filepath = f"model_checkpoints/{name}/{name}" + \
            "_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list, name

    def DeXpression(self):

        name = 'DeXpression'
        # Preprocess
        inputs = Input(shape=(224, 224, 3))
        convolution_1 = Convolution2D(filters=64, kernel_size=(7, 7), strides=(
            2, 2), activation='relu', padding='same', name='convolution_1')(inputs)
        pooling_1 = MaxPooling2D(pool_size=(3, 3), strides=(
            2, 2), padding='same', name='pooling_1')(convolution_1)
        lrn_1 = BatchNormalization(name='lrn_1')(pooling_1)

        # FeatEx_1
        convolution_2a = Convolution2D(filters=96, kernel_size=(
            1, 1), strides=(1, 1), activation='relu', name='convolution_2a')(lrn_1)
        convolution_2b = Convolution2D(filters=208, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation='relu', name='convolution_2b')(convolution_2a)
        pooling_2a = MaxPooling2D(pool_size=(
            3, 3), strides=(1, 1), padding='same', name='pooling_2a')(lrn_1)
        convolution_2c = Convolution2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', name='convolution_2c')(pooling_2a)
        concat_2 = concatenate(
            [convolution_2b, convolution_2c], name='concat_2')
        pooling_2b = MaxPooling2D(pool_size=(3, 3), strides=(
            2, 2), padding='same', name='pooling_2b')(concat_2)

        # FeatEx_2
        convolution_3a = Convolution2D(filters=96, kernel_size=(1, 1), strides=(
            1, 1), padding='valid', activation='relu', name='convolution_3a')(pooling_2b)
        convolution_3b = Convolution2D(filters=208, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation='relu', name="convolution_3b")(convolution_3a)
        pooling_3a = MaxPooling2D(pool_size=(3, 3), strides=(
            1, 1), padding='same', name='pooling_3a')(pooling_2b)
        convolution_3c = Convolution2D(filters=64, kernel_size=(1, 1), strides=(
            1, 1), activation='relu', padding='valid', name='convolution_3c')(pooling_3a)
        concat_3 = concatenate(
            [convolution_3b, convolution_3c], name='concat_3')
        pooling_3b = MaxPooling2D(pool_size=(3, 3), strides=(
            2, 2), padding='same', name='pooling_3b')(concat_3)

        # Final Layer
        classifier = Flatten()(pooling_3b)
        classifier = Dense(
            self.num_classes, activation='softmax', name='ouput')(classifier)

        # Generate The model
        model = Model(inputs, classifier, name='DeXpression')

        # Compile
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=['accuracy'])

        filepath = f"model_checkpoints/{name}/{name}" + \
            "_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list, name
