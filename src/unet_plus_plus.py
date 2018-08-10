import math
import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model, plot_model
from keras.losses import binary_crossentropy

from dice_coefficient import dice_coef_loss


class UNetPP(object):
    def __init__(self, input_shape, start_filter=32, depth=4, class_num=1):
        self.__input_shape = input_shape
        self.__class_num = class_num

        inputs = Input(self.__input_shape)

        filters_list = [start_filter * (k + 1) for k in range(depth)]
        layer = inputs
        encode_layers = []
        concat_layers_list = []
        for k, filters in enumerate(filters_list):
            layer = self.__add_encode_layers(filters, layer, is_first=(k==0))
            encode_layers.append(layer)
            concat_layers_list.append(layer)

        add_drop_layer_indexes = [2, 3]
        for base_k, bottom_layer in enumerate(encode_layers[1:]):
            deconv_item = zip(reversed(filters_list[:(base_k + 1)]), reversed(concat_layers_list[:(base_k + 1)]))
            sub_layer = bottom_layer
            for k, (filters, concat_layers) in enumerate(deconv_item):
                sub_layer = self.__add_decode_layers(filters, sub_layer, concat_layers
                                                     , add_drop_layer=(k in add_drop_layer_indexes))
                concat_layers_list[base_k - (k + 1)].append(sub_layer)

        layer = concatenate(concat_layers_list[0][1:])
        outputs = Conv2D(class_num, 1, activation='sigmoid')(layer)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])


    def __add_encode_layers(self, filter_size, input_layer, is_first=False):
        layer = input_layer
        if is_first:
            layer = Conv2D(filter_size, 3, padding='same', input_shape=self.__input_shape)(layer)
        else:
            layer = MaxPooling2D(2)(layer)
            layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)

        #layer = Conv2D(filter_size, 3, padding='same')(layer)
        #layer = BatchNormalization()(layer)
        #layer = Activation(activation='relu')(layer)
        return layer


    def __add_decode_layers(self, filter_size, input_layer, concat_layers, add_drop_layer=False):
        layer = UpSampling2D(2)(input_layer)
        layer = concatenate(concat_layers.insert(0, layer))

        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)

        #layer = Conv2D(filter_size, 3, padding='same')(layer)
        #layer = BatchNormalization()(layer)
        #layer = Activation(activation='relu')(layer)

        if add_drop_layer:
            layer = Dropout(0.5)(layer)
        return layer


    def comple_model(self):
        self.__model.compile(optimizer=Adam(), loss=self.__losses)


    def __losses(self, y_true, y_pred):
        binx_loss = binary_crossentropy(y_true, y_pred)
        dice_loss = dice_coef_loss(y_true, y_pred)
        return tf.reduce_sum(tf.stack([binx_loss, dice_loss]))


    def get_model(self, with_comple=False):
        if with_comple:
            self.comple_model()
        return self.__model


    def get_parallel_model(self, gpu_num, with_comple=False):
        self.__model = multi_gpu_model(self.__model, gpus=gpu_num)
        return self.get_model(with_comple)


    def show_model_summary(self):
        self.__model.summary()


    def plot_model_summary(self, file_name):
        plot_model(self.__model, to_file=file_name)

