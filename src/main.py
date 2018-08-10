import os
import numpy as np
from matplotlib import pyplot
import keras.callbacks as KC
import math

from unet_plus_plus import UNetPP
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint


CLASS_NUM = 1
DEPTH = 4
INPUT_IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 20
EPOCHS = 1000
GPU_NUM = 4

START_FILTER = 32

DIR_BASE = os.path.join('.', '..')
DIR_MODEL = os.path.join(DIR_BASE, 'model')
DIR_INPUTS = os.path.join(DIR_BASE, 'inputs')
DIR_OUTPUTS = os.path.join(DIR_BASE, 'outputs')
DIR_TEACHERS = os.path.join(DIR_BASE, 'teachers')
DIR_TEST = os.path.join(DIR_BASE, 'test_data')
DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data')

File_MODEL = 'segmentation_model.hdf5'


def train(gpu_num=None, with_generator=False, show_info=True):
    print('network creating ... ', end='', flush=True)
    network = UNetPP(INPUT_IMAGE_SHAPE, start_filter=START_FILTER, depth=DEPTH, class_num=CLASS_NUM)
    print('... created')

    if show_info:
        network.plot_model_summary('../model_plot.png')
        network.show_model_summary()
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num, with_comple=True)
    else:
        model = network.get_model(with_comple=True)

    model_filename = os.path.join(DIR_MODEL, File_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     #, save_best_only=True
                                     , period=10
                                    )
                ]

    print('data generator creating ... ', end='', flush=True)
    train_generator = DataGenerator(DIR_INPUTS, DIR_TEACHERS, INPUT_IMAGE_SHAPE)
    print('... created')

    if with_generator:
        train_data_num = train_generator.data_size()
        #valid_data_num = train_generator.valid_data_size()
        his = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                                  , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                                  , epochs=EPOCHS
                                  , verbose=1
                                  , use_multiprocessing=True
                                  , callbacks=callbacks
                                  #, validation_data=valid_generator
                                  #, validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                                 )
    else:
        print('data generateing ... ') #, end='', flush=True)
        inputs, teachers = train_generator.generate_data()
        print('... generated')
        history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                            , shuffle=True, verbose=1, callbacks=callbacks)
    print('model saveing ... ', end='', flush=True)
    model.save_weights(model_filename)
    print('... saved')
    print('learning_curve saveing ... ', end='', flush=True)
    save_learning_curve(history)
    print('... saved')


def save_learning_curve(history):
    """ save_learning_curve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir, gpu_num=None):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE)

    network = UNetPP(INPUT_IMAGE_SHAPE, start_filter=START_FILTER, depth=DEPTH, class_num=CLASS_NUM)
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num)
    else:
        model = network.get_model()

    print('loading weghts ...')
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('... loaded')
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('result saveing ...')
    save_images(DIR_OUTPUTS, preds, file_names)
    print('... finish .')


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM, with_generator=False)
    #train(gpu_num=GPU_NUM, with_generator=True)

    #predict(DIR_INPUTS, gpu_num=GPU_NUM)
    #predict(DIR_PREDICTS, gpu_num=GPU_NUM)
