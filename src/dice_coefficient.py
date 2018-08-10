"""
Dice coefficient
"""

import keras.backend as KB
import tensorflow as tf
import numpy as np

def dice_coef(y_true, y_pred):
    y_true = KB.flatten(y_true)
    y_pred = KB.flatten(y_pred)
    intersection = KB.sum(y_true * y_pred)
    denominator = KB.sum(y_true) + KB.sum(y_pred)
    if denominator == 0:
        return 1
    if intersection == 0:
        return 1 / (denominator + 1)
    return (2.0 * intersection) / denominator


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
