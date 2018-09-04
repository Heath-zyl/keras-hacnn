import keras
# from . import backend
import keras.backend as KB
import keras.utils.np_utils as np_utils

def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
     # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
     # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y

def deep_supervision(num_classes):
    def _cross_entropy_label_smooth(y_true, y_pred):
        labels          = y_true
        classification  = KB.softmax(y_pred)
        # classification  = smooth_labels(np_utils.to_categorical(classification, num_classes), .1)
        labels          = keras.utils.to_categorical(labels, num_classes=num_classes)
        cls_loss        = KB.categorical_crossentropy(labels, classification)
        return cls_loss

    def _deep_supervision(y_true, y_pred):
        # split - global feature, local feature
        y_pred_global = y_pred[:,:num_classes]
        y_pred_local = y_pred[:,num_classes:]

        loss = 0.
        loss += _cross_entropy_label_smooth(y_true, y_pred_global)
        loss += _cross_entropy_label_smooth(y_true, y_pred_local)
        loss /= 2
        return loss

    return _deep_supervision

def cross_entropy_label_smooth(num_classes):
    def _cross_entropy_label_smooth(y_true, y_pred):
        print('cross_entropy_label_smooth', y_true.shape, y_true)
        print('cross_entropy_label_smooth', y_pred.shape, y_pred)
        labels          = y_true
        # labels          = keras.utils.to_categorical(y_true, num_classes=num_classes)
        # labels          = keras.utils.to_categorical(labels, num_classes=num_classes)
        # labels          = smooth_labels(np_utils.to_categorical(labels, num_classes), .1)
        classification  = KB.softmax(y_pred)
        print('classification', classification)
        cls_loss        = KB.categorical_crossentropy(labels, classification)
        print('cls_loss', cls_loss.shape, cls_loss)
        return cls_loss

    return _cross_entropy_label_smooth