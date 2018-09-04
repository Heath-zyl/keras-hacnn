import numpy as np
import keras
import keras.utils.np_utils as np_utils
from PIL import Image
import numpy as np
import os.path as osp
import keras.preprocessing.image as image

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

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, pids, mode='training', batch_size=32, num_classes=10, shuffle=True, target_size=(160,64), learn_region=True):
        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.images = images
        self.pids = pids
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.target_size = target_size
        self.learn_region = learn_region
        self.on_epoch_end()
        print('self.num_classes', self.num_classes, target_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        images_batch = [self.images[k] for k in indices]
        pids_batch = [self.pids[k] for k in indices]

        # Generate data
        X, Y = self.__data_generation(images_batch, pids_batch)
        # print('__getitem__', 'X', X.shape, 'Y', Y.shape)

        if self.mode == 'training':
            if self.learn_region:
                return X, [Y, Y]
            else :
                return X, Y
        else:
            return X, Y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        return image.img_to_array(img)

    def __data_generation(self, images_temp, Y):
        'Generates data containing batch_size samples'
        X = np.array(list(map(self.load_image, images_temp)))
        if self.mode == 'training':
            Y = keras.utils.to_categorical(Y, num_classes=self.num_classes)

        # print('__data_generation', Y[0].shape, Y[0])
        # Y = smooth_labels(keras.utils.to_categorical(Y, self.num_classes), .1)
        # print('__data_generation', Y_test[0].shape, Y_test[0])
        
        return X, Y