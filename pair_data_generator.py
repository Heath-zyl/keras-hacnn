import numpy as np
import keras
import keras.utils.np_utils as np_utils
import numpy as np
import os.path as osp
import keras.preprocessing.image as image
import os
import cv2
import random
import sys
import glob

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dir, mode='training', batch_size=32, target_size=(160,64), learn_region=True):
        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.target_size = target_size
        self.learn_region = learn_region
        self.data_dir = data_dir
        self.train_ids = self.get_id(self.data_dir)

    def __len__(self):
        'Denotes the number of batches per epoch'
        img_paths = glob.glob(osp.join(self.data_dir, '*.jpg'))
        return int(np.floor(len(img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_images_l, batch_images_r, batch_labels = self.read_data()
        return [batch_images_l, batch_images_r], batch_labels

    def on_epoch_end(self):
        'Updates indices after each epoch'
        '''
        self.indices = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        '''

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        return image.img_to_array(img)

    def get_pair(self, path, ids, positive):
        pair = []
        pic_name = []
        files = os.listdir(path)
        if positive:
            value = random.sample(ids, 1)
            id = [str(value[0]), str(value[0])]
        else:
            id = random.sample(ids, 2)
        id = [str(id[0]), str(id[1])]
        for i in range(2):
            id_files = [f for f in files if f.split('_')[0] == id[i]]
            pic_name.append(random.sample(id_files, 1))
        for pic in pic_name:
            pair.append(os.path.join(path, pic[0]))
        
        return pair
        
    def get_id(self, path):
        files = os.listdir(path)
        
        IDs = []
        for f in files:
            IDs.append(f.split('_')[0])
        
        IDs = list(set(IDs))
        return IDs

    def read_data(self):
        images_l = []
        images_r = []
        labels = []
        for i in range(self.batch_size // 2):
            pairs = [self.get_pair(self.data_dir, self.train_ids, True), self.get_pair(self.data_dir, self.train_ids, False)]  
            for pair in pairs:
                for p_idx, p in enumerate(pair):
                    if p_idx == 0:
                        images_l.append(p)
                    else :
                        images_r.append(p)
            labels.append([1., 0.])
            labels.append([0., 1.])

        batch_images_l = np.array(list(map(self.load_image, images_l)))
        batch_images_r = np.array(list(map(self.load_image, images_r)))
        
        return batch_images_l, batch_images_r, np.array(labels)