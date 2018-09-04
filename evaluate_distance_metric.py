import os
import sys
import numpy as np
import argparse

import keras
from keras.utils import plot_model
import keras.preprocessing.image as image

import models.hacnn as modellib
import data_manager
from data_generator import DataGenerator
import losses
from eval_metrics import evaluate


def parse_args(args):
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--batch-size', default=32, type=int, help="train batch size")
    parser.add_argument('--dataset-path', type=str, default='data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=data_manager.get_names())
    parser.add_argument('--height', type=int, default=160, help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=64, help="width of an image (default: 128)")
    parser.add_argument('--learn-region', type=int, default=True, help="learn_region")
    parser.add_argument('--snapshot', help='saved weights')
    
    return parser.parse_args(args)

def load_weights(model, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = model.inner_model.layers if hasattr(model, "inner_model") else model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # data generator
    dataset = data_manager.init_imgreid_dataset(
        dataset_path=args.dataset_path, name=args.dataset
    )

    query_generator = DataGenerator(dataset.query[0],
                                    dataset.query[1], 
                                    batch_size=args.batch_size, 
                                    num_classes=dataset.num_query_pids, 
                                    target_size=(args.height, args.width), 
                                    learn_region=args.learn_region,
                                    shuffle=False,
                                    mode='inference')

    gallery_generator = DataGenerator(  dataset.gallery[0],
                                        dataset.gallery[1], 
                                        batch_size=args.batch_size, 
                                        num_classes=dataset.num_gallery_pids, 
                                        target_size=(args.height, args.width), 
                                        learn_region=args.learn_region,
                                        shuffle=False,
                                        mode='inference')

    model = modellib.HACNN(mode='inference', 
                           num_classes=dataset.num_query_pids, 
                           batch_size=args.batch_size, 
                           learn_region=args.learn_region).model

    load_weights(model, filepath=args.snapshot, by_name=True)
    
    '''
    img_path = '/Users/luke/Documents/ml_datasets/person_re_id/videotag_scene/dataset_7_lite/bounding_box_train/3434_0096.jpg'
    img = image.load_img(img_path, target_size=(args.height, args.width))
    img = image.img_to_array(img)
    
    ouput = model.predict(np.array([img]) ,verbose=1)
    print('ouput', np.array(ouput).shape)
    print('ouput', np.argmax(ouput[0]), np.argmax(ouput[1]))
    '''
    # evaluate
    qf, q_pids = [], []
    for index in range(len(query_generator)):
        imgs, pids = query_generator[index]
        print('query', index)
        features = model.predict(imgs ,verbose=0)
        print('features', features)
        qf.append(features)
        q_pids.extend(pids)
    qf = np.concatenate(qf, axis=0)
    q_pids = np.asarray(q_pids)
    print(qf.shape)
        
    gf, g_pids = [], []
    for index in range(len(gallery_generator)):
        print('gallery', index)
        imgs, pids = gallery_generator[index]
        features = model.predict(imgs ,verbose=0)
        print('features', features)
        gf.append(features)
        g_pids.extend(pids)
    gf = np.concatenate(gf, axis=0)
    g_pids = np.asarray(g_pids)
    
    m, n = qf.shape[0], gf.shape[0]
    
    # qf = qf*0.001
    qf_pow = np.power(qf, 2)
    qf_sum = np.sum(qf_pow, axis=1, keepdims=True)
    qf_ext = np.repeat(qf_sum, n, axis=1)

    # gf = gf*0.01
    gf_pow = np.power(gf, 2)
    gf_sum = np.sum(gf_pow, axis=1, keepdims=True)
    gf_ext = np.repeat(gf_sum, m, axis=1)
    gf_ext_t = gf_ext.T
    distmat = qf_ext + gf_ext_t
    distmat = distmat + np.dot(qf, gf.T)*(-2.0)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, None, None, dataset_type=args.dataset)
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    ranks=[1, 5, 10, 20]
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")
    
if __name__ == '__main__':
    main()