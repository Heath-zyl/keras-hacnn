import os
import sys
import numpy as np
import argparse

import keras
from keras.utils import plot_model

import models.hacnn as modellib
import data_manager
from data_generator import DataGenerator
import losses


def parse_args(args):
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--batch-size', default=32, type=int, help="train batch size")
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help="initial learning rate")
    parser.add_argument('--dataset-path', type=str, default='data', help="root path to data directory")
    parser.add_argument('--dataset', type=str, default='market1501', choices=data_manager.get_names())
    parser.add_argument('--height', type=int, default=160, help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=64, help="width of an image (default: 128)")
    parser.add_argument('--learn-region', type=int, default=True, help="learn_region")
    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--log-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--save-summary', help='save summary to image.', type=int, default=False)
    
    return parser.parse_args(args)

def create_callbacks(args):
    callbacks = []
    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    os.makedirs(args.snapshot_path, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            args.snapshot_path,
            # '{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type=args.dataset)
            'hacnn_{dataset_type}_epoch_{{epoch:02d}}_loss_{{loss:.5f}}.h5'.format(dataset_type=args.dataset)
        ),
        verbose=1,
        save_weights_only=True
    )
    callbacks.append(checkpoint)

    os.makedirs(args.log_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir                = args.log_dir,
        histogram_freq         = 0,
        batch_size             = args.batch_size,
        write_graph            = True,
        write_grads            = False,
        write_images           = False,
        embeddings_freq        = 0,
        embeddings_layer_names = None,
        embeddings_metadata    = None
    )
    callbacks.append(tensorboard_callback)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # data generator
    dataset = data_manager.init_imgreid_dataset(
        dataset_path=args.dataset_path, name=args.dataset
    )
    print('dataset.train', len(dataset.train), dataset.num_train_pids)
    train_generator = DataGenerator(dataset.train[0], 
                                    dataset.train[1], 
                                    batch_size=args.batch_size, 
                                    num_classes=dataset.num_train_pids, 
                                    shuffle=True, 
                                    target_size=(args.height, args.width), 
                                    learn_region=args.learn_region)

    # create the callbacks
    callbacks = create_callbacks(args)

    # create model
    model = modellib.HACNN(mode='training', 
                           num_classes=dataset.num_train_pids, 
                           batch_size=args.batch_size, 
                           learn_region=args.learn_region).model

    # save model-summary
    if args.save_summary:
        plot_model(model, show_shapes=True, to_file='model.png')
        print(model.summary())
        
    # load saved weight
    if args.snapshot:
        model.load_weights(args.snapshot, by_name=True, skip_mismatch=False)

    # compile model
    if args.learn_region:
        model.compile(
            '''
            loss=losses.deep_supervision(num_classes=dataset.num_train_pids),
            loss={
                'classifier_global': losses.cross_entropy_label_smooth(num_classes=dataset.num_train_pids),
                'classifier_local': losses.cross_entropy_label_smooth(num_classes=dataset.num_train_pids),
            },
            '''
            loss='categorical_crossentropy',
            loss_weights={"classifier_global": 0.5, "classifier_local": 0.5},
            metrics=['accuracy'], 
            optimizer=keras.optimizers.SGD(lr=0.03),
        )
    else:
        model.compile(
            # loss=losses.cross_entropy_label_smooth(num_classes=dataset.num_train_pids),
            loss='categorical_crossentropy',
            metrics=['accuracy'], 
            optimizer=keras.optimizers.SGD(lr=0.03),
        )

    model.fit_generator(
        generator=train_generator,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

if __name__ == '__main__':
    main()