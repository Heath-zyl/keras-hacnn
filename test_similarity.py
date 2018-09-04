import os
import sys
import argparse
import numpy as np
import keras.preprocessing.image as image
from models.siamese import network

def parse_args(args):
    parser = argparse.ArgumentParser(description='HA-CNN')
    parser.add_argument('--height', type=int, default=160, help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=64, help="width of an image (default: 128)")
    parser.add_argument('--snapshot', help='load training from a snapshot.')
    parser.add_argument('--image1', help='First image path to compare')
    parser.add_argument('--image2', help='Second image path to compare')
    
    
    return parser.parse_args(args)

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # create model
    model = network(args.snapshot)
    
    img_1 = image.load_img(args.image1, target_size=(args.height, args.width))
    img_1 = image.img_to_array(img_1)
    img_2 = image.load_img(args.image2, target_size=(args.height, args.width))
    img_2 = image.img_to_array(img_2)

    prediction = model.predict([[img_1], [img_2]], verbose=0)
    print('prediction', prediction)
    print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    main()