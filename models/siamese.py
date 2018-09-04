import keras
from keras.utils import plot_model
import keras.layers as KL
import keras.models as KM
import models.hacnn as modellib

def network(snapshot, mode='training', num_classes=2, batch_size=32, learn_region=True, backbone=True):
    cnn_1 = modellib.HACNN( mode='training', 
                            num_classes=2, 
                            learn_region=learn_region,
                            backbone=True).model

    cnn_2 = modellib.HACNN( mode='training', 
                            num_classes=2, 
                            learn_region=learn_region,
                            backbone=True).model

    # rename layer         
    for layer in cnn_1.layers:
        layer.trainable = True
        layer.name = layer.name + "_1"
    for layer in cnn_2.layers:
        layer.trainable = True
        layer.name = layer.name + "_2"
    
    feature_1 = cnn_1.output
    feature_2 = cnn_2.output
    concat = KL.concatenate([feature_1, feature_2], name='fc_concat')
    fc = KL.Dense(512, activation='relu', name='fc_2')(concat)
    fc = KL.Dense(2, activation='softmax', name='fc_1')(fc)

    model = KM.Model(inputs=[cnn_1.input, cnn_2.input], outputs=fc)

    if snapshot:
        model.load_weights(snapshot, by_name=True, skip_mismatch=False)

    return model