from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501 import Market1501
from .videotag import VIDEOtag


__imgreid_factory = {
    'market1501': Market1501,
    'videotag': VIDEOtag
}


__vidreid_factory = {
    
}


def get_names():
    return list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)