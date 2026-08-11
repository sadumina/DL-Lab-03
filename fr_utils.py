#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####

import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
#from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

import h5py
import matplotlib.pyplot as plt


_FLOATX = 'float32'

def variable(value, dtype=_FLOATX, name=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    _get_session().run(v.initializer)
    return v

def shape(x):
    return x.get_shape()

def square(x):
    return tf.square(x)

def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)

def concatenate(tensors, axis=-1):
    if axis < 0:
        axis = axis % len(tensors[0].get_shape())
    return tf.concat(axis, tensors)

def LRN2D(x):
    return tf.nn.lrn(x, alpha=1e-4, beta=0.75)

def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor

WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}


def load_weights_from_FaceNet(FRmodel, weights_dir=None):
    """Load weights into FRmodel from CSV files.

    Parameters
    ----------
    FRmodel : keras.Model
        The FaceNet Keras model instance.
    weights_dir : str or None
        Path to directory that contains *.csv weights. If None, common paths are tried.
    """
    weights = WEIGHTS
    weights_dict = load_weights(weights_dir)

    for name in weights:
        try:
            layer = FRmodel.get_layer(name)
            layer.set_weights(weights_dict[name])
        except Exception as e:
            # Layer not found or shape mismatch
            print(f"[load_weights_from_FaceNet] Skipping '{name}': {e}")


def load_weights(weights_dir=None):
    """
    Load FaceNet weights from CSV files. Robust to nested folders and different roots.

    Parameters
    ----------
    weights_dir : str or None
        Directory containing CSVs (e.g., '/content/weights'). If None, try common candidates.

    Returns
    -------
    dict : mapping from layer name to list of numpy arrays for weights
    """
    import glob

    # Candidate roots to search
    candidates = []
    if weights_dir:
        candidates.append(weights_dir)
    candidates += [
        "./weights",
        "/content/weights",
        "./content/weights",
        "./weights/weights",
        "/content/weights/weights",
    ]

    found_root = None
    for c in candidates:
        try_path = os.path.abspath(c)
        if os.path.isdir(try_path):
            csvs = glob.glob(os.path.join(try_path, "*.csv"))
            if csvs:
                found_root = try_path
                break
    if found_root is None:
        raise FileNotFoundError(
            "Could not locate weights CSVs. Please unzip so that you have a folder "
            "containing files like conv1_w.csv, bn1_w.csv, dense_w.csv."
        )

    # index all csv files by basename without extension
    fileNames = [f for f in os.listdir(found_root) if f.endswith(".csv")]
    paths = { os.path.splitext(n)[0] : os.path.join(found_root, n) for n in fileNames }

    weights_dict = {}
    missing = []

    for name in WEIGHTS:
        if 'conv' in name:
            w_key, b_key = name + '_w', name + '_b'
            if w_key not in paths or b_key not in paths:
                missing.append((name, [w_key, b_key]))
                continue
            conv_w = genfromtxt(paths[w_key], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[b_key], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]
        elif 'bn' in name:
            w_key, b_key, m_key, v_key = name + '_w', name + '_b', name + '_m', name + '_v'
            if any(k not in paths for k in [w_key, b_key, m_key, v_key]):
                missing.append((name, [w_key, b_key, m_key, v_key]))
                continue
            bn_w = genfromtxt(paths[w_key], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[b_key], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[m_key], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[v_key], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dw = os.path.join(found_root, 'dense_w.csv')
            db = os.path.join(found_root, 'dense_b.csv')
            if not (os.path.exists(dw) and os.path.exists(db)):
                missing.append((name, ['dense_w.csv', 'dense_b.csv']))
                continue
            dense_w = genfromtxt(dw, delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(db, delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    if missing:
        print("[load_weights] WARNING: Missing some expected CSVs. The following layer keys were not found:")
        for n, keys in missing:
            print(f"  - {n}: missing one of {keys}")

    return weights_dict



def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def img_to_encoding(image_path, model):
    """Read an image file, resize to 96x96, and produce a FaceNet embedding."""
    img_bgr = cv2.imread(image_path, 1)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = img_bgr[..., ::-1]
    # Resize to FaceNet expected size
    try:
        import tensorflow as _tf
        img_rgb = _tf.image.resize(img_rgb, (96, 96)).numpy()
    except Exception:
        img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    # Normalize and transpose to (C, H, W)
    img = np.around(np.transpose(img_rgb, (2, 0, 1)) / 255.0, decimals=12)
    x = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x)
    return embedding