# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

def create_graph(model_file):
    with open(model_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,name='')

def infrence_on_image(image,model_file):
    if not tf.gfile.Exists(image):
        tf.logging.fatal('Image %s not exists' % image)
    image_data = open(image,'rb').read()
    
    create_graph(model_file)
    
    with tf.Session as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('output:0')
        predictions = sess.run(softmax_tensor,{'input:0':image_data})
        predictions = np.squeeze(predictions)
        print("***************")
        print(predictions)