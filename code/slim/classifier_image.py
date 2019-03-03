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

def test():
    print('*********test************')

def create_dict_from_label(label_file):
    label_dict = dict()
    for line in open(label_file):
        line_new = line.replace('\n','')
        list_line = line_new.split(':')
        label_dict[int(list_line[0])]=list_line[1]
    return label_dict
    

def create_graph(model_file):
    with open(model_file,'rb') as f:
        print("**************create graph**************")
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,name='')
        

def infrence_on_imageFile(image,model_file,label_file):
    
    if not tf.gfile.Exists(image):
        tf.logging.fatal('Image %s not exists' % image)
    image_data = open(image,'rb').read()
    label_dict = create_dict_from_label(label_file)
    
    create_graph(model_file)
    
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('output:0')
        predictions = sess.run(softmax_tensor,{'input:0':image_data})
        predictions = np.squeeze(predictions)
        
        
        top5_index = (np.argsort(-predictions)[:5])
        for item in top5_index:
            print('chexin:%s*******softmax:%s' % (label_dict[item],predictions[item]))
            
def infrence_on_image(imageTensor,label_file):    
    
    label_dict = create_dict_from_label(label_file)
    
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('output:0')
        predictions = sess.run(softmax_tensor,{'input:0':imageTensor})
        predictions = np.squeeze(predictions)
        
        maxIndex = np.argmax(predictions)
        sess.close()
        return maxIndex,label_dict[maxIndex]
        
#         top5_index = (np.argsort(-predictions)[:5])
#         for item in top5_index:
#             print('chexin:%s*******softmax:%s' % (label_dict[item],predictions[item]))
        