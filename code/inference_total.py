import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from utils import visualization_utils as vis_util
from utils import label_map_util

from slim import classifier_image as ci

NUM_CLASSES = 1


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

def filterByThresh(boxes,classes,scores,min_score_thresh=.3,):
    boxes_new = list()
    classes_new = list()
    scores_new = list()
    for index,item in enumerate(scores):
        if item >= min_score_thresh:
            boxes_new.append(boxes[index])
            classes_new.append(classes[index])
            scores_new.append(scores[index])
    return np.array(boxes_new),np.array(classes_new),np.array(scores_new)
    

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'exported_graphs/frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels_items.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            image = Image.open(test_img_path)
            image_size = image.size
            image_width = image_size[0]
            image_height = image_size[1]
            
            image_np = load_image_into_numpy_array(image)
            
            
            model_file='/home/david/tmp/vehicle-model/freezed_inception_v4.pb'
            label_file = '/home/david/tmp/vehicle-model/labels.txt'
            ci.create_graph(model_file)
            

            
            
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            #[ymin, xmin, ymax, xmax]
            boxes,classes,scores = filterByThresh(boxes,classes,scores)
            category_index = {}
                
            for index,item in enumerate(boxes):
                
                offset_height = int(item[0]*image_height)
                offset_width = int(item[1]*image_width)

                max_height = item[2]*image_height
                max_width = item[3]*image_width

                target_height = int(max_height-offset_height)
                target_width = int(max_width-offset_width)
               
                tensor_str = tf.image.crop_to_bounding_box(image_np, offset_height, offset_width, target_height,target_width)
                
                
                vehicle_index,vehicle_label = ci.infrence_on_image(tf.image.encode_jpeg(tensor_str).eval(),label_file)
                
                classes[index]=vehicle_index
            
                category_index[vehicle_index] = {'id':vehicle_index,'name':vehicle_label}
            
           

               
            
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)
