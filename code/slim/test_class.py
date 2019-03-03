# ==============================================================================
"""Generic training script that trains a model using a given dataset."""




import classifier_image as ci

image='/home/david/tmp/vehicle-model/test.jpg'
model_file='/home/david/tmp/vehicle-model/freezed_inception_v4.pb'
label_file = '/home/david/tmp/vehicle-model/labels.txt'
ci.infrence_on_imageFile(image,model_file,label_file)
