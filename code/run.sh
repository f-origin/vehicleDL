#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
output_dir=/home/david/tmp/VOC2012-model  # 训练目录
dataset_dir=/home/david/tmp/VOC2012 # 数据集目录，这里是写死的，记得修改


PIPELINE_CONFIG_PATH=${dataset_dir}/ssd_mobilenet_v1_my.config
MODEL_DIR=${output_dir}/model
NUM_TRAIN_STEPS=5000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

echo "############ train & eval #################"
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr

echo "############ export #################"
# 导出模型
python3 object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${NUM_TRAIN_STEPS} \
    --output_directory=${output_dir}/exported_graphs
    
echo "############ inference #################"
# 在test.jpg上验证导出的模型
python3 inference.py --output_dir=${output_dir} --dataset_dir=${dataset_dir}