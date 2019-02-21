#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
output_dir=/output  # 训练目录
dataset_dir=/data/forigin/my-object # 数据集目录，这里是写死的，记得修改

# config文件??
PIPELINE_CONFIG_PATH=${dataset_dir}/ssd_mobilenet_v1_my.config
MODEL_DIR=${output_dir}/model
NUM_TRAIN_STEPS=500
SAMPLE_1_OF_N_EVAL_EXAMPLES=10

python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr


echo "############ export model #################"
# 导出模型
INPUT_TYPE=image_tensor
TRAINED_CKPT_PREFIX=${MODEL_DIR}/model.ckpt-${NUM_TRAIN_STEPS}
EXPORT_DIR=${output_dir}/export

python3 object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
echo "############ inference #################"
# 在test.jpg上验证导出的模型
python3 ./inference.py --output_dir=${output_dir} --dataset_dir=${dataset_dir}
