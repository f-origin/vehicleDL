#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
#output_dir=/output  # 训练目录
#dataset_dir=/data/forigin/my-object # 数据集目录，这里是写死的，记得修改

output_dir=/home/david/tmp/voc-model  # 训练目录
dataset_dir=/home/david/tmp/voc-model/data # 数据集目录，这里是写死的，记得修改

# 在test.jpg上验证导出的模型
python3 inference.py --output_dir=${output_dir} --dataset_dir=${dataset_dir}
