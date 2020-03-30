#!/bin/sh

python27 finetune_multiple_gpu.py \
	--learning_rate "0.00001" \
	--train_layers "fc"

python27 finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3"

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3,scale5/block2"

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3,scale5/block2,scale5/block1"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4/block6"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4/block6,scale4/block5"

# ============= 20190115
python27 get_image_feature.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3"

/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190115_193142/tensorboard --port 6066
http://euclid:6066

bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=example_model --model_base_path=/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/export_base
# =============
sudo docker run -p 8501:8501 --mount type=bind,source=/data/yp/export_base,target=/models/image_feature -e MODEL_NAME=image_feature -t tensorflow/serving &
/data/yp/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=8501 \
   --port=8500 \
   --model_base_path=/data/yp/export_base \
   --model_name="image_feature" \
   --enable_model_warmup=true

# =============center loss cate3
resnet_20190124_193003   # no filter
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190124_193003/tensorboard --port 6068
accuracy: 0.565
builder model epoch_10
category: 529
feature_dim: 100

resnet_20190128_154921  # add filter,epoch 15,python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190128_154921/tensorboard --port 6069
accuracy: 0.6238 # epoch 15

resnet_20190128_175701  # add l2,epoch 15,python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190128_175701/tensorboard --port 6070
accuracy: 0.6138 # epoch 15

resnet_20190128_181938  # feature dim 2,l2, epoch 15,python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190128_181938/tensorboard --port 6071
accuracy: 0.617 # epoch 15
builder
category: 88
feature_dim: 2

resnet_20190131_164405  # feature dim 5,total epoch 50,python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190131_164405/tensorboard --port 6072
accuracy: 0.6242 # epoch 15
builder
category: 88
feature_dim: 5

resnet_20190208_102654  # feature dim 2,l2,epoch 30,python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190208_102654/tensorboard --port 6073
accuracy: 0.6211 # epoch 15
builder
category: 88
feature_dim: 2

resnet_20190210_142128  # feature dim 5, epoch 30, python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190210_142128/tensorboard --port 6074
accuracy: 0.6285
builder: model 8 ***
category: 88
feature_dim: 5

resnet_20190210_143002  # feature dim 20, epoch 30, python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190210_143002/tensorboard --port 6075
accuracy: 0.6076 # epoch 13
builder
category: 88
feature_dim: 20

resnet_20190210_143711  # feature dim 20, epoch 30, python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5,scale4"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190210_143711/tensorboard --port 6076
accuracy: 0.4717 # epoch 7
builder
category: 88
feature_dim: 20

resnet_20190212_115639  # 152,feature dim 5, epoch 15, python27 get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5"
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190212_115639/tensorboard --port 6077
accuracy: 0.6445 # epoch 9
builder
category: 88
feature_dim: 5

# ======start server
#
/data/yp/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500 \
--rest_api_port=8501 \
--model_name=image_feature \
--model_base_path=/data/yp/export_base &

#
sudo nvidia-docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/data/yp/export_base,target=/models/image_feature -e MODEL_NAME=image_feature -t tensorflow/serving:latest-gpu &

# ============params select
l2_loss:
center_loss
center_feature_dim: 5
train_layer: fc+scale5
filter_limit:

# =========brand model
no_filter: 4973
    acc:
filter_1000: 100 ***
    acc:
    model: resnet_20190213_175759
    /opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190213_175759/tensorboard --port 6078
filter_500: 189
    acc:
# =========category 1 model
no_filter: 15
    acc:
filter_1000: 11
    acc:
filter_500: 12
    acc:
# =========category 2 model
no_filter: 82
    acc:
filter_1000: 41
    acc:
filter_500: 46
    acc:
# =========category 3 model
no_filter: 537
    acc:
filter_1000: 98 ***
    acc:
    model: resnet_20190213_180358
    /opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190213_180358/tensorboard --port 6079
    builder: epoch 9
filter_500: 144
    acc:
# ============series center loss category 3 model, feature 100
model: resnet_20190215_110219
/opt/anaconda2/bin/tensorboard --logdir ../training/resnet_20190215_110219/tensorboard --port 6080

# ============评测
1.类目2: 98 filter 100
acc: 0.7519 # epoch 9
model: resnet_20190215_162653

2.品牌: 616 filter 100
acc: 0.5022 # epoch 10
model: resnet_20190215_162640

# =============
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
python build_image_model.py
sudo docker run -p 8501:8501 --mount type=bind,source=/data/yp/export_base,target=/models/image_feature -e MODEL_NAME=image_feature -t tensorflow/serving &

python finetune.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"

# ====== 20200213
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
# ====== resnet_20200214_103624
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5,scale4"
# ====== 0.8692 resnet_20200214_222540
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3"
#
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt"
#
python get_image_feature.py --learning_rate "0.0001" --train_layers "fc,scale5" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt"

# 2020032410017
python get_image_feature.py --learning_rate "0.001" --train_layers "scale1,scale2,scale3,scale4,scale5,fc" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt" --is_pretrain "0" --batch_size 32

# 202003241346
python get_image_feature.py --learning_rate "0.00001" --train_layers "fc,scale5/block3" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt" --is_pretrain "1"
# 20200324_174746
python get_image_feature.py --learning_rate "0.001" --train_layers "fc,scale5" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt" --is_pretrain "1" --alpha_label_smooth 0.

2020-03-24 20:45:21.885732 Validation Accuracy = 0.7228
2020-03-24 20:45:22.542010 Validation F1 values:
               precision    recall  f1-score   support

          0       0.72      0.96      0.82      1898
          1       0.91      0.30      0.45       792
          2       0.86      0.07      0.12        89
          3       0.50      0.40      0.44       165

avg / total       0.76      0.72      0.68      2944

#
python get_image_feature.py --learning_rate "0.001" --train_layers "fc" --train_file "../data/train_13645_type7.txt" --val_file "../data/dev_3000_type7.txt" --is_pretrain "1" --alpha_label_smooth 0.
