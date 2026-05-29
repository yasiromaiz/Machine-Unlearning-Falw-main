model_arch=vgg16_bn_lth
ForgetType=LT
seed=4307
NumIndexes=9000
gpu=0
epoch_num=150
data_path=./data/
lr=0.01
python main_train.py --data ${data_path} --save_dir ./checkpoints/scratch/${seed}/${dataset}_${model_arch}/  --arch ${model_arch} --dataset ${dataset} --lr ${lr} --epochs ${epoch_num} --seed ${seed} --gpu ${gpu}