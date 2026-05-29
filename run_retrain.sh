model_arch=vgg16_bn_lth
ForgetType=LT
seed=3407
NumIndexes=22500
data_path=./data
dataset=cifar100
epoch_num=150
gpu=0
LT_type=LT
python main_forget.py --data ${data_path} --arch ${model_arch} --dataset ${dataset}  --save_dir ./checkpoints/unlearn/retrain/${seed}/${dataset}_${model_arch}_${NumIndexes}_rl_${ForgetType}/  --model_path ./checkpoints/scratch/${seed}/${dataset}_${model_arch}/0checkpoint.pth.tar   --unlearn retrain --num_indexes_to_replace ${NumIndexes} --unlearn_epochs ${epoch_num} --unlearn_lr 0.01 --seed ${seed} --gpu ${gpu} --batch_size 512 --sample_forget_type ${LT_type}