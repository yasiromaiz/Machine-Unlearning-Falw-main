dataset_type=cifar100
model_arch=resnet18
ForgetType=LT
seed=4307
NumIndexes=22500
unlearn_lr=0.005
gpu=6
alpha=0.5
epoch_num=20
mask_rate=0.5
other_info=_0.5
data_path=./data/
python generate_mask.py --data ${data_path} --arch ${model_arch}  --dataset ${dataset_type}  --save_dir ./checkpoints/mask/${seed}/${dataset_type}_${model_arch}_${NumIndexes}_${ForgetType} --model_path ./checkpoints/scratch/${seed}/${dataset_type}_${model_arch}/0checkpoint.pth.tar --num_indexes_to_replace ${NumIndexes} --unlearn_epochs 1 --seed ${seed} --gpu ${gpu} --sample_forget_type ${ForgetType}