import sys
import copy
import os
from collections import OrderedDict
import numpy as np 
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate

def main():
    args = arg_parser.parse_args()
    print("args:",str(args))

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    # os.makedirs(args.save_dir, exist_ok=True)
    # log_path = os.path.join(args.save_dir, "output.log")
    # sys.stdout = open(log_path, "a",buffering=1)
    # sys.stderr = sys.stdout

    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = torch.nn.functional.cross_entropy
    distribution = {}
    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        distribution[name] = utils.calculate_dataset_distribution(loader,args)
        print(f"{name} distribution: {distribution[name]}")
    args.distribution = distribution

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        assert (False)


        
    

    out = []
    model.eval()
    with torch.no_grad():
        for  image,target in forget_loader :
            image = image.cuda()
            target = target.cuda()
            index = target==0
            image = image[index]
            target = target[index]
            model = model.cuda()
            p = model(image).softmax(dim=-1)
            p_c = p[torch.arange(image.shape[0]),target]
            out.append(p_c)
    out = torch.concatenate(out,dim=0)
    # np.save(f'./tmp/forget_p_c/{args.unlearn}_{args.dataset}_{args.num_indexes_to_replace}_{args.sample_forget_type}.npy',out.cpu().numpy())            
    np.save(f'./tmp/forget_val_p_c/forget_{args.dataset}_{args.unlearn}_{args.num_indexes_to_replace}_{args.sample_forget_type}.npy',out.cpu().numpy())            
    
    out = []
    model.eval()
    with torch.no_grad():
        for  image,target in val_loader :
            image = image.cuda()
            target = target.cuda()
            index = target==0
            image = image[index]
            target = target[index]
            model = model.cuda()
            p = model(image).softmax(dim=-1)
            p_c = p[torch.arange(image.shape[0]),target]
            out.append(p_c)
    out = torch.concatenate(out,dim=0)
    
    np.save(f'./tmp/forget_val_p_c/val_{args.dataset}_{args.unlearn}_{args.num_indexes_to_replace}_{args.sample_forget_type}.npy',out.cpu().numpy())            
    
    # 删除model最后一个全连接层
    # 针对常见的模型结构（如resnet、vgg等），通常最后一个全连接层为model.fc或model.classifier
    # 这里做通用处理，若有fc属性则置为nn.Identity，否则若有classifier则置为nn.Identity
    
    # if hasattr(model, 'fc'):
    #     model.fc = nn.Identity()
    #     print("已删除model的fc层（最后一个全连接层）")
    # elif hasattr(model, 'classifier'):
    #     model.classifier = nn.Identity()
    #     print("已删除model的classifier层（最后一个全连接层）")
    # else:
    #     print("未找到可删除的全连接层，请手动检查模型结构")

    # class_idx = [0,1,50,51,98,99]
    # retain_features = []
    # retain_labels = []
    # forget_features = [] 
    # forget_labels = []
    # model.eval()
    # with torch.no_grad():
    #     for i,c_idx in enumerate(class_idx):
    #         features = []
    #         targets = [] 
    #         for image,tar in retain_loader:
    #             image = image.cuda()
    #             tar = tar.cuda()
    #             index = tar==c_idx
    #             image = image[index]
    #             tar = tar[index]
    #             model = model.cuda()
    #             f = model(image)
    #             features.append(f)
    #             targets.append(torch.zeros_like(tar)+i)
    #         retain_features.append(torch.concatenate(features,dim=0))
    #         retain_labels.append(torch.concatenate(targets,dim=0))
            
    #         features = []
    #         targets = []
    #         for image,tar in forget_loader:
    #             image = image.cuda()
    #             tar = tar.cuda()
    #             index = tar==c_idx
    #             image = image[index]
    #             tar = tar[index]
    #             model = model.cuda()
    #             f = model(image)
    #             features.append(f)
    #             targets.append(torch.zeros_like(tar)+i)
    #         forget_features.append(torch.concatenate(features,dim=0))
    #         forget_labels.append(torch.concatenate(targets,dim=0))
    #     retain_features = torch.concatenate(retain_features,dim=0)
    #     retain_labels = torch.concatenate(retain_labels,dim=0)
    #     forget_features = torch.concatenate(forget_features,dim=0)
    #     forget_labels = torch.concatenate(forget_labels,dim=0)
        
    #     np.save(f'./tmp/features/{args.unlearn}_{args.num_indexes_to_replace}_{args.sample_forget_type}.npy',{
    #         "retain_features":retain_features.cpu().numpy(),
    #         "retain_labels":retain_labels.cpu().numpy(),
    #         "forget_features":forget_features.cpu().numpy(),
    #         "forget_labels":forget_labels.cpu().numpy()
    #     })
    
    

if __name__ == "__main__":
    main()
    
    
# tsne_model = TSNE(n_components=2, perplexity=10, n_iter=2000, random_state=42)
# transformed_data = tsne_model.fit_transform(features)