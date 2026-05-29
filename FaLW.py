import time
from copy import deepcopy

import random
import numpy as np
import torch
import torch.utils
import torch.utils.data
import utils

from .impl import iterative_unlearn


class MarkedDataset(torch.utils.data.Dataset):
    # forget:source = 1, retain:source = 0
    def __init__(self,dataset,source):
        # self.data = dataset.data
        # self.target = dataset.targets
        self.dataset = dataset
        self.source = source
        if self.source :
            self.random_target = np.random.randint(0, max(self.dataset.targets)+1,self.dataset.targets.shape)
        else:
            self.random_target = deepcopy(self.dataset.targets)
        
    def __getitem__(self, index) :
        image,target = self.dataset[index]
        return image,target,self.random_target[index], self.source
    def __len__(self):
        return len(self.dataset)
    

def calc_w_f_i(p_i,mu_i,sigma_i,alpha,beta,distribution_f,target_f):
    distribution_f = torch.tensor(distribution_f).cuda()
    B_c = (distribution_f.sum()/distribution_f.clamp(min=1)/distribution_f.shape[0])**alpha
    B_i = B_c[target_f]
    # T_i = 1 + ((p_i-mu_i).relu()**beta)
    # T_i = 1+ torch.sign(p_i-mu_i)*((p_i-mu_i).abs()**beta)
    # T_i = torch.ones(T_i.shape).cuda()
    z_i = (p_i-mu_i)/sigma_i
    T_i = 1 + torch.sign(z_i)*(torch.nn.functional.tanh(z_i.abs()))**(1/B_i)
    return T_i 
    # return B_i*T_i
    # return T_i

# def calc_w_r_i(alpha,distribution_r,target_r):
#     distribution_r = torch.tensor(distribution_r).cuda()
#     B_c = (distribution_r.max()/distribution_r.clamp(min=1))**alpha
#     B_i = B_c[target_r]
#     return B_i

def get_val_predict_distribution(model,args,val_loader):
    model.eval()
    class_sums = torch.zeros(args.num_classes,requires_grad=False).cuda()
    class_sumsq = torch.zeros(args.num_classes, requires_grad=False).cuda()  # 新增
    class_counts = torch.zeros(args.num_classes,requires_grad=False).cuda()
    with torch.no_grad():
        for it,(image,targets) in enumerate(val_loader):
            image = image.cuda()
            targets = targets.cuda()
            outputs = model(image)
            probabilities = outputs.softmax(dim=-1)
            p_i = torch.gather(probabilities, 1, targets.unsqueeze(1)).squeeze()
            class_sums.index_add_(0, targets, p_i)
            class_sumsq.index_add_(0, targets, p_i ** 2)
            class_counts.index_add_(0, targets, torch.ones_like(p_i))
    model.train()
    
    mu_c = class_sums/class_counts.clamp(min=1)
    mean_sq_c = class_sumsq / class_counts.clamp(min=1)
    sigma_c = (mean_sq_c - mu_c ** 2).clamp(min=0).sqrt()
    assert(torch.all((mu_c >= 0) & (mu_c <= 1)).item())
    assert(mu_c.shape[0]==args.num_classes and sigma_c.shape[0]==args.num_classes)
    # print("val mu:",mu_c)
    # print('max mu:',mu_c.max(),'min mu:',mu_c.min())
    # print('max sigma:',sigma_c.max(),'min sigma:',sigma_c.min())
    model.train()
    return mu_c,sigma_c

@iterative_unlearn
def FaLW(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    val_loader = data_loaders['val']
    forget_dataset = deepcopy(forget_loader.dataset)
    
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet" or args.dataset == 'cifar10':
        # exit(0)
        # try:
        #     forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
        # except:
        #     forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))

        retain_dataset = retain_loader.dataset

        train_dataset = torch.utils.data.ConcatDataset([MarkedDataset(forget_dataset,1),MarkedDataset(retain_dataset,0)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        losses_r = utils.AverageMeter()
        top1_r = utils.AverageMeter()
        losses_f = utils.AverageMeter()
        top1_f = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
    
        loader_len = len(forget_loader) + len(retain_loader)
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
            
        # train_class_sums = torch.zeros(args.num_classes,requires_grad=False).cuda()
        # train_class_counts = torch.zeros(args.num_classes,requires_grad=False).cuda()
        for it, (image, target,target_random ,source) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            target_random = target_random.cuda()
            source = source.cuda()
            retain_mask = (source == 0)
            forget_mask = (source == 1)
            outputs = model(image)
            mu_c,sigma_c = get_val_predict_distribution(model,args,val_loader)
            # calc loss for retain
            if retain_mask.sum() > 0 :
                # image_r = image[retain_mask]
                target_r = target[retain_mask]
                outputs_r = outputs[retain_mask]
                # w_r_i = calc_w_r_i(args.alpha,args.distribution['retain'],target_r).detach()
            
                # loss_r = (w_r_i*criterion(outputs_r,target_r,reduction="none")).sum()
                loss_r = criterion(outputs_r,target_r,reduction="none").sum()
            
            else:
                assert(False)
            
            if forget_mask.sum() > 0:
                # image_f = image[forget_mask]
                target_f = target[forget_mask]
                target_random_f = target_random[forget_mask]
                outputs_f = outputs[forget_mask]
                loss_f_samples = criterion(outputs_f,target_random_f,reduction="none")
                probs = torch.nn.functional.softmax(outputs_f, dim=-1)
                p_i = probs.gather(1, target_f.unsqueeze(1)).squeeze()
                # ==============================
                # train_class_sums.index_add_(0, target_f, p_i)
                # train_class_counts.index_add_(0, target_f, torch.ones_like(p_i))
                # ==============================
                mu_i = mu_c[target_f]
                # ==============================
                small_count = (p_i<mu_i).sum()
                big_count = (p_i>=mu_i).sum()
                # print("small_count:",small_count,"big_count:",big_count)
                # ==============================
                sigma_i = sigma_c[target_f]
                w_f_i = calc_w_f_i(p_i,mu_i,sigma_i,args.alpha,args.beta,args.distribution['forget'],target_f).detach()
                loss_f = (w_f_i * loss_f_samples).sum()
                # exit(0)
                # loss_f = (loss_f_samples).sum()
            else:
                assert(False)

            
            loss = (loss_f + loss_r)/target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"参数 {name} 的梯度形状: {param.grad.shape}")
            #         print(f"参数 {name} 的梯度范数: {param.grad.norm()}")
            #         # 打印梯度的前10个元素（如果维度较大）
            #         print(f"参数 {name} 的梯度前10个元素: {param.grad.flatten()[:10]}")
            #     else:
            #         print(f"参数 {name} 的梯度为None（可能未参与损失计算）")
            # exit(0)
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()
            # measure accuracy and record loss on r
            with torch.no_grad():
                output_r = outputs_r.float()
                loss_r = loss_r.float()
                prec1_r = utils.accuracy(output_r.data, target_r)[0]
                losses_r.update(loss_r.item(), target_r.size(0))
                top1_r.update(prec1_r.item(), target_r.size(0))
                # measure accuracy and record loss on f 
                output_f = outputs_f.float()
                loss_f = loss_f.float()
                prec1_f = utils.accuracy(output_f.data, target_f)[0]
                losses_f.update(loss_f.item(), target_f.size(0))
                top1_f.update(prec1_f.item(), target_f.size(0))
            
            if (i + 1) % args.print_freq == 0:
                print("max w_f_i:",w_f_i.max(),'min w_f_i:',w_f_i.min())
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss_r {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                      'Accuracy_r {top1_r.val:.3f} ({top1_r.avg:.3f})\t'
                      'Loss_f {loss_f.val:.4f} ({loss_f.avg:.4f})\t'
                      'Accuracy_f {top1_f.val:.3f} ({top1_f.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end-start, loss_r=losses_r, top1_r=top1_r,loss_f=losses_f,top1_f=top1_f))
                start = time.time()
        # train_mu_c = train_class_sums/train_class_counts.clamp(min=1)
        # print("train_mu_c:",train_mu_c)
    elif args.dataset == "svhn": #  or args.dataset == "cifar10":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

    return top1_r.avg