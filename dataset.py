"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from tqdm import tqdm
import random


def cifar10_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar10", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    sample_forget_type = "random",
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    num_class = max(train_set.targets) + 1
    for i in range(num_class):
        class_idx = np.where(train_set.targets == i)[0]        
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            sample_Df__type= sample_forget_type,
            seed=seed - 1,
            only_mark=only_mark,
            num_classes=num_class
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)
        args.data_dir = os.path.join(args.data,args.data_dir)
        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        sample_forget_type = 'random',
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        num_class = max(train_set.targets) + 1
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                sample_Df__type= sample_forget_type,
                seed=seed - 1,
                only_mark=only_mark,
                num_classes=num_class
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


def cifar10_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    sample_forget_type = "random",
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    num_class = max(train_set.targets) + 1
    for i in range(num_class):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            sample_Df__type = sample_forget_type,
            seed=seed - 1,
            only_mark=only_mark,
            num_classes=num_class,
        )
        
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    # print("(train_set.labels<0).sum():",(train_set.targets<0).sum())
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        # print("good",indexes.shape,only_mark)
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    sample_Df__type: str = "random",
    seed: int = 0,
    only_mark: bool = False,
    num_classes: int = None,
):
    rng = np.random.RandomState(seed)
    if class_to_replace == -1:
        if sample_Df__type == "random":
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.targets))
            except:
                try:
                    indexes = np.flatnonzero(np.ones_like(dataset.labels))
                except:
                    indexes = np.flatnonzero(np.ones_like(dataset._labels))
        else:
            assert(num_indexes_to_replace is not None and num_classes is not None)
            indexes = []
            if sample_Df__type == "uniform":
                count_per_class = get_uniform_sample_counts(num_indexes_to_replace,num_classes)
            elif sample_Df__type == "LT":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class)
            elif sample_Df__type == "1_3LT":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class,order=1/3)
            elif sample_Df__type == "1_2LT":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class,order=1/2)            
            elif sample_Df__type == "softLT":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class,order=0.25)
            elif sample_Df__type == "LT1.5":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class,order=1.5)
            elif sample_Df__type == "LT2":
                max_samples_per_class = len(np.where(dataset.targets == 0)[0])
                count_per_class = get_long_tail_sample_counts(num_indexes_to_replace,num_classes,max_samples_per_class,order=2)
            # random.seed(seed)
            # random.shuffle(count_per_class) 
            for i in range(num_classes):
                try:
                    class_idx = np.where(dataset.targets == i)[0]
                except:
                    try:
                        class_idx = np.where(dataset.labels == i)[0]
                    except:
                        class_idx = np.where(dataset._labels == i)[0]
                
                indexes.append(
                    rng.choice(class_idx, count_per_class[i], replace=False)
                )
            indexes = np.hstack(indexes)
            print("the way to sample forget data is:",sample_Df__type)
            print("the number of samples to be forgotten in each class:",count_per_class)
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)
    if num_indexes_to_replace is not None and (class_to_replace != -1 or sample_Df__type == "random" ):
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
    print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


def get_long_tail_sample_counts(x: int, num_classes: int,  max_samples_per_class: int, order = 1 ) -> list[int]:
    """
    根据长尾分布，计算每个类别需要抽取的样本数，并考虑每个类别的最大可用样本数。

    Args:
        x (int): 需要抽取的总样本数。
        num_classes (int): 数据集中类别的总数量。
        max_samples_per_class (int): 每个类别可用的最大样本数。

    Returns:
        list[int]: 一个列表，表示每个类别需要抽取的样本数。
                   列表的索引对应类别ID (0 到 num_classes-1)。
    """
    # 1. 实现长尾分布的权重
    # 使用与类别索引的反比关系来定义权重
    # 类别0将具有最高权重，类别 num_classes-1 最低。
    class_indices = np.arange(num_classes)
    # 为了避免除以零并使其更直观，将类别索引加1
    weights = 1 / ((class_indices**order) + 1)
    probabilities = weights / np.sum(weights) # 将权重归一化为概率

    print(f"每个类别的长尾分布概率:\n{probabilities}")

    # 2. 根据概率计算初步期望样本数，并立即限制在 max_samples_per_class 内
    # 使用 float 保持精度，然后四舍五入并取最小值
    raw_desired_counts_float = x * probabilities
    num_samples_per_class_final = np.minimum(np.round(raw_desired_counts_float), max_samples_per_class).astype(int)

    # 3. 调整总和以确保恰好为 x
    current_total = np.sum(num_samples_per_class_final)
    diff = x - current_total

    # 需要添加样本
    if diff > 0:
        # 优先添加到长尾分布中权重更高的类别，但前提是该类别尚未达到最大样本数
        for _ in range(diff):
            # 找到可以继续添加样本的类别
            can_add_mask = (num_samples_per_class_final < max_samples_per_class)
            
            # 如果没有类别可以添加了，跳出循环 (x 太高，无法满足上限)
            if not np.any(can_add_mask):
                break
                
            # 在可以添加的类别中，选择概率最高的那个 (即长尾分布中更“重要”的类别)
            priorities = probabilities * can_add_mask # 将不能添加的类别的优先级置为0
            
            # 如果所有可添加的类别优先级都为0 (例如，所有类别都已达到最大值)
            if np.sum(priorities) == 0:
                break # 无法再添加任何样本
                
            # 获取要添加样本的类别索引
            idx_to_add = np.argmax(priorities)
            num_samples_per_class_final[idx_to_add] += 1

    # 需要移除样本
    elif diff < 0:
        # 优先从长尾分布中权重较低的类别（尾部）移除样本，但前提是该类别样本数大于0
        for _ in range(abs(diff)):
            # 找到可以移除样本的类别
            can_remove_mask = (num_samples_per_class_final > 0)
            
            # 如果没有类别可以移除了，跳出循环 (样本数已为0)
            if not np.any(can_remove_mask):
                break

            # 在可以移除的类别中，选择长尾概率最低的那个（即最接近尾部的）
            temp_probs = probabilities.copy()
            # 将不能移除的类别的概率设为无穷大，这样np.argmin就不会选中它们
            temp_probs[~can_remove_mask] = np.inf 
            
            # 如果所有可移除的类别优先级都为无穷大 (即所有类别样本数都为0)
            if np.all(temp_probs == np.inf):
                break # 无法再移除任何样本

            # 获取要移除样本的类别索引
            idx_to_remove = np.argmin(temp_probs)
            num_samples_per_class_final[idx_to_remove] -= 1
            
    return num_samples_per_class_final.tolist()

def get_uniform_sample_counts(x: int,num_classes: int) -> list[int]:
    """
    根据均匀分布，计算每个类别需要抽取的样本数。

    Args:
        num_classes (int): 数据集中类别的总数量。
        x (int): 需要抽取的总样本数。

    Returns:
        list[int]: 一个列表，表示每个类别需要抽取的样本数。
                   列表的索引对应类别ID (0 到 num_classes-1)。
    """
    # 均匀分布：每个类别的概率相同
    probabilities = np.full(num_classes, 1 / num_classes)

    print(f"每个类别的均匀分布概率:\n{probabilities}")

    # 计算每个类别需要抽取的样本数
    num_samples_per_class_desired = np.round(x * probabilities).astype(int)

    # 调整以确保由于四舍五入导致的总和恰好为 x
    diff = np.sum(num_samples_per_class_desired) - x
    if diff > 0:
        # 如果抽取的样本过多，从样本数最多的类别开始减去
        for _ in range(diff):
            max_idx = np.argmax(num_samples_per_class_desired)
            if num_samples_per_class_desired[max_idx] > 0: # 确保不会减到负数
                num_samples_per_class_desired[max_idx] -= 1
    elif diff < 0:
        # 如果抽取的样本过少，从样本数最多的类别开始加上
        for _ in range(abs(diff)):
            max_idx = np.argmax(num_samples_per_class_desired)
            num_samples_per_class_desired[max_idx] += 1
            
    return num_samples_per_class_desired.tolist()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
