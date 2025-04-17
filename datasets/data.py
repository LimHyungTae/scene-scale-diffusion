import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.carla_dataset import CarlaDataset
from datasets.threed_dataset import ThreeDFrontDataset


def get_data_id(args):
    return "{}".format(args.dataset)


def get_class_weights(freq):
    """
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    """
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))

    return weights


def get_data(args):
    if args.dataset == "carla":
        train_dir = os.path.join(args.dataset_dir, "Train")
        val_dir = os.path.join(args.dataset_dir, "Val")
        test_dir = os.path.join(args.dataset_dir, "Test")

        x_dim = 128
        y_dim = 128
        z_dim = 8
        data_shape = [x_dim, y_dim, z_dim]
        args.data_shape = data_shape

        binary_counts = True
        transform_pose = True
        remap = True

        train_ds = CarlaDataset(
            directory=train_dir,
            random_flips=True,
            remap=remap,
            binary_counts=binary_counts,
            transform_pose=transform_pose,
        )
        if remap:
            class_frequencies = train_ds.remap_frequencies_cartesian
            args.num_classes = 11
        else:
            args.num_classes = 23

        comp_weights = get_class_weights(class_frequencies).to(torch.float32)
        seg_weights = get_class_weights(class_frequencies[1:]).to(torch.float32)

        val_ds = CarlaDataset(
            directory=val_dir,
            remap=remap,
            binary_counts=binary_counts,
            transform_pose=transform_pose,
        )
        test_ds = CarlaDataset(
            directory=test_dir,
            remap=remap,
            binary_counts=binary_counts,
            transform_pose=transform_pose,
        )

        if args is not None and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=train_ds.collate_fn,
            num_workers=args.num_workers,
        )
        dataloader_val = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=val_ds.collate_fn,
            num_workers=args.num_workers,
        )
        dataloader_test = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_ds.collate_fn,
            num_workers=args.num_workers,
        )

    elif args.dataset == "3dfront":
        x_dim = 256
        y_dim = 256
        z_dim = 1
        args.data_shape = [x_dim, y_dim, z_dim]

        binary_counts = True
        remap = True

        train_ds = ThreeDFrontDataset(
            directory=args.dataset_dir,
            split="train",
            random_flips=True,
            remap=remap,
            binary_counts=binary_counts,
        )
        val_ds = ThreeDFrontDataset(
            directory=args.dataset_dir,
            split="val",
            remap=remap,
            binary_counts=binary_counts,
        )
        test_ds = ThreeDFrontDataset(
            directory=args.dataset_dir,
            split="test",
            remap=remap,
            binary_counts=binary_counts,
        )

        class_frequencies = train_ds.remap_frequencies_cartesian
        args.num_classes = 11  # same as Carla if remapped

        comp_weights = get_class_weights(class_frequencies).to(torch.float32)
        seg_weights = get_class_weights(class_frequencies[1:]).to(torch.float32)

        train_sampler = None
        val_sampler = None

        dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=train_sampler,
            collate_fn=train_ds.collate_fn,
            num_workers=args.num_workers,
        )
        dataloader_val = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=val_ds.collate_fn,
            num_workers=args.num_workers,
        )
        dataloader_test = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_ds.collate_fn,
            num_workers=args.num_workers,
        )

    else:
        raise NotImplementedError(
            "Wrong `dataset` has come. Other datasets are not supported."
        )

    return (
        dataloader,
        dataloader_val,
        dataloader_test,
        args.num_classes,
        comp_weights,
        seg_weights,
        train_sampler,
    )
