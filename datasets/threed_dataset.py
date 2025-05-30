import os
import pickle

import numpy as np
import yaml
from threed_front.room_datasets.rendering import render_polygons_to_canvas
from threed_front.room_datasets.room_graph_dataset import (
    DatasetCollection,
    Jitter,
    PartialPatch,
    RoomClassEncoder,
    RoomConnectionEncoder,
    RoomGraphAugmentationBase,
    RoomPositionEncoder,
    RoomShapeEncoder,
    RotationAugmentation,
)
from torch.utils.data import Dataset

MAX_BOUND = 12.0
EMPTY_CLASS_NUM = 17


class SamplePointCloud(RoomGraphAugmentationBase):
    def __init__(self, dataset, render_size, render_bound=None):
        """
        Args:
            dataset (RoomGraphDataset): dataset to encode
            render_size (np.array): (height, width) of the rendered image
            render_bound (tuple of np.array): 2D room position bound in meters
            (default: [-12, -12], [12, 12])
        """
        super().__init__(dataset)
        pos_min, pos_max = dataset.bounds["positions"]
        self._render_size = np.array(render_size).astype(np.int64)
        self._render_bound = (
            (
                np.array(render_bound[0]).astype(np.float32),
                np.array(render_bound[1]).astype(np.float32),
            )
            if render_bound is not None
            else (
                np.array([-MAX_BOUND, -MAX_BOUND], dtype=np.float32),
                np.array([MAX_BOUND, MAX_BOUND], dtype=np.float32),
            )
        )
        assert np.all(self.render_size > 0), "Invalid render size"
        assert np.all(self.render_bound[0] < pos_min[[0, 2]]) and np.all(
            self.render_bound[1] > pos_max[[0, 2]]
        ), "Invalid render bound"

    @property
    def render_size(self):
        return self._render_size

    @property
    def render_bound(self):
        return self._render_bound

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        class_indices = sample_params["class_labels"].argmax(axis=1)  # [N x C]
        positions = sample_params["positions"]  # [N x 3]
        shapes = sample_params["shapes"]  # [N x (Px2)]

        # compute position shifted point clouds
        point_clouds = (
            shapes.reshape(shapes.shape[0], -1, 2)
            + positions.reshape(positions.shape[0], 1, 3)[:, :, [0, 2]]
        )
        pos2d_min = self.render_bound[0]
        pos2d_max = self.render_bound[1]
        img = np.full(self.render_size, len(self._dataset.room_types), dtype=np.uint8)
        img = render_polygons_to_canvas(
            point_clouds, pos2d_min, pos2d_max, img, class_indices.tolist()
        )
        sample_params.update(
            {
                "point_clouds": point_clouds,  # [N x P x 2], np.float32
                "map_img": img,  # [H x W], np.uint8
            }
        )

        # compute original position shifted point clouds if original data is available
        if all(
            [
                k + "_orig" in sample_params
                for k in ["class_labels", "positions", "shapes"]
            ]
        ):
            # if original data is available, return the original data
            class_indices = sample_params["class_labels_orig"].argmax(axis=1)
            positions = sample_params["positions_orig"]
            shapes = sample_params["shapes_orig"]
            point_clouds = (
                shapes.reshape(shapes.shape[0], -1, 2)
                + positions.reshape(positions.shape[0], 1, 3)[:, :, [0, 2]]
            )
            img = np.full(
                self.render_size, len(self._dataset.room_types), dtype=np.uint8
            )
            img = render_polygons_to_canvas(
                point_clouds, pos2d_min, pos2d_max, img, class_indices.tolist()
            )
            sample_params.update(
                {
                    "point_clouds_orig": point_clouds,
                    "map_img_orig": img,
                }
            )

        return sample_params


class ThreeDFrontDataset(Dataset):
    """ThreeDFrontDataset for baseline comparison"""

    def __init__(
        self,
        directory,
        split,
        augmentations=[],
        min_size=(2.0, 2.0),
        max_size=(20.0, 20.0),
        patch_bound=((-12.0, -12.0), (12.0, 12.0)),
        render_bound=None,
        voxelize_input=True,
        binary_counts=True,
        random_flips=False,
        num_frames=1,
        transform_pose=True,
        get_gt=True,
    ):
        """Constructor.
        Parameters:
            directory: directory to the dataset
        """
        base_dir = os.path.dirname(__file__)
        config_file = os.path.join(base_dir, "3dfront.yaml")
        threed_config = yaml.safe_load(open(config_file, "r"))
        LABELS_REMAP = threed_config["learning_map"]
        REMAP_FREQUENCIES = threed_config["remap_content"]
        FREQUENCIES = threed_config["content"]

        self.LABELS_REMAP = np.asarray(list(LABELS_REMAP.values()))
        self.frequencies_cartesian = np.asarray(list(FREQUENCIES.values()))
        self.remap_frequencies_cartesian = np.asarray(list(REMAP_FREQUENCIES.values()))
        self.remap_colormap = threed_config["color_map"].values()

        self.data_shape = [256, 256, 1]

        self.get_gt = get_gt
        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = directory
        self._num_frames = num_frames
        self.random_flips = random_flips
        self.transform_pose = transform_pose
        self.sparse_output = True

        pickled_dataset_path = os.path.join(directory, "room_graph_dataset.pkl")

        raw_data = pickle.load(open(pickled_dataset_path, "rb"))
        raw_dataset = raw_data[f"{split}_dataset"]
        # [N x C] one-hot encoding of size len(dataset.class_labels)
        self.class_labels = RoomClassEncoder(raw_dataset)

        # [N x 3] 3D positions of rooms centered at scene center
        self.positions = RoomPositionEncoder(raw_dataset)

        # [N x F] latent shape features or flattened pc of rooms
        self.shapes = RoomShapeEncoder(raw_dataset, encoder_net=None)

        # [N x N] upper triangular vector of room connectivity as adjacency matrix
        self.edges = RoomConnectionEncoder(raw_dataset)

        feat_encoders = [self.class_labels, self.positions, self.shapes, self.edges]
        dataset_collection = DatasetCollection(*feat_encoders)

        # Apply augmentations
        if isinstance(augmentations, str):
            augmentations = [augmentations]

        if "partial_patch" in augmentations:
            assert (
                augmentations[0] == "partial_patch"
            ), "Partial patch must be the first augmentation"

        for aug_type in augmentations:
            if aug_type == "rotation":
                dataset_collection = RotationAugmentation(
                    dataset_collection, fixed=False
                )
                print("Applying rotation augmentations")
            elif aug_type == "fixed_rotation":
                dataset_collection = RotationAugmentation(
                    dataset_collection, fixed=True
                )
                print("Applying fixed rotation augmentations")
            elif aug_type == "jitter":
                dataset_collection = Jitter(dataset_collection, dist=0.5)
                print("Applying jittering augmentations to room positions")
            elif aug_type == "partial_patch":
                dataset_collection = PartialPatch(
                    dataset_collection,
                    patch_bound=patch_bound,
                    min_size=min_size,
                    max_size=max_size,
                )
                print("Applying partial patch augmentations")
            else:
                raise RuntimeError(f"Unknown augmentation type: {aug_type}")

        self.encoded_dataset = SamplePointCloud(
            dataset_collection, render_size=(256, 256), render_bound=render_bound
        )

        print("Total # data: ", len(self.encoded_dataset))

        self._grid_size = [256, 256, 1]

        # TODO(hlim): Should follow XZY coordinate system?
        self.coor_ranges = [-MAX_BOUND, -MAX_BOUND, -1.0] + [MAX_BOUND, MAX_BOUND, 1.0]
        self.voxel_sizes = [
            abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0],
            abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
            abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2],
        ]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        counts_batch = [bi[2] for bi in data]
        output_orig_batch = [bi[3] for bi in data]
        return voxel_batch, output_batch, counts_batch, output_orig_batch

    def __getitem__(self, idx):
        encoded_data = self.encoded_dataset[idx]
        # [256 x 256] np.uint8 labeled point cloud map
        output = encoded_data["map_img"][..., np.newaxis]

        # if self.random_flips:
        #     if np.random.randint(2):
        #         output = np.flip(output, axis=0)  # vertical flip
        #     if np.random.randint(2):
        #         output = np.flip(output, axis=1)  # horizontal flip

        mask = (output != EMPTY_CLASS_NUM).astype(np.float32)
        voxel_input = mask.copy()
        voxel_input = np.stack([mask] * self._num_frames, axis=0)
        counts = mask.copy()

        if "map_img_orig" in encoded_data:
            output_orig = encoded_data["map_img_orig"][..., np.newaxis]
        else:
            output_orig = None

        return voxel_input, output, counts, output_orig


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # NOTE(hlim): Jitter should not be used
    augmentations = ["partial_patch", "fixed_rotation"]
    binary_counts = True

    train_ds = ThreeDFrontDataset(
        directory="../ThreedFront/data/room_graphs_compact/",
        split="train",
        augmentations=augmentations,
        random_flips=True,
        binary_counts=binary_counts,
    )

    encoded_data = train_ds.encoded_dataset[10]
    print(encoded_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # partial image
    ax1.imshow(
        cv2.flip(encoded_data["map_img"], 0), cmap="gray", interpolation="nearest"
    )
    ax1.set_title("Partial View")
    ax1.axis("off")

    # original image
    if "map_img_orig" in encoded_data:
        ax2.imshow(
            cv2.flip(encoded_data["map_img_orig"], 0),
            cmap="gray",
            interpolation="nearest",
        )
        ax2.set_title("Original View")
        ax2.axis("off")
    else:
        print("`map_img_orig` does not exist")
        ax2.set_visible(False)

    print("NumPy array shape:", encoded_data["map_img"].shape)
    plt.tight_layout()
    plt.show()
