import os
import pickle

import numpy as np
import yaml
from torch.utils.data import Dataset


class ThreeDFrontDataset(Dataset):
    """ThreeDFrontDataset for baseline comparison"""

    def __init__(
        self,
        directory,
        split,
        voxelize_input=True,
        binary_counts=True,
        random_flips=False,
        remap=True,
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

        self.get_gt = get_gt
        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = directory
        self._num_frames = num_frames
        self.random_flips = random_flips
        self.remap = remap
        self.transform_pose = transform_pose
        self.sparse_output = True

        pickled_dataset_path = os.path.join(directory, "room_graph_dataset.pkl")

        data = pickle.load(open(pickled_dataset_path, "rb"))
        self.raw_data = data[f"{split}_dataset"]
        self.num_data = len(self.raw_data)

        # For easy visualization to 256x256x3 image
        self._grid_size = [256, 256, 1]
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = [np.uint32(dim) for dim in self._grid_size]

        # TODO(hlim): Should follow XZY coordinate system?
        self.coor_ranges = [-18.0, -18.0, -1.0] + [18.0, 18.0, 1.0]
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
        return self.num_data

    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        counts_batch = [bi[2] for bi in data]
        return voxel_batch, output_batch, counts_batch

    def points_to_voxels(self, voxel_grid, points):
        voxels = np.floor((points - self.min_bound) / self.voxel_sizes).astype(np.int32)
        maxes = np.reshape(self.grid_dims - 1, (1, 3))
        voxels = np.clip(voxels, 0, maxes).astype(np.int32)
        if self.binary_counts:
            voxel_grid[0, voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1
        else:
            unique_voxels, counts = np.unique(voxels, return_counts=True, axis=0)
            voxel_grid[
                0, unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]
            ] += counts
        return voxel_grid

    def __getitem__(self, idx):
        scene = self.raw_data[idx]
        points = scene.points  # (N, 3)
        labels = scene.occupancy  # (256, 256, 1) or flattened

        voxel_input = np.zeros((1, *self._grid_size), dtype=np.float32)
        voxel_input = self.points_to_voxels(voxel_input, points)

        output = labels.astype(np.uint8)
        counts = voxel_input[0].copy()

        if self.voxelize_input and self.random_flips:
            if np.random.randint(2):
                output = np.flip(output, axis=0)
                counts = np.flip(counts, axis=0)
                voxel_input = np.flip(voxel_input, axis=1)
            if np.random.randint(2):
                output = np.flip(output, axis=1)
                counts = np.flip(counts, axis=1)
                voxel_input = np.flip(voxel_input, axis=2)

        if self.remap:
            output = self.LABELS_REMAP[output].astype(np.uint8)

        return voxel_input, output, counts
