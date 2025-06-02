import os
from typing import List

import cv2
import numpy as np
import torch
import yaml
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import LineString
from threed_front.room_datasets.room_graph_dataset import (
    CachedRoomGraphDataset,
    Jitter,
    PartialPatch,
    Permutation,
    RotationAugmentation,
    get_basic_encoding,
)
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from threed_front.room_evaluation.room_graph_result import RoomPointCloudResult

from datasets.threed_dataset import SamplePointCloud

# Define project paths
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, "data")
RESULT_DIR = os.path.join(PROJ_DIR, "output/train_log")

RENDERED_DIR = os.path.join(PROJ_DIR, "output/rendered")

EMPTY_CLASS_NUM_IN_3DFRONT = 17


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def generate_layouts(
    network,
    encoded_dataset,
    num_syn_scenes,
    sampling_rule="random",
    experiment="unconditioned",
    batch_size=16,
    device="cpu",
):
    # Sample floor layout
    if sampling_rule == "random":
        sampled_indices = np.random.choice(
            len(encoded_dataset), num_syn_scenes
        ).tolist()
    elif sampling_rule == "uniform":
        sampled_indices = np.arange(len(encoded_dataset)).tolist() * (
            num_syn_scenes // len(encoded_dataset)
        )
        sampled_indices += np.random.choice(
            len(encoded_dataset), num_syn_scenes - len(sampled_indices)
        ).tolist()
    elif isinstance(sampling_rule, list):
        assert (
            len(sampling_rule) == num_syn_scenes
        ), "Number of sample indices does not match num_syn_scenes."
        sampled_indices = sampling_rule
    else:
        raise ValueError(f"Sampling rule {sampling_rule} not implemented.")

    if experiment == "partial_patch":
        assert (
            "is_full" in encoded_dataset[0]
        ), "Conditioned sampling requires is_full feature."

    # Generate layouts
    network.to(device)
    network.eval()
    layout_list = []
    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i : min(i + batch_size, num_syn_scenes)]

        if experiment == "unconditioned":
            raise NotImplementedError("Not implemented!")
        else:
            voxel_inputs = []
            for idx in scene_indices:
                encoded_data = encoded_dataset[idx]
                # [256 x 256] np.uint8 labeled point cloud map
                output = encoded_data["map_img"][..., np.newaxis]

                EMPTY_CLASS_NUM_IN_3DFRONT = 17
                mask = (output != EMPTY_CLASS_NUM_IN_3DFRONT).astype(np.float32)
                voxel_input = mask.copy()
                voxel_inputs.append(voxel_input)

        # Convert to tensor
        voxel_input = torch.from_numpy(np.asarray(voxel_inputs)).float().to(device)
        recon = network.sample(voxel_input)

        if recon.ndim == 4:  # (B, H, W, D)
            for j in range(recon.size(0)):
                layout_list.append(recon[j].cpu().numpy())
        else:
            layout_list.append(recon.cpu().numpy())
    return sampled_indices, layout_list


def get_room_pointclouds(
    room_positions: torch.Tensor,
    latent_features: torch.Tensor,
    shape_decoder: torch.nn.Module,
    device=None,
) -> List[torch.Tensor]:
    """Convert the latent features of rooms to point clouds and shift the results by positions."""
    num_rooms = room_positions.shape[0]
    assert latent_features.shape[0] == num_rooms
    if device is not None:
        shape_decoder.to(device)
    else:
        try:
            device = next(shape_decoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    point_clouds = shape_decoder(latent_features.to(device))
    room_pcs = []
    for i in range(num_rooms):
        reconstructed_pc = point_clouds[i].cpu() + room_positions[i, [0, 2]].cpu()
        room_pcs.append(reconstructed_pc)
    return room_pcs


def get_routes(manager, routing, solution):
    """Get vehicle routes from a OR-Tools solution and store them in an array."""
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def solve_tsp(dist_matrix: np.ndarray):
    """Solve the TSP problem using OR-Tools."""
    # problem setup
    manager = pywrapcp.RoutingIndexManager(
        dist_matrix.shape[0], 1, 0
    )  # num_nodes, num_vehicles, depot
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # solver setup
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # solution
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return get_routes(manager, routing, solution)
    else:
        print("No solution found! Solver status: ", routing.status())
        return None


def order_points(points, sort_option="angle", max_dist=None) -> np.ndarray:
    """
    Order points in a 2D polygon in counter-clockwise order.
    Args:
        points: Nx2 array of 2D points.
        sort_option: Sorting option for the points ('angle' or 'tsp').
        max_dist: Maximum distance between consecutive points for orered point filtering.
    Returns:
        Nx2 numpy array of the input 2D points.
    """
    if sort_option == "tsp":
        distance_matrix = squareform(pdist(points, "euclidean"))
        # find points to be ordered
        if max_dist is None:
            pt_bool = np.ones(distance_matrix.shape[0], dtype=bool)
        else:
            pt_bool = (
                distance_matrix + (1 + max_dist) * np.eye(distance_matrix.shape[0])
            ).min(
                axis=0
            ) < max_dist  # True if there is another point within max_dist
        sorted_indices = solve_tsp(
            (distance_matrix[pt_bool, :][:, pt_bool] * 1000).astype(int)
        )
        if sorted_indices is None:  # use angle sorting as fallback
            print("TSP solver failed! Fallback to angle sorting.")
            return order_points(points, max_dist=max_dist, sort_option="angle")
        else:
            return np.array(points[pt_bool, :][sorted_indices[0]])

    elif sort_option == "angle":
        centroid = np.mean(points, axis=0)

        def angle_from_centroid_func(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        sorted_points = sorted(points, key=angle_from_centroid_func)

        if max_dist is None:
            return np.asarray(sorted_points)

        filtered_points = []
        for i in range(len(sorted_points)):
            if (
                np.linalg.norm(sorted_points[i - 1] - sorted_points[i]) < max_dist
                or np.linalg.norm(sorted_points[i - 2] - sorted_points[i - 1])
                < max_dist
            ):
                filtered_points.append(sorted_points[i - 1])
        return np.asarray(filtered_points)

    else:
        raise ValueError(f"Unsupported sort option: {sort_option}")


def get_encoded_dataset_for_ssc(
    raw_dataset: CachedRoomGraphDataset,
    augmentations=[],
    encoder_net=None,
    max_length=None,
    **sampling_kwargs,
):
    """Create an encoded dataset from the raw dataset."""
    if isinstance(augmentations, str):
        augmentations = [augmentations]

    if "partial_patch" in augmentations:
        assert (
            augmentations[0] == "partial_patch"
        ), "Partial patch must be the first augmentation"

    if encoder_net is not None:
        encoder_net.eval()
        encoder_net.requires_grad_(False)
        if ("rotation" in augmentations) or ("fixed_rotation" in augmentations):
            raise RuntimeError(
                "Rotation augmentations not supported with encoded features"
            )

    # Get basic encoding
    dataset_collection = get_basic_encoding(raw_dataset, encoder_net)
    print("Input properties: ", dataset_collection.feature_names)

    # Apply augmentations
    perm_keys = [f for f in dataset_collection.feature_names if f != "edges"]
    for aug_type in augmentations:
        if aug_type == "rotation":
            dataset_collection = RotationAugmentation(dataset_collection, fixed=False)
            print("Applying rotation augmentations")
        elif aug_type == "fixed_rotation":
            dataset_collection = RotationAugmentation(dataset_collection, fixed=True)
            print("Applying fixed rotation augmentations")
        elif aug_type == "jitter":
            dataset_collection = Jitter(dataset_collection, dist=0.5)
            print("Applying jittering augmentations to room positions")
        elif aug_type == "perm":
            dataset_collection = Permutation(dataset_collection, perm_keys)
            print("Applying permutation augmentations")
        elif aug_type == "partial_patch":
            dataset_collection = PartialPatch(
                dataset_collection,
                patch_bound=sampling_kwargs.get("patch_bound", None),
                min_size=sampling_kwargs.get("min_size", (2.0, 2.0)),
                max_size=sampling_kwargs.get("max_size", None),
                sampled_patches=sampling_kwargs.get("sampled_patches", None),
            )
            print("\033[1;32mApplying partial patch augmentations\033[0m")
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    # NOTE(hlim): Unlike Siyi's hierarchical graph prediction,
    # SSC uses 256 x 256 x 1 image format as an input

    dataset_collection = SamplePointCloud(
        dataset_collection, render_size=(256, 256), render_bound=[-12.0, 12.0]
    )

    return dataset_collection


def interpolate_contour(contour, num_points=256):
    """Uniformly interpolate a closed contour to have exactly `num_points` points."""
    if len(contour) < 3:
        return None
    # Ensure closed loop
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    line = LineString(contour)
    interpolated = [
        line.interpolate(dist, normalized=True).coords[0]
        for dist in np.linspace(0, 1, num_points, endpoint=False)
    ]
    return np.array(interpolated, dtype=np.float32)


def label_img_to_point_clouds(voxel_label_img, min_bound, max_bound):
    if min_bound > max_bound:
        raise RuntimeError("`min_bound` should be smaller than `max_bound`.")

    H, W = voxel_label_img.shape[:2]
    canvas_size = np.array([W, H])  # (width, height)
    coord_min = np.array([min_bound, min_bound], dtype=np.float32)
    coord_max = np.array([max_bound, max_bound], dtype=np.float32)
    label_map = voxel_label_img.squeeze()
    unique_classes = np.unique(label_map)

    point_clouds = []
    class_labels = []
    positions = []
    for class_id in unique_classes:
        if class_id == EMPTY_CLASS_NUM_IN_3DFRONT:
            continue

        binary_mask = (label_map == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.squeeze(1)
            if contour.ndim != 2:
                continue

            # Step 1: Interpolate contour to exactly 256 points
            interpolated = interpolate_contour(contour, num_points=256)
            if interpolated is None:
                continue

            # Step 2: Convert pixel coordinates to real-world
            pixel_values = interpolated.astype(np.float32)
            real_scale_values = (pixel_values / (canvas_size - 1)) * (
                coord_max - coord_min
            ) + coord_min

            # Step 3: Append
            point_clouds.append(real_scale_values.astype(np.float32))
            class_labels.append(class_id)
            center_xz = np.mean(real_scale_values, axis=0)
            center_xyz = np.array([center_xz[0], 0.0, center_xz[1]], dtype=np.float32)
            positions.append(center_xyz)

    # Convert to arrays
    class_labels = np.array(class_labels)
    positions = np.array(positions)

    # Construct RoomPointCloudResult
    result_i = RoomPointCloudResult(
        class_labels=class_labels,
        positions=positions,
        point_clouds=point_clouds,
        edge_tuples=[],
    )

    return result_i
