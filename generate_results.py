import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from threed_front.room_evaluation import ThreedFrontRoomResults
from tqdm import tqdm

from layers.Voxel_Level.Con_Diffusion import Con_Diffusion
from layers.Voxel_Level.Gen_Diffusion import Diffusion
from utils.hd_utils import (
    RENDERED_DIR,
    generate_layouts,
    get_encoded_dataset_for_ssc,
    label_img_to_point_clouds,
    set_random_seed,
)

"""
# For 3DFront
> python3 generate_results.py con result_cond_diffusion_3dfront_tmp/epoch1299.tar --dataset_dir ../ThreedFront/data/room_graphs_compact/  --output_directory parsed
# For MP3D
> python3 generate_results.py con result_cond_diffusion_3dfront_tmp/epoch1299.tar --dataset_dir ../ThreedFront/data/mp3d_repartitioned_dataset/ --dataset mp3d  --output_directory parsed_mp3d --experiment "spark_dsg"
"""


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate results from a trained model"
    )

    parser.add_argument("mode", default="con", choices="gen, con")
    parser.add_argument(
        "model_file",
        help="Path to saved model weights. "
        "This should be in a <experiment_tag> dir containing config.yaml",
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="3dfront", choices=["carla", "3dfront", "mp3d"]
    )
    parser.add_argument(
        "--output_directory",
        default=None,
        help="Path to the output directory (default: RENDERED_DIR/<experiment_tag>)",
    )
    parser.add_argument("--recon_loss", default=False)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--diffusion_dim", type=int, default=32)
    parser.add_argument(
        "--num_samples",
        default=1024,
        type=int,
        help="Number of samples to generate (default: 1024)",
    )
    parser.add_argument(
        "--experiment",
        default="partial_patch",
        choices=["unconditioned", "partial_patch", "spark_dsg"],
        help="Experiment name",
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for sampling (default: 8)"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gpu", default=1, type=int, help="GPU device ID to use (default: 1)"
    )

    args = parser.parse_args(argv)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.device_count() > args.gpu else "cpu"
    )
    print(f"Using device: {device}")
    torch.cuda.set_device(args.gpu)
    set_random_seed(args.seed)

    # output directory
    result_dir = os.path.dirname(args.model_file)
    if args.output_directory is None:
        experiment_tag = os.path.basename(result_dir)
        output_dir = os.path.join(RENDERED_DIR, experiment_tag)
    else:
        output_dir = args.output_directory
    if os.path.exists(output_dir) and os.listdir(output_dir):
        input(
            "Output directory ({}) already exists and is non-empty. "
            "Press any key to continue.".format(output_dir)
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving '{args.experiment}' experimental results to {output_dir}")

    # diffusion net
    args.num_classes = 18
    augmentations = ["partial_patch"]

    # `config` will be used as a flag to check `is_dsg_input`
    config = {}
    if args.dataset == "3dfront":
        pickled_dataset_path = os.path.join(args.dataset_dir, "room_graph_dataset.pkl")
    elif args.dataset == "mp3d":
        pickled_dataset_path = os.path.join(args.dataset_dir, "spark_dsg_dataset.pkl")
        config["data"] = {}
        config["data"]["dsg_dataset_path"] = os.path.realpath(pickled_dataset_path)
    else:
        raise NotImplementedError("Not implemented!")
    print("\033[1;31m", args.dataset, "\033[0m")
    print("\033[1;32m", pickled_dataset_path, "\033[0m")

    raw_data = pickle.load(open(pickled_dataset_path, "rb"))
    raw_train_dataset, raw_test_dataset = (
        raw_data["train_dataset"],
        raw_data["test_dataset"],
    )

    augmentations = []
    sampling_kwargs = {}
    if args.experiment == "partial_patch":
        augmentations = ["partial_patch"]
        sampled_patches_path = "test_sampled/partial_patches.npy"
        if os.path.exists(sampled_patches_path):
            sampled_patches = np.load(sampled_patches_path)
            print("\033[1;32mSampled patches found.\033[0m")
        else:
            sampled_patches = None
            input(
                "Patch file does not exist. "
                "Press any key to continue with random sampled patches."
            )
        sampling_kwargs = {
            "patch_bound": ((-12.0, -12.0), (12.0, 12.0)),
            # "min_size": (2.0, 2.0),
            "sampled_patches": sampled_patches,
        }
    elif args.experiment == "spark_dsg":
        augmentations = ["partial_patch"]
        full_patch = np.array([[-12, -12], [12, 12]], dtype=np.float32)
        sampling_kwargs = {
            "patch_bound": ((-12.0, -12.0), (12.0, 12.0)),
            "sampled_patches": [full_patch] * len(raw_test_dataset),
        }

    encoded_test_dataset = get_encoded_dataset_for_ssc(
        raw_test_dataset, augmentations=augmentations, **sampling_kwargs
    )

    ############################################################
    # Load model
    ############################################################
    # # room label to color mapping
    # # It should be tuple!
    room_list = raw_train_dataset.room_types
    color_map_path = "../ThreedFront/data/color_map.json"
    room_colors_dict = json.load(open(color_map_path, "r"))
    room_class_to_color = [
        np.array(room_colors_dict[room_class]) * 255 for room_class in room_list
    ]

    # class_frequencies = train_ds.remap_frequencies_cartesian
    # comp_weights = get_class_weights(class_frequencies).to(torch.float32)

    comp_weights = torch.ones(18, dtype=torch.float32)
    comp_weights = comp_weights / comp_weights.sum()
    completion_criterion = torch.nn.CrossEntropyLoss(weight=comp_weights)

    if args.mode == "gen":
        network = Diffusion(args, completion_criterion).cuda()
    elif args.mode == "con":
        network = Con_Diffusion(args, completion_criterion).cuda()
    else:
        raise NotImplementedError("Other modes are not supported")

    checkpoint = torch.load(args.model_file)

    network.load_state_dict(checkpoint["model"])
    network.eval()

    ############################################################
    # Main loop
    ############################################################
    sampling_rule = "uniform"

    if args.dataset == "mp3d":
        test_index_path = os.path.join(args.dataset_dir, "test_sample_indices.txt")
        with open(test_index_path, "r") as f:
            sampled_indices = [int(line.strip()) for line in f.readlines()]
        print(f"Loaded {len(sampled_indices)} test indices from MP3D.")
        args.num_samples = len(sampled_indices)

    sampled_indices, layout_list = generate_layouts(
        network,
        encoded_test_dataset,
        args.num_samples,
        sampling_rule=sampling_rule,
        experiment=args.experiment,
        batch_size=args.batch_size,
        device=device,
    )

    # post-process
    results = ThreedFrontRoomResults(raw_train_dataset, raw_test_dataset, config)
    print(f"result render_size: {results.render_size}")
    print(f"result render_bound: {results.render_bound}")
    for i, voxel_label_img in enumerate(tqdm(layout_list)):
        result_i = label_img_to_point_clouds(voxel_label_img, -12.0, 12.0)
        if args.experiment == "unconditioned":
            results.add_result(result_i)  # no input from test dataset
        else:
            results.add_result(result_i, scene_idx=sampled_indices[i])

        image_path = os.path.join(output_dir, f"syn_{i}.png")
        results.render_projection(i, image_path, room_class_to_color)

    pickle.dump(results, open(os.path.join(output_dir, "results.pkl"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
