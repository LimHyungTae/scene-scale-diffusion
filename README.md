# Diffusion Probabilistic Models for Scene-Scale 3D Categorical Data


## In the 3DFront dataset

1. Train VQVAE

```
python3 SSC_train.py --mode l_vae --vq_size 100 --l_size 16162 --init_size 32 --l_attention True --log_path ./result_vqvae_3dfront --dataset_dir ../ThreedFront/data/room_graphs_compact/ --dataset 3dfront --epoch 200 --batch_size 6 --num_workers 16
```

1-1. How to resume the training

```
python3 SSC_train.py --mode l_vae --vq_size 100 --l_size 16162 --init_size 32 --l_attention True --log_path ./result_vqvae_3dfront \
--dataset_dir ../ThreedFront/data/room_graphs_compact/ --dataset 3dfront --epoch 100 \
--batch_size 8 --num_workers 8 --resume True --resume_path result_vqvae_3dfront/epoch29.tar

```

2. Train Diffusion models


```
python3 SSC_train.py --mode l_gen --vq_size 100 --l_size 16162 --init_size 32 --l_attention True --log_path ./result_latent_diffusion \
--dataset_dir ../ThreedFront/data/room_graphs_compact/ --dataset 3dfront --epoch 50 \
--batch_size 16 --num_workers 8 --vqvae_path ./result_vqvae_3dfront/epoch99.tar
```

---

<img src=https://user-images.githubusercontent.com/65997635/210452550-2c7c7c6d-7260-43ce-b4b6-18d3f15fccde.png width="480"
  height="400">

Comparison of object-scale and scene scale generation (ours). Our result includes multiple objects in a generated scene,
while the object-scale generation crafts one object at a time. (a) is obtained by [Point-E](https://github.com/openai/point-e)

## Abstract
In this paper, we learn a diffusion model to generate 3D data on a scene-scale. Specifically, our model crafts a 3D scene consisting of multiple objects, while recent diffusion research has focused on a single object. To realize our goal, we represent a scene with discrete class labels, i.e., categorical distribution, to assign multiple objects into semantic categories. Thus, we extend discrete diffusion models to learn scene-scale categorical distributions. In addition, we validate that a latent diffusion model can reduce computation costs for training and deploying. To the best of our knowledge, our work is the first to apply discrete and latent diffusion for 3D categorical data on a scene-scale. We further propose to perform semantic scene completion (SSC) by learning a conditional distribution using our diffusion model, where the condition is a partial observation in a sparse point cloud. In experiments, we empirically show that our diffusion models not only generate reasonable scenes, but also perform the scene completion task better than a discriminative model.


## Instructions
### Dataset
: We use [CarlaSC](https://umich-curly.github.io/CarlaSC.github.io/download/) cartesian dataset.

### Training
: There are some argparse in 'SSC_train.py'.

    python SSC_train.py

- For **multi-GPU** : --distribution True
- For **Discrete Diffusion Model** : --mode gen/con/vis
- For **Latent Diffusion Model** : --mode l_vae/l_gen --l_size 882/16162/32322 --init_size 32 --l_attention True --vq_size 100

Example for training l_gen mode

    python SSC_train.py --mode l_gen --vq_size 100 --l_size 32322 --init_size 32 --l_attention True --log_path ./result --vqvae_path ./lst_stage.tar


### Visualization
: We save the result to a txt file using the `utils/table.py/visulization` function.
If you use open3d, you will be able to easily visualize it.

## Result
### 3D Scene Generation
![image](https://github.com/zoomin-lee/scene-scale-diffusion/blob/main/images/3D_scene_generation.png?raw=true)

### Semantic Scene Completion
![image](https://github.com/zoomin-lee/scene-scale-diffusion/blob/main/images/table4.PNG?raw=true)


![image](https://github.com/zoomin-lee/scene-scale-diffusion/blob/main/images/semantic_scene_completion.png?raw=true)


## Acknowledgments
This project is based on the following codebase.
- [Multinomial Diffusion](https://github.com/ehoogeboom/multinomial_diffusion/tree/9d907a60536ad793efd6d2a6067b3c3d6ba9fce7)
- [MotionSC](https://github.com/UMich-CURLY/3DMapping)
- [Cylinder3D](https://github.com/xinge008/Cylinder3D)
