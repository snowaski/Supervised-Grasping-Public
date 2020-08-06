# Supervised-Grasping

This strategy uses action images and a Resnet model to optimize grasping chess pieces.

## Requirements

[Tensor2Robot](https://github.com/google-research/tensor2robot) must be cloned and placed into this directory.

## Action Image Usage
```
python action_images/action_image.py [-h] [--mode {target,no-target}] [--balance]
                                     [--data-dir DATA_DIR]

optional arguments:
  -h, --help                         show this help message and exit
  --mode {target,no-target}          determines whether to create target images
  --balance                          balances positive and negative examples
  --data-dir DATA_DIR                the directory to find data
  ```
The expected format for data in this repository is a directory with sub directories that contain the actual data. Each sub directory has a csv within called 'data.csv' to navigate the data. Example data can be found [here](https://drive.google.com/drive/folders/1zBJdu87r0Avqv1P0hISI9w004spJdMN1?usp=sharing).

## Training the Model
```
python run_t2r_trainer.py --logtostderr --gin_configs="config/train_grasping.gin"
```
