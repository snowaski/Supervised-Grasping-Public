# Supervised-Grasping

This strategy uses action images and a Resnet model to optimize grasping chess pieces.

## Action Image Usage
```
python action_images/action_image.py [-h] [--mode {target,no-target}] [--balance]
                                     [--data-dir DATA_DIR]

optional arguments:
  -h, --help                         show this help message and exit
  --mode {target,no-target}          determines whether to create target images
  --balance                          determines whether to balance positive and negative examples
  --data-dir DATA_DIR                the directory to find data
  ```
The expected format for data in this repository is a directory with sub directories that contain the actual data. Each sub directory has a csv within called 'data.csv' to navigate the data. Example data can be found here:

## Running the Model
```
python -m run_t2r_trainer.py --logtostdrr --gin_configs="config/train_grasping.gin"
```
