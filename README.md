# Supervised-Grasping

This repository contains the means to train a model with Action Image inputs. From there, grasps can be sampled and run through the netowrk to determine the grasp that has the most possibilty of success.

## Requirements

[Tensor2Robot](https://github.com/google-research/tensor2robot) must be cloned and placed into this directory, along with the python modules in requirements.txt. They can be installed with:
```
pip install -r tensor2robot/requirements.txt
```

## Action Image Usage
```
python action_image.py [-h] [--target] [--balance] [--data-dir]

optional arguments:
  -h, --help           show this help message and exit
  --target              determines whether to include a target action image
  --balance           determines whether to balance positive and negative examples
  --data-dir           the directory to find data

  ```
The expected format for data in this repository is a directory with sub directories that contain the actual data. Each sub directory has a csv within called 'data.csv' to navigate the data.

## Training the Model
```
python run_t2r_trainer.py --logtostderr --gin_configs="config/train_grasping.gin"
```
The path to your data should be specified in config/train_grasping.gin.
