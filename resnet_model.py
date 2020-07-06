import gin
import numpy as np
import tensorflow as tf
from tensor2robot.models import abstract_model    
from tensor2robot.utils import tensorspec_utils as utils
import film_resnet_model as resnet_lib
from tensor2robot.layers import resnet

@gin.configurable
class GraspingModel(abstract_model.AbstractT2RModel):
    def __init__(self, embedding_loss_fn=tf.compat.v1.losses.log_loss):
        super(GraspingModel, self).__init__()
        self._embedding_loss_fn = embedding_loss_fn

    def get_feature_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['imgs/RGB'] = utils.ExtendedTensorSpec(shape=(128, 128, 3), dtype=tf.float32, name='rgb', data_format='jpeg')
        spec['imgs/Feature_RGB'] = utils.ExtendedTensorSpec(shape=(128, 128, 3), dtype=tf.float32, name='feature_rgb', data_format='jpeg')
        spec['imgs/Depth'] = utils.ExtendedTensorSpec(shape=(128, 128, 3), dtype=tf.float32, name='depth', data_format='jpeg')
        spec['imgs/Feature_depth'] = utils.ExtendedTensorSpec(shape=(128, 128, 3), dtype=tf.float32, name='feature_depth', data_format='jpeg')
        return spec

    def get_label_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['grasp_success_spec'] = utils.ExtendedTensorSpec(shape=(1),  dtype=tf.float32, name='grasp_success')
        return spec

    def inference_network_fn(self, features, labels, mode, config=None, params=None):
        model = resnet_lib.Model(resnet_size=50,
                                 bottleneck=False,
                                 num_classes=1,
                                 num_filters=7, 
                                 kernel_size=7,
                                 conv_stride=2,
                                 first_pool_size=3,
                                 first_pool_stride=2,
                                 block_sizes=[2, 2],
                                 block_strides=[1,1],
                                 weight_decay=None)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        output = model(features.imgs, is_training)

        # assert(False), f'shape: {img}'

        return {'grasp_success': output}

    def model_train_fn(self,
                        features,
                        labels,
                        inference_outputs,
                        mode,
                        config = None,
                        params = None):
        # assert(False), f'This: {inference_outputs["grasp_success"]}'
        loss = self._embedding_loss_fn(labels.grasp_success_spec, inference_outputs["grasp_success"])
        return loss