import gin
import numpy as np
import tensorflow as tf
from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils as utils
import film_resnet_model as resnet_lib
from tensor2robot.layers import resnet
from tensor2robot.preprocessors import image_transformations, abstract_preprocessor


@gin.configurable
class GraspingPreprocessor(abstract_preprocessor.AbstractPreprocessor):
    def _preprocess_fn(self, features, labels, mode):
        # features = image_transformations.ApplyPhotometricImageDistortions(features, random_brightness=True, random_hue=True, random_contrast=True)
        features.imgs.RGB = tf.cast(features.imgs.RGB, tf.float32)
        features.imgs.Feature_RGB = tf.cast(features.imgs.Feature_RGB,
                                            tf.float32)
        features.imgs.Depth = tf.map_fn(lambda x: tf.ensure_shape(
            tf.io.parse_tensor(x, out_type=tf.float32), (128, 128, 3)),
                                        features.imgs.Depth,
                                        dtype=tf.float32)
        features.imgs.Feature_Depth = tf.map_fn(lambda x: tf.ensure_shape(
            tf.io.parse_tensor(x, out_type=tf.float32), (128, 128, 3)),
                                                features.imgs.Feature_Depth,
                                                dtype=tf.float32)
        return features, labels

    def get_in_feature_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['imgs/RGB'] = utils.ExtendedTensorSpec(shape=(128, 128, 3),
                                                    dtype=tf.uint8,
                                                    name='rgb',
                                                    data_format='jpeg')
        spec['imgs/Feature_RGB'] = utils.ExtendedTensorSpec(shape=(128, 128,
                                                                   3),
                                                            dtype=tf.uint8,
                                                            name='feature_rgb',
                                                            data_format='jpeg')
        spec['imgs/Depth'] = utils.ExtendedTensorSpec(shape=(),
                                                      dtype=tf.string,
                                                      name='depth')
        spec['imgs/Feature_Depth'] = utils.ExtendedTensorSpec(
            shape=(), dtype=tf.string, name='feature_depth')
        return spec

    def get_in_label_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['grasp_success_spec'] = utils.ExtendedTensorSpec(
            shape=(1), dtype=tf.int64, name='grasp_success')
        return spec

    def get_out_feature_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['imgs/RGB'] = utils.ExtendedTensorSpec(shape=(128, 128, 3),
                                                    dtype=tf.float32,
                                                    name='rgb')
        spec['imgs/Feature_RGB'] = utils.ExtendedTensorSpec(shape=(128, 128,
                                                                   3),
                                                            dtype=tf.float32,
                                                            name='feature_rgb',
                                                            data_format='jpeg')
        spec['imgs/Depth'] = utils.ExtendedTensorSpec(shape=(128, 128, 3),
                                                      dtype=tf.float32,
                                                      name='depth')
        spec['imgs/Feature_Depth'] = utils.ExtendedTensorSpec(
            shape=(128, 128, 3), dtype=tf.float32, name='feature_depth')
        return spec

    def get_out_label_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['grasp_success_spec'] = utils.ExtendedTensorSpec(
            shape=(1), dtype=tf.int64, name='grasp_success')
        return spec


@gin.configurable
class GraspingModel(abstract_model.AbstractT2RModel):
    def __init__(self, embedding_loss_fn=tf.compat.v1.losses.log_loss):
        super(GraspingModel, self).__init__()
        self._embedding_loss_fn = embedding_loss_fn

    def get_feature_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['imgs/RGB'] = utils.ExtendedTensorSpec(shape=(128, 128, 3),
                                                    dtype=tf.float32,
                                                    name='rgb')
        spec['imgs/Feature_RGB'] = utils.ExtendedTensorSpec(shape=(128, 128,
                                                                   3),
                                                            dtype=tf.float32,
                                                            name='feature_rgb',
                                                            data_format='jpeg')
        spec['imgs/Depth'] = utils.ExtendedTensorSpec(shape=(128, 128, 3),
                                                      dtype=tf.float32,
                                                      name='depth')
        spec['imgs/Feature_Depth'] = utils.ExtendedTensorSpec(
            shape=(128, 128, 3), dtype=tf.float32, name='feature_depth')
        return spec

    def get_label_specification(self, mode):
        spec = utils.TensorSpecStruct()
        spec['grasp_success_spec'] = utils.ExtendedTensorSpec(
            shape=(1), dtype=tf.int64, name='grasp_success')
        return spec

    def inference_network_fn(self,
                             features,
                             labels,
                             mode,
                             config=None,
                             params=None):

        model = resnet_lib.Model(resnet_size=50,
                                 bottleneck=False,
                                 num_classes=1,
                                 num_filters=7,
                                 kernel_size=7,
                                 conv_stride=2,
                                 first_pool_size=3,
                                 first_pool_stride=2,
                                 block_sizes=[2, 2],
                                 block_strides=[1, 1],
                                 weight_decay=None)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        output = model(features.imgs, is_training)

        return {'grasp_success': output}

    @property
    def default_preprocessor_cls(self):
        return GraspingPreprocessor

    def model_train_fn(self,
                       features,
                       labels,
                       inference_outputs,
                       mode,
                       config=None,
                       params=None):
        loss = self._embedding_loss_fn(labels.grasp_success_spec,
                                       inference_outputs["grasp_success"])
        return loss
