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
        # inference_outputs['grasp_success'] = tf.cast(tf.math.round(inference_outputs['grasp_success']), tf.int64)
        inference_outputs['grasp_success'] = tf.cast(
            inference_outputs['grasp_success'], tf.int64)
        return loss, inference_outputs

    def add_summaries(self,
                      features,
                      labels,
                      inference_outputs,
                      train_loss,
                      train_outputs,
                      mode,
                      config=None,
                      params=None):
        """Add summaries to the graph.
        Having a central place to add all summaries to the graph is helpful in order
        to compose models. For example, if an inference_network_fn is used within
        a while loop no summaries can be added. This function will allow to add
        summaries after the while loop has been processed.
        Args:
        features: This is the first item returned from the input_fn and parsed by
            tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
            requirements of the self.get_feature_specification.
        labels: This is the second item returned from the input_fn and parsed by
            tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
            requirements of the self.get_feature_specification.
        inference_outputs: A dict containing the output tensors of
            model_inference_fn.
        train_loss: The final loss from model_train_fn.
        train_outputs: A dict containing the output tensors (dict) of
            model_train_fn.
        mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
        config: (Optional tf.estimator.RunConfig or contrib_tpu.RunConfig) Will
            receive what is passed to Estimator in config parameter, or the default
            config (tf.estimator.RunConfig). Allows updating things in your model_fn
            based on  configuration such as num_ps_replicas, or model_dir.
        params: An optional dict of hyper parameters that will be passed into
            input_fn and model_fn. Keys are names of parameters, values are basic
            python types. There are reserved keys for TPUEstimator, including
            'batch_size'.
        """
        if not self.use_summaries(params):
            return

        tf.summary.histogram('predicted', train_outputs['grasp_success'])

        acc = tf.contrib.metrics.accuracy(train_outputs["grasp_success"],
                                          labels["grasp_success_spec"])

        tf.summary.scalar('accuracy', acc)

        tf.summary.image('RGB', features['imgs/RGB'], max_outputs=8)
        tf.summary.image('Feature_RGB',
                         features['imgs/Feature_RGB'],
                         max_outputs=8)
        tf.summary.image('Depth', features['imgs/Depth'])
        tf.summary.image('Feature_Depth', features['imgs/Feature_Depth'])

    def model_eval_fn(self,
                      features,
                      labels,
                      inference_outputs,
                      train_loss,
                      train_outputs,
                      mode,
                      config=None,
                      params=None):
        eval_mse = tf.metrics.mean_squared_error(
            labels=labels['grasp_success_spec'],
            predictions=inference_outputs['grasp_success'],
            name='eval_mse')

        predictions_rounded = tf.round(inference_outputs['grasp_success'])

        eval_precision = tf.metrics.precision(
            labels=labels['grasp_success_spec'],
            predictions=predictions_rounded,
            name='eval_precision')

        eval_accuracy = tf.metrics.accuracy(
            labels=labels['grasp_success_spec'],
            predictions=predictions_rounded,
            name='eval_accuracy')

        eval_recall = tf.metrics.recall(labels=labels['grasp_success_spec'],
                                        predictions=predictions_rounded,
                                        name='eval_recall')

        # eval_f1 = eval_precision / eval_recall

        metric_fn = {
            'eval_mse': eval_mse,
            'eval_precision': eval_precision,
            'eval_accuracy': eval_accuracy,
            'eval_recall': eval_recall
            # 'eval_f1': eval_f1
        }

        return metric_fn
