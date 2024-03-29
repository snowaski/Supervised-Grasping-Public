from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensor2robot.utils import tensorspec_utils as utils
from typing import Callable, Optional

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16, )
ALLOWED_TYPES = (DEFAULT_DTYPE, ) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def _batch_norm(inputs: utils.TensorSpecStruct, training: bool,
                data_format: str) -> tf.layers.batch_normalization:
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)


def _fixed_padding(inputs: utils.TensorSpecStruct, kernel_size: int,
                   data_format: str) -> utils.TensorSpecStruct:
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                  Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def _conv2d_fixed_padding(inputs: utils.TensorSpecStruct, filters: int,
                          kernel_size: int, strides: int, data_format: str,
                          weight_decay: float) -> tf.layers.conv2d:
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size, data_format)

    if weight_decay is not None:
        weight_decay = contrib_layers.l2_regularizer(weight_decay)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=weight_decay,
        data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(
        inputs: utils.TensorSpecStruct,
        filters: int,
        training: bool,
        projection_shortcut: Callable,
        strides: int,
        data_format: str,
        weight_decay: float,
        film_gamma_beta: Optional[tf.Tensor] = None) -> utils.TensorSpecStruct:
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
      weight_decay: L2 weight decay.
      film_gamma_beta: [batch, 2*filters] Tensor corresponding to FiLM params.

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = _batch_norm(inputs=shortcut,
                               training=training,
                               data_format=data_format)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=strides,
                                   data_format=data_format,
                                   weight_decay=weight_decay)
    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)
    inputs = _batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(
        inputs: utils.TensorSpecStruct,
        filters: int,
        training: bool,
        projection_shortcut: Callable,
        strides: int,
        data_format: str,
        weight_decay: Optional[tf.Tensor] = None,
        film_gamma_beta: Optional[tf.Tensor] = None) -> utils.TensorSpecStruct:
    """A single block for ResNet v2, without a bottleneck.
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
      weight_decay: L2 weight decay.
      film_gamma_beta: [batch, 2*filters] Tensor corresponding to FiLM params.

    Returns:
      The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs
    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=strides,
                                   data_format=data_format,
                                   weight_decay=weight_decay)

    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)

    return inputs + shortcut


def _bottleneck_block_v1(
        inputs: utils.TensorSpecStruct,
        filters: int,
        training: bool,
        projection_shortcut: Callable,
        strides: int,
        data_format: str,
        weight_decay: Optional[tf.Tensor] = None,
        film_gamma_beta: Optional[tf.Tensor] = None) -> utils.TensorSpecStruct:
    """A single block for ResNet v1, with a bottleneck.
    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
      weight_decay: L2 weight regularizer.
      film_gamma_beta: [batch, 2*filters] Tensor corresponding to FiLM params.

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = _batch_norm(inputs=shortcut,
                               training=training,
                               data_format=data_format)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)
    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=strides,
                                   data_format=data_format,
                                   weight_decay=weight_decay)
    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=4 * filters,
                                   kernel_size=1,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)
    inputs = _batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v2(
        inputs: utils.TensorSpecStruct,
        filters: int,
        training: bool,
        projection_shortcut: Callable,
        strides: int,
        data_format: str,
        weight_decay: Optional[tf.Tensor] = None,
        film_gamma_beta: Optional[tf.Tensor] = None) -> utils.TensorSpecStruct:
    """A single block for ResNet v2, with a bottleneck.
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of:
      Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
      weight_decay: L2 weight regularizer.
      film_gamma_beta: [batch, 2*filters] Tensor corresponding to FiLM params.

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)

    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=filters,
                                   kernel_size=3,
                                   strides=strides,
                                   data_format=data_format,
                                   weight_decay=weight_decay)

    inputs = _batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = _conv2d_fixed_padding(inputs=inputs,
                                   filters=4 * filters,
                                   kernel_size=1,
                                   strides=1,
                                   data_format=data_format,
                                   weight_decay=weight_decay)

    return inputs + shortcut


def _block_layer(inputs: utils.TensorSpecStruct, filters: int,
                 bottleneck: bool, block_fn: Callable, blocks: int,
                 strides: int, training: bool, name: str, data_format: str,
                 weight_decay: Optional[tf.Tensor],
                 film_gamma_betas: tf.Tensor) -> utils.TensorSpecStruct:
    """Creates one layer of blocks for the ResNet model.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').
      weight_decay: L2 weight regularizer.
      film_gamma_betas: List of FiLM parameters for each ResNet block in this
        block layer. Parameters can be None or a tf.Tensor.

    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return _conv2d_fixed_padding(inputs=inputs,
                                     filters=filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     data_format=data_format,
                                     weight_decay=weight_decay)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format, weight_decay, None)

    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format,
                          weight_decay, None)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for building the Resnet Model."""
    def __init__(self,
                 resnet_size: int,
                 bottleneck: bool,
                 num_classes: int,
                 num_filters: int,
                 kernel_size: int,
                 conv_stride: int,
                 first_pool_size: int,
                 first_pool_stride: int,
                 block_sizes: list,
                 block_strides: list,
                 weight_decay: Optional[tf.Tensor],
                 resnet_version: int = DEFAULT_VERSION,
                 data_format=None,
                 dtype: str = DEFAULT_DTYPE):
        """Creates a model for classifying an image.
        Args:
          resnet_size: A single integer for the size of the ResNet model.
          bottleneck: Use regular blocks or bottleneck blocks.
          num_classes: The number of classes used as labels.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
          block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
          block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
          weight_decay: L2 weight regularizer.
          resnet_version: Integer representing which version of the ResNet network
            to use. See README for details. Valid values: [1, 2]
          data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
          dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.
            
        Raises:
          ValueError: if invalid version is selected.
        """
        self.resnet_size = resnet_size

        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda()
                           else 'channels_last')

        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.weight_decay = weight_decay
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def _custom_dtype_getter(
            self,
            getter: Callable,
            name: str,
            shape=None,
            dtype=DEFAULT_DTYPE,  # pylint: disable=keyword-arg-before-vararg
            *args,
            **kwargs) -> tf.Tensor:
        """Creates variables in fp32, then casts to fp16 if necessary.
        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.
        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.
        Args:
          getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
          name: The name of the variable to get.
          shape: The shape of the variable to get.
          dtype: The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32 variable,
            then cast to the appropriate dtype
          *args: Additional arguments to pass unmodified to getter.
          **kwargs: Additional keyword arguments to pass unmodified to getter.
          
        Returns:
          A variable which is cast to fp16 if necessary.
        """

        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.
        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.
        Returns:
          A variable scope for the model.
        """

        return tf.variable_scope('resnet_model',
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self,
                 inputs: utils.TensorSpecStruct,
                 training: bool,
                 include_target_img: bool,
                 include_height_map: bool,
                 include_action_imgs: bool,
                 film_generator_fn: Optional[Callable] = None,
                 film_generator_input: Optional[Callable] = None) -> tf.Tensor:
        """Add operations to classify a batch of input images.
        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.
          include_target_img: Whether or not the model has a target image tower.
          include_height_map: Whether or not the model has a height map tower.
          include_action_imgs: Whether or not the model has feature action image towers.
          film_generator_fn: Callable that takes in a list of lists.
          film_generator_input: Tensor to be passed into film_generator_fn.

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        # assert(False), f'{inputs} {include_action_imgs}'
        filters = [4, [256, 32, 32]]
        if include_target_img:
            filters.append(4)
        if include_height_map:
            filters.append([256, 32, 32])
        if include_action_imgs:
            filters += [8, 8]
        imgs = inputs
        inputs = [inputs.RGB, inputs.Depth]
        if include_target_img:
            inputs.append(imgs.Target)
        if include_height_map:
            inputs.append(imgs.Height_Map)
        if include_action_imgs:
            inputs.append(imgs.Feature_RGB)
            inputs.append(imgs.Feature_Depth)
        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                for i, img in enumerate(inputs):
                    inputs[i] = tf.transpose(a=img, perm=[0, 3, 1, 2])
            imgs = []
            for i, f in enumerate(filters):
                if type(f) is list:
                    f = f[0]
                img = _conv2d_fixed_padding(inputs=inputs[i],
                                            filters=f,
                                            kernel_size=self.kernel_size,
                                            strides=self.conv_stride,
                                            data_format=self.data_format,
                                            weight_decay=self.weight_decay)
                imgs.append(tf.identity(img, f'initial_conv{i}'))

            # We do not include batch normalization or activation functions in V2
            # for the initial conv1 because the first ResNet unit will perform these
            # for both the shortcut and non-shortcut paths as part of the first
            # block's projection. Cf. Appendix of [2].
            if self.resnet_version == 1:
                for i, img in enumerate(imgs):
                    imgs[i] = _batch_norm(img, training, self.data_format)
                    imgs[i] = tf.nn.relu(img)

            if self.first_pool_size:
                for i, img in enumerate(imgs):
                    img = tf.layers.max_pooling2d(
                        inputs=img,
                        pool_size=self.first_pool_size,
                        strides=self.first_pool_stride,
                        padding='SAME',
                        data_format=self.data_format)
                    imgs[i] = tf.identity(img, f'initial_max_pool{i}')

            for i, num_blocks in enumerate(self.block_sizes):
                for j, img in enumerate(imgs):
                    if type(filters[j]) is list:
                        f = filters[j][i + 1]
                    else:
                        f = filters[j] * (2**i)
                    imgs[j] = _block_layer(inputs=img,
                                           filters=f,
                                           bottleneck=self.bottleneck,
                                           block_fn=self.block_fn,
                                           blocks=num_blocks,
                                           strides=self.block_strides[i],
                                           training=training,
                                           name='rgb_block_layer{}'.format(i +
                                                                           1),
                                           data_format=self.data_format,
                                           weight_decay=self.weight_decay,
                                           film_gamma_betas=None)

            inputs = tf.concat(imgs, axis=3)

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = _batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)
            inputs = tf.identity(inputs, 'pre_final_tower')

            for i in range(2):
                inputs = _block_layer(inputs=inputs,
                                      filters=64,
                                      bottleneck=self.bottleneck,
                                      block_fn=self.block_fn,
                                      blocks=3,
                                      strides=self.block_strides[i],
                                      training=training,
                                      name=f'tower_block_layer{i}',
                                      data_format=self.data_format,
                                      weight_decay=self.weight_decay,
                                      film_gamma_betas=None)

            for _ in range(5):
                inputs = _conv2d_fixed_padding(inputs=inputs,
                                               filters=1,
                                               kernel_size=8,
                                               strides=self.conv_stride,
                                               data_format=self.data_format,
                                               weight_decay=self.weight_decay)

            inputs = tf.squeeze(inputs)
            inputs = tf.nn.sigmoid(inputs)
            inputs = tf.identity(inputs, 'predictions')
            return inputs
