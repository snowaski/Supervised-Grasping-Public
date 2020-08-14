import sys, os
import gin
import numpy as np
import shutil
import tensorflow.compat.v1 as tf
sys.path.append(os.path.abspath('.'))

import resnet_model
from tensor2robot.input_generators import default_input_generator
from tensor2robot.utils import train_eval
from tensorflow.contrib import predictor as contrib_predictor
from typing import Tuple, List

_MAX_TRAIN_STEPS = 100
_EVAL_STEPS = 40
_BATCH_SIZE = 8
_EVAL_THROTTLE_SECS = 0.0


def create_dummy_data() -> Tuple[List[dict], np.ndarray]:
    """creates dummy data to test model."""
    features = []
    for _ in range(_BATCH_SIZE):
        entry = {
            'imgs/RGB':
            np.random.randint(0,
                              high=256,
                              size=(1, 128, 128, 3),
                              dtype=np.uint8),
            'imgs/Feature_RGB':
            np.random.randint(0,
                              high=256,
                              size=(1, 128, 128, 3),
                              dtype=np.uint8),
            'imgs/Depth':
            np.expand_dims(tf.io.serialize_tensor(
                np.random.randn(128, 128, 1).astype(np.float32)).numpy(),
                           axis=0),
            'imgs/Feature_Depth':
            np.expand_dims(tf.io.serialize_tensor(
                np.random.randn(128, 128, 3).astype(np.float32)).numpy(),
                           axis=0),
            'imgs/Target':
            np.random.randint(0,
                              high=256,
                              size=(1, 128, 128, 3),
                              dtype=np.uint8)
        }
        features.append(entry)

    return features, np.random.randint(0, 2, size=(_BATCH_SIZE))


class TrainEvalTest(tf.test.TestCase):
    def tearDown(self):
        gin.clear_config()
        super(TrainEvalTest, self).tearDown()

    def test_train_eval_model(self):
        """Tests that a simple model trains and exported models are valid."""
        gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps',
                           100)
        model_dir = './model_tests/test_model'
        t2r_model = resnet_model.GraspingModel()

        input_generator_train = default_input_generator.DefaultRecordInputGenerator(
            batch_size=_BATCH_SIZE,
            file_patterns='model_tests/test_files/testing_train.tfrecord')
        input_generator_eval = default_input_generator.DefaultRecordInputGenerator(
            batch_size=_BATCH_SIZE,
            file_patterns='model_tests/test_files/testing_test.tfrecord')

        train_eval.train_eval_model(
            t2r_model=t2r_model,
            input_generator_train=input_generator_train,
            input_generator_eval=input_generator_eval,
            max_train_steps=_MAX_TRAIN_STEPS,
            model_dir=model_dir,
            train_hook_builders=None,
            eval_hook_builders=None,
            eval_steps=_EVAL_STEPS,
            eval_throttle_secs=_EVAL_THROTTLE_SECS,
            create_exporters_fn=train_eval.create_default_exporters)

        # We ensure that both numpy and tf_example inference models are exported.
        latest_exporter_numpy_path = os.path.join(model_dir, 'export',
                                                  'latest_exporter_numpy', '*')
        numpy_model_paths = sorted(
            tf.io.gfile.glob(latest_exporter_numpy_path))
        # There should be at least 1 exported model.
        self.assertGreater(len(numpy_model_paths), 0)
        # This mock network converges nicely which is why we have several best
        # models, by default we keep the best 5 and the latest one is always the
        # best.
        self.assertLessEqual(len(numpy_model_paths), 5)

        latest_exporter_tf_example_path = os.path.join(
            model_dir, 'export', 'latest_exporter_tf_example', '*')

        tf_example_model_paths = sorted(
            tf.io.gfile.glob(latest_exporter_tf_example_path))
        # There should be at least 1 exported model.
        self.assertGreater(len(tf_example_model_paths), 0)
        # This mock network converges nicely which is why we have several best
        # models, by default we keep the best 5 and the latest one is always the
        # best.
        self.assertLessEqual(len(tf_example_model_paths), 5)

        # Now we can load our exported estimator graph with the numpy feed_dict
        # interface, there are no dependencies on the model_fn or preprocessor
        # anymore.
        # We load the latest model since it had the best eval performance.
        numpy_predictor_fn = contrib_predictor.from_saved_model(
            numpy_model_paths[-1])

        features, labels = create_dummy_data()

        numpy_predictions = []
        for feature, label in zip(features, labels):
            predicted = numpy_predictor_fn(feature)['grasp_success'].flatten()
            numpy_predictions.append(predicted)

        shutil.rmtree('model_tests/test_model')

    def test_robot_error(self):
        rgb = np.load('model_tests/test_files/error_example/rgb.npy')
        feature_rgb = np.load(
            'model_tests/test_files/error_example/feature_rgb.npy')
        depth = np.load(
            'model_tests/test_files/error_example/depth.npy').astype(
                np.float32)

        # for some reason there's a seg fault when feature_depth has zeros, so
        # the zeros are replaced with a decimal very close to zero.
        feature_depth = np.full((128, 128, 3), 0.000000001).astype(np.float32)
        norms = [0.6587999, 0.671825, 0.5795498]
        for c in range(3):
            for row in range(128):
                for col in range(128):
                    if feature_rgb[row, col, c] != 0:
                        feature_depth[row, col, c] = norms[c]

        best_exporter_numpy_path = os.path.join('latest_model', 'export',
                                                'latest_exporter_numpy', '*')
        numpy_model_paths = sorted(tf.io.gfile.glob(best_exporter_numpy_path))
        numpy_predictor_fn = contrib_predictor.from_saved_model(
            numpy_model_paths[-1])

        data_no_target = {
            'imgs/RGB':
            np.expand_dims(rgb, axis=0),
            'imgs/Feature_RGB':
            np.expand_dims(feature_rgb, axis=0),
            'imgs/Depth':
            np.expand_dims(tf.io.serialize_tensor(depth), axis=0),
            'imgs/Feature_Depth':
            np.expand_dims(tf.io.serialize_tensor(feature_depth), axis=0),
        }

        data_target = {
            'imgs/RGB':
            np.expand_dims(rgb, axis=0),
            'imgs/Feature_RGB':
            np.expand_dims(feature_rgb, axis=0),
            'imgs/Depth':
            np.expand_dims(tf.io.serialize_tensor(depth), axis=0),
            'imgs/Feature_Depth':
            np.expand_dims(tf.io.serialize_tensor(feature_depth), axis=0),
            'imgs/Target':
            np.expand_dims(rgb, axis=0)
        }

        try:
            predicted = numpy_predictor_fn(
                data_target)['grasp_success'].flatten()
        except ValueError:
            predicted = numpy_predictor_fn(
                data_no_target)['grasp_success'].flatten()

        self.assertLessEqual(predicted, 0.001)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
