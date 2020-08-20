import sys, os
sys.path.append(os.path.abspath('.'))

import tensorflow.compat.v1 as tf
import numpy as np
import action_image as action
from sklearn.model_selection import train_test_split

CROP_SIZE = (128, 128)
IMG_WIDTH = 640
IMG_HEIGHT = 512
cipm = np.array([[739.22373806060455 / 2, 0, 640.56587982177734 / 2],
                 [0, 739.22373806060455 / 2, 512.44139003753662 / 2],
                 [0, 0, 1]])


class ActionImageTest(tf.test.TestCase):
    def tearDown(self):
        super(ActionImageTest, self).tearDown()

    def test_crop_to_target(self):
        """tests that crop to target crops around the right point."""
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        img[IMG_HEIGHT // 2, IMG_WIDTH // 2, :] = 1

        crop = action.crop_to_target(
            np.array([IMG_WIDTH // 2, IMG_HEIGHT // 2], dtype=np.int32), img,
            CROP_SIZE)

        target = np.zeros((CROP_SIZE[0], CROP_SIZE[1], 3))
        target[CROP_SIZE[0] // 2, CROP_SIZE[1] // 2, :] = 1

        target = tf.convert_to_tensor(target)

        self.assertAllEqual(target, crop)

    def test_project_points_to_image_space(self):
        """tests that 3D points are properly projected to image space."""
        points = [[-0.112247, 0.192249, 0.820621]]

        projected = action.project_points_to_image_space(points, cipm)

        self.assertAllEqual(projected, [[[269, 342]]])

    def test_draw_point(self):
        """tests that points are appropriately drawn on an image."""
        img = action.draw_point(0, IMG_WIDTH // 2, IMG_HEIGHT // 2, 1)

        target = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
        target[IMG_HEIGHT // 2, IMG_WIDTH // 2, :] = 1

        self.assertAllEqual(img, target)

    def test_create_photometric_distortion_with_noise(self):
        """tests that distortion is applied to an image."""
        images = np.zeros((1, IMG_HEIGHT, IMG_HEIGHT, 3))
        distorted = action.create_photometric_distortion_with_noise(images)
        delta = tf.reduce_sum(tf.square(images - distorted))

        # Check if any distortion applied.
        self.assertGreater(delta, 0)

    def test_write_imgs(self):
        """tests that data is properly written to a tfrecord."""
        X = []
        batch_size = 10
        for _ in range(batch_size):
            X.append([
                np.zeros((128, 128, 3)),
                np.zeros((128, 128, 3)),
                np.zeros((128, 128, 1)),
                np.zeros((128, 128, 3)),
                np.zeros((128, 128, 3)),
            ])

        y = np.random.randint(0, 2, size=(batch_size))

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=890,
                                                            shuffle=True)

        train_file_path = 'action_images/test_files/test_train.tfrecord'
        test_file_path = 'action_images/test_files/test_train.tfrecord'
        action.write_imgs(X_train, X_test, y_train, y_test, train_file_path,
                          test_file_path)

        train_file = tf.io.gfile.glob(train_file_path)
        self.assertEqual(len(train_file), 1)

        test_file = tf.io.gfile.glob(test_file_path)
        self.assertEqual(len(test_file), 1)

    def test_create_dataset_with_target(self):
        """tests that action images are correctly created."""
        rgbd = np.load('action_images/test_files/rgbd.npy')
        points = np.load('action_images/test_files/points.npy')
        success = np.load('action_images/test_files/success.npy')

        data = [[rgbd, points[0], points[1], points[2], points[3], success]]

        imgs, lbls = action.create_dataset(cipm, data)

        imgs = imgs[0]

        self.assertAllEqual(
            imgs[0],
            np.load('action_images/test_files/generic_example/rgb.npy'))
        self.assertAllEqual(
            imgs[1],
            np.load(
                'action_images/test_files/generic_example/feature_rgb.npy'))
        self.assertAllEqual(
            imgs[2],
            np.load('action_images/test_files/generic_example/depth.npy'))
        self.assertAllEqual(
            imgs[3],
            np.load(
                'action_images/test_files/generic_example/feature_depth.npy'))
        self.assertAllEqual(
            imgs[4],
            np.load('action_images/test_files/generic_example/target.npy'))

    def test_create_dataset_without_target(self):
        """tests that action images are correctly created."""
        rgbd = np.load('action_images/test_files/rgbd.npy')
        points = np.load('action_images/test_files/points.npy')
        success = np.load('action_images/test_files/success.npy')

        data = [[rgbd, points[0], points[1], points[2], None, success]]

        imgs, lbls = action.create_dataset(cipm, data)

        imgs = imgs[0]

        self.assertAllEqual(
            imgs[0],
            np.load('action_images/test_files/generic_example/rgb.npy'))
        self.assertAllEqual(
            imgs[1],
            np.load(
                'action_images/test_files/generic_example/feature_rgb.npy'))
        self.assertAllEqual(
            imgs[2],
            np.load('action_images/test_files/generic_example/depth.npy'))
        self.assertAllEqual(
            imgs[3],
            np.load(
                'action_images/test_files/generic_example/feature_depth.npy'))
        self.assertAllEqual(
            imgs[4],
            None)

    def test_get_data_with_balance(self):
        """tests that get_data balances the positive and negative examples"""
        data = action.get_data(True, True, False, './action_images/test_files/example_data')
        pos = 0
        neg = 0

        for sample in data:
            if sample[-1]:
                pos += 1
            else:
                neg += 1

        self.assertEqual(pos, neg)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
