import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import os
import argparse
from sklearn.model_selection import train_test_split
from tensor2robot.preprocessors import image_transformations
from typing import List, Tuple, Optional

from matplotlib import pyplot as plt

CROPPED_IMAGE_SIZE = (128, 128)


def crop_to_target(center: np.ndarray, img: np.ndarray,
                   dims: Tuple[int, int]) -> tf.image:
    """Crops the input image according to the dimensions.

    Args:
        center: an array representing the center position.
        img: the image to crop.
        dims: a Tuple[int, int] representing the dimensions to crop.

    Returns:
        a tf.image.
    """
    t_x, t_y = center
    height, width, _ = img.shape
    half_crop_width = dims[0] // 2
    half_crop_height = dims[1] // 2

    t_x = tf.clip_by_value(t_x, half_crop_width, width - half_crop_width)
    t_y = tf.clip_by_value(t_y, half_crop_height, height - half_crop_height)

    return tf.image.crop_to_bounding_box(img, t_y - half_crop_height,
                                         t_x - half_crop_width, dims[0],
                                         dims[1])


def _extract_array_from_string(s: str) -> np.ndarray:
    """Ease of use function to extract an array from a string.

    Args:
        s: string to extract from.

    Returns:
        an np array.
    """
    arr = s.split(' ')
    return np.array(arr, dtype=np.float32)


def _bytes_feature(value: tf.Tensor) -> tf.train.Feature:
    """Converts a byte string to a tf.Feature to be used in a record.

    Args:
        value: the value to converts.

    Returns:
        a tf.train.Feature from value.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: tf.Tensor) -> tf.train.Feature:
    """Converts an int_64 to a tf.Feature to be used in a record.

    Args:
        value: the value to converts.

    Returns:
        a tf.train.Feature from value
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def project_points_to_image_space(points: np.ndarray,
                                  cipm: np.ndarray) -> np.ndarray:
    """Projects the points from 3D camera frame to a 2D image frame.

    Args:
        points: the array of points to project.

    Returns:
        an np array of the projected points.
    """
    D = [0, 0, 0.0, 0.0, 0]

    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    points = np.array(points)
    points, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3, ), cipm,
                                  dist_coeffs)

    return points.astype('int32')


def draw_point(radius: int, x: int, y: int, value: int) -> np.ndarray:
    """Draws a point of a certain radius on an image.

    Args:
        radius: the radius of the points drawn.
        x: the x value to draw the point at.
        y: the y value to draw the point at.
        value: the value that is filled in the image.
  
    Returns:
        an np array representing the image.
    """
    img = np.zeros((512, 640, 1))
    for r in range(radius * 2 + 1):
        for c in range(radius * 2 + 1):
            t_y = y - radius + r
            t_x = x - radius + c
            if 0 <= t_x < 640 and 0 <= t_y < 512:
                img[t_y, t_x] = value

    return img


def draw_feature_img(points: np.ndarray, value: list) -> np.ndarray:
    """Creates a feature image by drawing a point at point with a specific value.

    Args:
        points: an array of points.
        value: an array with what value to draw at each point.

    Returns:
        an np array representing an image.
    """
    img = []
    circle_radius = 6
    for i, val in enumerate(value):
        img.append(
            draw_point(circle_radius, points[i, 0, 0], points[i, 0, 1], val))
    return np.concatenate(img, axis=2)


def crop_imgs(points: np.ndarray, feature_rgb: np.ndarray,
              feature_depth: np.ndarray,
              rgbd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crops the action images around point and target point.

    Args:
        points: the grasp candidate points to crop around.
        feature_rgb: the feature rgb image.
        feature_depth: the feature_depth image.
        rgbd: the rgbd image.

    Returns:
        a tuple of cropped np array images.
    """
    points = np.squeeze(points)
    points = np.sum(points, axis=0) / 3
    points = points.astype(np.int32)

    feature_rgb = crop_to_target(points, feature_rgb, CROPPED_IMAGE_SIZE)
    feature_depth = crop_to_target(points, feature_depth, CROPPED_IMAGE_SIZE)
    rgbd = crop_to_target(points, rgbd, CROPPED_IMAGE_SIZE)

    return feature_rgb, feature_depth, rgbd


def create_photometric_distortion_with_noise(rgb: np.ndarray) -> np.ndarray:
    """Creates a new image based on rgbd with photomoetric distortion, 
    including gaussian noise.

    Args:
        rgb: the image to transform.

    Returns:
        the transformed image.
    """
    image = np.expand_dims(rgb, axis=0) / 256
    image = image_transformations.ApplyPhotometricImageDistortions(
        image,
        random_brightness=True,
        random_saturation=True,
        random_hue=True,
        random_noise_level=0.05)
    return image[0] * 256


def create_photometric_distortion_no_noise(rgb: np.ndarray) -> np.ndarray:
    """Creates a new image based on rgbd with photomoetric distortion, 
    including gaussian noise.

    Args:
        rgb: the image to transform.

    Returns:
        the transformed image.
    """
    image = np.expand_dims(rgb, axis=0) / 256
    image = image_transformations.ApplyPhotometricImageDistortions(
        image, random_brightness=True, random_saturation=True, random_hue=True)
    return image[0] * 256


def get_data(mode: str, balance: bool, data_dir: str = 'data/') -> list:
    """Extracts the data from data_dir using data.csv.

    Args:
        mode: determines which images are created. Either "target" or "no-target".
        balance: determines whether or not to balance the negative and positive examples.
        data_dir: the directory path where the data is.

    Returns:
        a list with the rgbd data, the positions of the left, right, and base gripper,
        the target position if the mode is target, otherwise None, and whether
        grasp is a success.
        the number of successful grasps.
    """
    data = []
    print(os.listdir(data_dir))
    for d in os.listdir(data_dir):
        if d == '.DS_Store':
            continue
        csv = pd.read_csv(f'{data_dir}/{d}/data.csv')
        bal = not (csv['grasp_success'].value_counts()[True] > csv['grasp_success'].value_counts()[False])
        example_limit = csv['grasp_success'].value_counts()[bal]
        print(example_limit)
        balance_examples = 0

        for i, id in enumerate(csv['scenario_id']):
            if balance and balance_examples > example_limit:
                continue
            entry = []
            try:
                entry.append(
                    np.load(f'{data_dir}/{d}/{id}/rgbd.data')['arr_0'])
            except KeyError:
                continue

            entry.append(
                _extract_array_from_string(csv['left_fingertip_position'][i]))
            entry.append(
                _extract_array_from_string(csv['right_fingertip_position'][i]))
            entry.append(
                _extract_array_from_string(csv['base_gripper_position'][i]))

            if mode == 'target':
                entry.append(
                    _extract_array_from_string(csv['grasp_pose_position'][i]))
            else:
                entry.append(None)

            # determine success
            entry.append(csv['grasp_success'][i]
                         and not csv['pieces_knocked_over'][i])
            
            if entry[-1] != bal:
                balance_examples += 1

            data.append(entry)
        print(balance_examples)

    return data


def create_dataset(cipm: np.ndarray, data: list) -> Tuple[list, np.ndarray]:
    '''transforms the data into action images, with rgb, depth, feature_rgb,
    feature_depth, and target images.

    Args:
        cipm: the camera intrinsic projection matrix.
        data: a list containg list with rgbd data, 3D points, and the success of the entry.

    Returns:
        a Tuple[list, np.ndarray] with the action images and their labels.
    '''
    imgs = []
    labels = []

    for rgbd, feature_left_finger, feature_right_finger, feature_wrist, target, success in data:
        # project points
        points = np.array(
            [feature_left_finger, feature_right_finger, feature_wrist])
        points = project_points_to_image_space(points, cipm)

        # draw action images
        feature_rgb = draw_feature_img(points, [255] * 3)
        norms = [
            np.linalg.norm(feature_left_finger),
            np.linalg.norm(feature_right_finger),
            np.linalg.norm(feature_wrist)
        ]
        feature_depth = draw_feature_img(points, norms)

        # create target image
        if target is not None:
            target_points = project_points_to_image_space(
                np.array([target]), cipm)
            target_img = crop_to_target(target_points[0, 0], rgbd[:, :, :3],
                                        CROPPED_IMAGE_SIZE)
        else:
            target_img = None

        # crop images
        feature_rgb, feature_depth, rgbd = crop_imgs(points, feature_rgb,
                                                     feature_depth, rgbd)

        depth = tf.expand_dims(rgbd[:, :, 3], axis=2)
        rgb = rgbd[:, :, :3]

        imgs.append([rgb, feature_rgb, depth, feature_depth, target_img])
        labels.append(success)

        # create imgs with photometric distortion
        for _ in range(2):
            transformed_image = create_photometric_distortion_with_noise(rgb)
            if target is not None:
                target_img = create_photometric_distortion_with_noise(
                    target_img)
            imgs.append([
                transformed_image, feature_rgb, depth, feature_depth,
                target_img
            ])
            labels.append(success)

        transformed_image = create_photometric_distortion_no_noise(rgb)
        if target is not None:
            target_img = create_photometric_distortion_no_noise(target_img)
        imgs.append(
            [transformed_image, feature_rgb, depth, feature_depth, target_img])
        labels.append(success)

    return imgs, np.array(labels)


def serialize(imgs: list, lbl: int) -> tf.train.Example:
    '''converts data to TFExamples.

    Args:
        imgs: a list that contains the action images.
        lbl: the label for the example.
    
    Returns:
        TFExample containing the passed in data.
    '''
    rgb, feature_rgb, depth, feature_depth, target = imgs
    rgb = tf.cast(rgb, tf.uint8)
    encoded_rgb = tf.io.encode_jpeg(rgb)
    if target is not None:
        target = tf.cast(target, tf.uint8)
        encoded_target = tf.io.encode_jpeg(target)
    feature_rgb = tf.cast(feature_rgb, tf.uint8)
    encoded_feature_rgb = tf.io.encode_jpeg(feature_rgb)
    depth = tf.cast(depth, tf.float32)
    encoded_depth = tf.io.serialize_tensor(depth)
    feature_depth = tf.cast(feature_depth, tf.float32)
    encoded_feature_depth = tf.io.serialize_tensor(feature_depth)

    feature = {
        'rgb': _bytes_feature(encoded_rgb),
        'feature_rgb': _bytes_feature(encoded_feature_rgb),
        'depth': _bytes_feature(encoded_depth),
        'feature_depth': _bytes_feature(encoded_feature_depth),
        'grasp_success': _int64_feature(lbl)
    }

    if target is not None:
        feature['target'] = _bytes_feature(encoded_target),

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_imgs(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
               y_test: np.ndarray, train_file: str, test_file: str):
    """Writes the data as a tfrecord to a file.

    Args:
        X_train: an array with the training data
        X_test: an array with the testing data
        y_train: the training labels
        y_test: the testing labels
    """
    with tf.io.TFRecordWriter(train_file) as writer:
        for img, lbl in zip(X_train, y_train):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter(test_file) as writer:
        for img, lbl in zip(X_test, y_test):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["target", "no-target"],
                        default="target",
                        help='determines whether to create target images')
    parser.add_argument(
        "--balance",
        action='store_true',
        help='determines whether to balance positive and negative examples')
    parser.add_argument("--data-dir",
                        default="data/",
                        help='the directory to find data')
    args = parser.parse_args()

    tf.enable_eager_execution()

    cipm = np.array([[739.22373806060455 / 2, 0, 640.56587982177734 / 2],
                     [0, 739.22373806060455 / 2, 512.44139003753662 / 2],
                     [0, 0, 1]])

    data = get_data(args.mode, args.balance, args.data_dir)
    print(len(data))
    imgs, labels = create_dataset(cipm, data)

    print(len(imgs))

    X_train, X_test, y_train, y_test = train_test_split(imgs,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=890,
                                                        shuffle=True)

    write_imgs(X_train, X_test, y_train, y_test, 'train.tfrecord',
               'test.tfrecord')
