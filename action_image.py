import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensor2robot.preprocessors import image_transformations
from matplotlib import pyplot as plt


def crop_to_target(t_x, t_y, img, dims):
    '''crop image around a point'''
    height, width, _ = img.shape

    if t_x + dims[0] // 2 > width:
        off = t_x + 64 - width
        t_x -= off
    if t_y + dims[1] // 2 > height:
        off = t_y + 64 - height
        t_y -= off
    if t_x - dims[0] // 2 < 0:
        off = 64 - t_x
        t_x += off
    if t_y - dims[1] // 2 < 0:
        off = 64 - t_y
        t_y += off

    return tf.image.crop_to_bounding_box(img, t_y - 64, t_x - 64, dims[0],
                                         dims[1])


def draw_circle(radius, x, y, value):
    img = np.zeros((512, 640, 1))
    for r in range(radius * 2 + 1):
        for c in range(radius * 2 + 1):
            t_y = y - radius + r
            t_x = x - radius + c
            if t_x < 640 and t_y < 512 and t_x >= 0 and t_y >= 0:
                img[t_y][t_x] = value

    return img


def extract_array_from_string(s):
    arr = s.split(' ')
    return np.array(arr, dtype=np.float32)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def project_points_to_image_space(p1, p2, p3):
    D = [0, 0, 0.0, 0.0, 0]
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    points = np.array([p1, p2, p3])
    points, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3, ), cipm,
                                  dist_coeffs)
    points = points.astype('int32')

    return points


def draw_feature_img(points):
    img = []
    for i in range(3):
        img.append(draw_circle(6, points[i][0][0], points[i][0][1], 255))
    return np.concatenate(img, axis=2)


def crop_imgs(points, feature_rgb, feature_depth, rgbd):
    center_x = points[0][0][0]
    center_y = points[0][0][1]
    feature_rgb = crop_to_target(center_x, center_y, feature_rgb, (128, 128))
    feature_depth = crop_to_target(center_x, center_y, feature_depth,
                                   (128, 128))
    rgbd = crop_to_target(center_x, center_y, rgbd, (128, 128))

    return feature_rgb, feature_depth, rgbd


def create_depth_img(rgbd):
    depth = tf.expand_dims(rgbd[:, :, 3], axis=2)
    depth = np.concatenate(
        [depth, np.zeros(depth.shape),
         np.zeros(depth.shape)], axis=2)

    return depth


def create_photometric_distortion_with_noise(rgbd):
    features = np.expand_dims(rgbd[:, :, :3], axis=0) / 256
    features = image_transformations.ApplyPhotometricImageDistortions(
        features,
        random_brightness=True,
        random_saturation=True,
        random_hue=True,
        random_noise_level=0.05)
    return features * 256


def create_photometric_distortion_no_noise(rgbd):
    features = np.expand_dims(rgbd[:, :, :3], axis=0) / 256
    features = image_transformations.ApplyPhotometricImageDistortions(
        features,
        random_brightness=True,
        random_saturation=True,
        random_hue=True)
    return features * 256


def create_dataset(cipm):
    '''creates the data set

    Parameters: 
    data -- the existing data (Pandas Dataframe)
    cipm -- the camera intrinsic projection matrix (np array)

    Returns:
    imgs -- An Nx5 array where N is the number of data points. Each element has the rgb, rgb action, depth, 
    and depth action images
    labels -- An Nx1 array that labels each grasp as a success or not
    '''
    imgs = []
    labels = []
    print(os.listdir('data/'))
    for d in os.listdir('data/'):
        if d == '.DS_Store':
            continue
        data = pd.read_csv(f'data/{d}/data.csv')
        for i, id in enumerate(data['scenario_id']):
            # extract data
            try:
                rgbd = np.load(f'data/{d}/{id}/rgbd.data')
            except:
                continue
            feature_left_finger = extract_array_from_string(
                data['left_fingertip_position'][i])
            feature_right_finger = extract_array_from_string(
                data['right_fingertip_position'][i])
            feature_wrist = extract_array_from_string(
                data['base_gripper_position'][i])

            points = project_points_to_image_space(feature_left_finger,
                                                   feature_right_finger,
                                                   feature_wrist)

            # draw action images
            feature_rgb = draw_feature_img(points)
            feature_depth = draw_feature_img(points)

            feature_rgb, feature_depth, rgbd = crop_imgs(
                points, feature_rgb, feature_depth, rgbd)

            depth = create_depth_img(rgbd)

            # determine success
            success = data['grasp_success'][
                i] and not data['pieces_knocked_over'][i]

            imgs.append([rgbd[:, :, :3], feature_rgb, depth, feature_depth])
            labels.append(success)

            # create imgs with photometric distortion
            for _ in range(2):
                features = create_photometric_distortion_with_noise(rgbd)
                imgs.append([features[0], feature_rgb, depth, feature_depth])
                labels.append(success)

            features = create_photometric_distortion_no_noise(rgbd)
            imgs.append([features[0], feature_rgb, depth, feature_depth])
            labels.append(success)

    return imgs, np.array(labels)


def serialize(img, lbl):
    '''converts data to TFExamples'''
    rgb, action_rgb, depth, action_depth = img
    rgb = tf.cast(rgb, tf.uint8)
    encoded_rgb = tf.io.encode_jpeg(rgb)
    action_rgb = tf.cast(action_rgb, tf.uint8)
    encoded_action_rgb = tf.io.encode_jpeg(action_rgb)
    depth = tf.cast(depth, tf.float32)
    encoded_depth = tf.io.serialize_tensor(depth)
    action_depth = tf.cast(action_depth, tf.float32)
    encoded_action_depth = tf.io.serialize_tensor(action_depth)

    feature = {
        'rgb': _bytes_feature(encoded_rgb),
        'feature_rgb': _bytes_feature(encoded_action_rgb),
        'depth': _bytes_feature(encoded_depth),
        'feature_depth': _bytes_feature(encoded_action_depth),
        'grasp_success': _int64_feature(lbl)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_imgs(X_train,
               X_test,
               y_train,
               y_test,
               record_file='images.tfrecords'):
    with tf.io.TFRecordWriter('train.tfrecord') as writer:
        for img, lbl in zip(X_train, y_train):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('test.tfrecord') as writer:
        for img, lbl in zip(X_test, y_test):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    tf.enable_eager_execution()

    cipm = np.array([[739.22373806060455 / 2, 0, 640.56587982177734 / 2],
                     [0, 739.22373806060455 / 2, 512.44139003753662 / 2],
                     [0, 0, 1]])

    imgs, labels = create_dataset(cipm)
    print(len(imgs))
    X_train, X_test, y_train, y_test = train_test_split(imgs,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=890,
                                                        shuffle=True)

    write_imgs(X_train, X_test, y_train, y_test)
