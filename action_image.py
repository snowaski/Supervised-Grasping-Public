import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import base64

def crop_to_target(t_x, t_y, img, dims):
    '''crop image around a point'''
    height, width, _ = img.shape

    if t_x + dims[0]//2 > width:
        off = t_x + 64 - width
        t_x -= off
    if t_y + dims[1]//2 > height:
        off = t_y + 64 - height
        t_y -= off
    if t_x - dims[0]//2 < 0:
        off = 64 - t_x
        t_x += off
    if t_y - dims[1]//2 < 0:
        off = 64 - t_y
        t_y += off

    return tf.image.crop_to_bounding_box(img, t_y-64, t_x-64, dims[0], dims[1])

def draw_circle(radius, x, y, value):
    img = np.zeros((512, 640, 1))
    for r in range(radius*2+1):
        for c in range(radius*2+1):
            if x < 640 and y < 512 and x >= 0 and y >= 0:
                img[y-radius+r][x-radius+c] = value

    return img

def extract_array_from_string(s):
    arr = s.split(' ')
    return np.array(arr, dtype=float)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_dataset(data, cipm):
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
    for i, id in enumerate(data['scenario_id']):
        # extract data
        try:
            rgbd = np.load(f'data/{id}/rgbd.data')
        except:
            continue
        feature_left_finger = extract_array_from_string(data['left_fingertip_position'][i])
        feature_right_finger = extract_array_from_string(data['right_fingertip_position'][i])
        feature_wrist = extract_array_from_string(data['base_gripper_position'][i])

        # project the points to image space
        D = [0, 0, 0.0, 0.0, 0]
        dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
        points = np.array([feature_left_finger, feature_right_finger, feature_wrist])
        points, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3,), cipm, dist_coeffs)
        points = points.astype('int32')

        # draw rgb action image
        f1_rgb = draw_circle(6, points[0][0][0], points[0][0][1], 255)
        f2_rgb = draw_circle(6, points[1][0][0], points[1][0][1], 255)
        f3_rgb = draw_circle(6, points[2][0][0], points[2][0][1], 255)
        img_rgb = np.concatenate([f1_rgb, f2_rgb, f3_rgb], axis=2)

        # draw depth action image
        f1_depth = draw_circle(6, points[0][0][0], points[0][0][1], np.linalg.norm([feature_left_finger]))
        f2_depth = draw_circle(6, points[1][0][0], points[1][0][1], np.linalg.norm(feature_right_finger))
        f3_depth = draw_circle(6, points[2][0][0], points[2][0][1], np.linalg.norm(feature_wrist))
        img_depth = np.concatenate([f1_depth, f2_depth, f3_depth], axis=2)

        # crop images around a central point
        center_x = points[0][0][0]
        center_y = points[0][0][1]
        img_rgb = crop_to_target(center_x, center_y, img_rgb, (128, 128))
        img_depth = crop_to_target(center_x, center_y, img_depth, (128, 128))
        rgbd = crop_to_target(center_x, center_y, rgbd, (128, 128))

        success = data['grasp_success'][i] and not data['pieces_knocked_over'][i]

        imgs.append([rgbd[:, :, :3], img_rgb, rgbd[:, :, 3], img_depth])
        labels.append(success)

    return imgs, np.array(labels)

def serialize(img, lbl):
    '''converts data to TFExamples'''
    rgb, action_rgb, depth, action_depth = img
    rgb = tf.cast(rgb, tf.uint8)
    encoded_rgb = tf.io.encode_jpeg(rgb)
    action_rgb = tf.cast(action_rgb, tf.uint8)
    encoded_action_rgb = tf.io.encode_jpeg(action_rgb)
    encoded_depth = tf.io.serialize_tensor(depth)
    encoded_action_depth = tf.io.serialize_tensor(action_depth)

    feature = {
        'height': _int64_feature(512),
        'width': _int64_feature(640),
        'channels': _int64_feature(3),
        'label': _int64_feature(lbl),
        'rgb_raw': _bytes_feature(encoded_rgb),
        'action_rgb_raw': _bytes_feature(encoded_action_rgb),
        'depth_raw': _bytes_feature(encoded_depth),
        'action_depth_raw': _bytes_feature(encoded_action_depth)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_imgs(X_train, X_test, y_train, y_test, record_file='images.tfrecords'):
    with tf.io.TFRecordWriter('train.tfrecord') as writer:
        for img, lbl in zip(X_train, y_train):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('test.tfrecord') as writer:
        for img, lbl in zip(X_test, y_test):
            tf_example = serialize(img, lbl)
            writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')
    tf.enable_eager_execution()

    cipm = np.array([[739.22373806060455/2, 0, 640.56587982177734/2],
                     [0, 739.22373806060455/2, 512.44139003753662/2],
                     [0,                    0,                    1]])

    imgs, labels = create_dataset(data, cipm)
    # assert(False), f'{imgs.shape}'
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=890)

    write_imgs(X_train, X_test, y_train, y_test)