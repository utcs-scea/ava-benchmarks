# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Open-source TensorFlow Inception v3 Example. Remove
   inception_preprocessing dependency"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops

from tensorflow.contrib import summary
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.training.python.training import evaluation

import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../../guestlib/python'))
from hook import *


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string(
    'data_dir', '',
    'Directory where input data is stored')

flags.DEFINE_string(
    'model_dir', None,
    'Directory where model output is stored')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_integer(
    'num_shards', 8,
    'Number of shards (workers).')

flags.DEFINE_integer(
    'iterations', 100,
    'Number of iterations per TPU training loop.')

flags.DEFINE_bool(
    'skip_host_call', default=True,
    help=('Skip the host call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the computation running on the TPU.'))

flags.DEFINE_integer(
    'train_batch_size', 1024,
    'Global (not per-shard) batch size for training')

flags.DEFINE_integer(
    'eval_total_size', 0,
    'Total batch size for evaluation, use the entire validation set if 0')

flags.DEFINE_integer(
    'eval_batch_size', 1024,
    'Global (not per-shard) batch size for evaluation')

flags.DEFINE_integer(
    'train_steps', 213000,
    'Number of steps use for training.')

flags.DEFINE_integer(
    'train_steps_per_eval', 2000,
    'Number of training steps to run between evaluations.')

flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval')

flags.DEFINE_integer(
    'min_eval_interval', 180,
    'Minimum number of seconds between evaluations')

flags.DEFINE_integer(
    'eval_timeout', None,
    'Evaluation timeout: Maximum number of seconds that '
    'may elapse while no new checkpoints are observed')

flags.DEFINE_bool(
    'use_tpu', True,
    'Use TPUs rather than plain CPUs')

flags.DEFINE_boolean(
    'per_host_input_for_training', True,
    'If true, input_fn is invoked per host rather than per shard.')

flags.DEFINE_string(
    'use_data', 'real',
    'One of "fake","real"')

flags.DEFINE_float(
    'learning_rate', 0.165,
    'Learning rate.')

flags.DEFINE_string(
    'optimizer', 'RMS',
    'Optimizer (one of sgd, RMS, momentum)')

flags.DEFINE_integer(
    'num_classes', 1001,
    'Number of classes to distinguish')

flags.DEFINE_integer(
    'width', 299,
    'Width of input image')

flags.DEFINE_integer(
    'height', 299,
    'Height of input image')

flags.DEFINE_bool(
    'transpose_enabled', False,
    'Boolean to enable/disable explicit I/O transpose')

flags.DEFINE_bool(
    'log_device_placement', False,
    'Boolean to enable/disable log device placement')

flags.DEFINE_integer(
    'save_summary_steps', 100,
    'Number of steps which must have run before showing summaries.')

flags.DEFINE_integer(
    'save_checkpoints_secs', 1000,
    'Interval (in seconds) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

flags.DEFINE_bool(
    'moving_average', False,
    'Whether to enable moving average computation on variables')

flags.DEFINE_string(
    'preprocessing', 'inception',
    'Preprocessing stage to use: one of inception or vgg')

flags.DEFINE_bool(
    'use_annotated_bbox', False,
    'If true, use annotated bounding box as input to cropping function, '
    'else use full image size')

flags.DEFINE_float(
    'learning_rate_decay', 0.94,
    'Exponential decay rate used in learning rate adjustment')

flags.DEFINE_integer(
    'learning_rate_decay_epochs', 3,
    'Exponential decay epochs used in learning rate adjustment')

flags.DEFINE_bool(
    'display_tensors', False,
    'Whether to dump prediction tensors for comparison')

flags.DEFINE_bool(
    'clear_update_collections', True,
    'Set batchnorm update_collections to None if true, else use default value')

flags.DEFINE_integer(
    'cold_epochs', 2,
    'Number of epochs using cold learning rate')

flags.DEFINE_integer(
    'warmup_epochs', 7,
    'Number of epochs using linearly increasing learning rate')

flags.DEFINE_bool(
    'use_learning_rate_warmup', False,
    'Apply learning rate warmup if true')

# Dataset specific paramenters
flags.DEFINE_bool(
    'prefetch_enabled', True,
    'Boolean to enable/disable prefetching')

flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 8*1024*1024,
    'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'num_files_infeed', 8,
    'Number of training files to read in parallel.')

flags.DEFINE_integer(
    'num_parallel_calls', 64,
    'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

flags.DEFINE_string(
    'precision', 'float32',
    help=('Precision to use; one of: {bfloat16, float32}'))

# inception_preprocessing
flags.DEFINE_float(
    'cb_distortion_range', 0.1, 'Cb distortion range +/-')

flags.DEFINE_float(
    'cr_distortion_range', 0.1, 'Cr distortion range +/-')

flags.DEFINE_boolean(
    'use_fast_color_distort', True,
    'apply fast color/chroma distortion if True, else apply'
    'brightness/saturation/hue/contrast distortion')

FLAGS = flags.FLAGS

# Dataset constants
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

# Random cropping constants
_RESIZE_SIDE_MIN = 300
_RESIZE_SIDE_MAX = 600

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3

WEIGHT_DECAY = 0.00004


# vgg_preprocessing constants
class vgg_constants:

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    _RESIZE_SIDE_MIN = 256
    _RESIZE_SIDE_MAX = 512


class vgg_preprocessing:

    @staticmethod
    def _crop(image, offset_height, offset_width, crop_height, crop_width):
      """Crops the given image using the provided offsets and sizes.

      Note that the method doesn't assume we know the input image size but it does
      assume we know the input image rank.

      Args:
        image: an image of shape [height, width, channels].
        offset_height: a scalar tensor indicating the height offset.
        offset_width: a scalar tensor indicating the width offset.
        crop_height: the height of the cropped image.
        crop_width: the width of the cropped image.

      Returns:
        the cropped (and resized) image.

      Raises:
        InvalidArgumentError: if the rank is not 3 or if the image dimensions are
          less than the crop size.
      """
      original_shape = tf.shape(image)

      rank_assertion = tf.Assert(
          tf.equal(tf.rank(image), 3),
          ['Rank of image must be equal to 3.'])
      with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

      size_assertion = tf.Assert(
          tf.logical_and(
              tf.greater_equal(original_shape[0], crop_height),
              tf.greater_equal(original_shape[1], crop_width)),
          ['Crop size greater than the image size.'])

      offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

      # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
      # define the crop size.
      with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
      return tf.reshape(image, cropped_shape)


    @staticmethod
    def _random_crop(image_list, crop_height, crop_width):
      """Crops the given list of images.

      The function applies the same crop to each image in the list. This can be
      effectively applied when there are multiple image inputs of the same
      dimension such as:

        image, depths, normals = _random_crop([image, depths, normals], 120, 150)

      Args:
        image_list: a list of image tensors of the same dimension but possibly
          varying channel.
        crop_height: the new height.
        crop_width: the new width.

      Returns:
        the image_list with cropped images.

      Raises:
        ValueError: if there are multiple image inputs provided with different size
          or the images are smaller than the crop dimensions.
      """
      if not image_list:
        raise ValueError('Empty image_list.')

      # Compute the rank assertions.
      rank_assertions = []
      for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

      with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
      image_height = image_shape[0]
      image_width = image_shape[1]
      crop_size_assert = tf.Assert(
          tf.logical_and(
              tf.greater_equal(image_height, crop_height),
              tf.greater_equal(image_width, crop_width)),
          ['Crop size greater than the image size.'])

      asserts = [rank_assertions[0], crop_size_assert]

      for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
          shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

      # Create a random bounding box.
      #
      # Use tf.random_uniform and not numpy.random.rand as doing the former would
      # generate random numbers at graph eval time, unlike the latter which
      # generates random numbers at graph definition time.
      with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
      with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
      offset_height = tf.random_uniform(
          [], maxval=max_offset_height, dtype=tf.int32)
      offset_width = tf.random_uniform(
          [], maxval=max_offset_width, dtype=tf.int32)

      return [vgg_preprocessing._crop(image, offset_height, offset_width,
                    crop_height, crop_width) for image in image_list]


    @staticmethod
    def _central_crop(image_list, crop_height, crop_width):
      """Performs central crops of the given image list.

      Args:
        image_list: a list of image tensors of the same dimension but possibly
          varying channel.
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.

      Returns:
        the list of cropped images.
      """
      outputs = []
      for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(vgg_preprocessing._crop(image, offset_height, offset_width,
                             crop_height, crop_width))
      return outputs


    @staticmethod
    def _mean_image_subtraction(image, means):
      """Subtracts the given means from each image channel.

      For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

      Note that the rank of `image` must be known.

      Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.

      Returns:
        the centered image.

      Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
          than three or if the number of channels in `image` doesn't match the
          number of values in `means`.
      """
      if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
      num_channels = image.get_shape().as_list()[-1]
      if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

      channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
      for i in range(num_channels):
        channels[i] -= means[i]
      return tf.concat(axis=2, values=channels)


    @staticmethod
    def _smallest_size_at_least(height, width, smallest_side):
      """Computes new shape with the smallest side equal to `smallest_side`.

      Computes new shape with the smallest side equal to `smallest_side` while
      preserving the original aspect ratio.

      Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        smallest_side: A python integer or scalar `Tensor` indicating the size of
          the smallest side after resize.

      Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: and int32 scalar tensor indicating the new width.
      """
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      height = tf.to_float(height)
      width = tf.to_float(width)
      smallest_side = tf.to_float(smallest_side)

      scale = tf.cond(tf.greater(height, width),
                      lambda: smallest_side / width,
                      lambda: smallest_side / height)
      new_height = tf.to_int32(height * scale)
      new_width = tf.to_int32(width * scale)
      return new_height, new_width


    @staticmethod
    def _aspect_preserving_resize(image, smallest_side):
      """Resize images preserving the original aspect ratio.

      Args:
        image: A 3-D image `Tensor`.
        smallest_side: A python integer or scalar `Tensor` indicating the size of
          the smallest side after resize.

      Returns:
        resized_image: A 3-D tensor containing the resized image.
      """
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      shape = tf.shape(image)
      height = shape[0]
      width = shape[1]
      new_height, new_width = vgg_preprocessing._smallest_size_at_least(height, width, smallest_side)
      image = tf.expand_dims(image, 0)
      resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                               align_corners=False)
      resized_image = tf.squeeze(resized_image)
      resized_image.set_shape([None, None, 3])
      return resized_image


    @staticmethod
    def preprocess_for_train(image,
                             output_height,
                             output_width,
                             resize_side_min=vgg_constants._RESIZE_SIDE_MIN,
                             resize_side_max=vgg_constants._RESIZE_SIDE_MAX):
      """Preprocesses the given image for training.

      Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
          aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
          aspect-preserving resizing.

      Returns:
        A preprocessed image.
      """
      resize_side = tf.random_uniform(
          [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

      image = vgg_preprocessing._aspect_preserving_resize(image, resize_side)
      image = vgg_preprocessing._random_crop([image], output_height, output_width)[0]
      image.set_shape([output_height, output_width, 3])
      image = tf.to_float(image)
      image = tf.image.random_flip_left_right(image)
      return vgg_preprocessing._mean_image_subtraction(image,
              [vgg_constants._R_MEAN, vgg_constants._G_MEAN,
                  vgg_constants._B_MEAN])

    @staticmethod
    def preprocess_for_eval(image, output_height, output_width, resize_side):
      """Preprocesses the given image for evaluation.

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side: The smallest side of the image for aspect-preserving resizing.

      Returns:
        A preprocessed image.
      """
      image = vgg_preprocessing._aspect_preserving_resize(image, resize_side)
      image = vgg_preprocessing._central_crop([image], output_height, output_width)[0]
      image.set_shape([output_height, output_width, 3])
      image = tf.to_float(image)
      return vgg_preprocessing._mean_image_subtraction(image,
              [vgg_constants._R_MEAN, vgg_constants._G_MEAN,
                  vgg_constants._B_MEAN])

    @staticmethod
    def preprocess_image(image, output_height, output_width, is_training=False,
                         resize_side_min=vgg_constants._RESIZE_SIDE_MIN,
                         resize_side_max=vgg_constants._RESIZE_SIDE_MAX):
      """Preprocesses the given image.

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
          `False` otherwise.
        resize_side_min: The lower bound for the smallest side of the image for
          aspect-preserving resizing. If `is_training` is `False`, then this value
          is used for rescaling.
        resize_side_max: The upper bound for the smallest side of the image for
          aspect-preserving resizing. If `is_training` is `False`, this value is
          ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max].

      Returns:
        A preprocessed image.
      """
      if is_training:
        image = vgg_preprocessing.preprocess_for_train(image, output_height, output_width,
                                     resize_side_min, resize_side_max)
      else:
        image = vgg_preprocessing.preprocess_for_eval(image, output_height, output_width,
                                    resize_side_min)
      # Scale to (-1,1). TODO(currutia): check whether this is actually needed
      image = tf.multiply(image, 1. / 128.)
      return image


class inception_preprocessing:

    @staticmethod
    def apply_with_random_selector(x, func, num_cases):
      """Computes func(x, sel), with sel sampled from [0...num_cases-1].

      Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

      Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
      """
      sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
      # Pass the real x only to one of the func calls.
      return control_flow_ops.merge([
          func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
          for case in range(num_cases)])[0]

    @staticmethod
    def distort_color_fast(image, scope=None):
      """Distort the color of a Tensor image.

      Distort brightness and chroma values of input image

      Args:
        image: 3-D Tensor containing single image in [0, 1].
        scope: Optional scope for name_scope.
      Returns:
        3-D Tensor color-distorted image on range [0, 1]
      """
      with tf.name_scope(scope, 'distort_color', [image]):
        br_delta = random_ops.random_uniform([], -32./255., 32./255., seed=None)
        cb_factor = random_ops.random_uniform(
            [], -FLAGS.cb_distortion_range, FLAGS.cb_distortion_range, seed=None)
        cr_factor = random_ops.random_uniform(
            [], -FLAGS.cr_distortion_range, FLAGS.cr_distortion_range, seed=None)

        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0., 1.)

        return image

    @staticmethod
    def distorted_bounding_box_crop(image,
                                    bbox,
                                    min_object_covered=0.1,
                                    aspect_ratio_range=(3./4., 4./3.),
                                    area_range=(0.05, 1.0),
                                    max_attempts=100,
                                    scope=None):
      """Generates cropped_image using a one of the bboxes randomly distorted.

      See `tf.image.sample_distorted_bounding_box` for more documentation.

      Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
          image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding box
          supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
          image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
        scope: Optional scope for name_scope.
      Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
      """
      with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox

    @staticmethod
    def preprocess_for_train(image, height, width, bbox,
                             fast_mode=True,
                             scope=None,
                             add_image_summaries=True):
      """Distort one image for training a network.

      Distorting images provides a useful technique for augmenting the data
      set during training in order to make the network invariant to aspects
      of the image that do not effect the label.

      Additionally it would create image_summaries to display the different
      transformations applied to the image.

      Args:
        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
          [0, 1], otherwise it would converted to tf.float32 assuming that the range
          is [0, MAX], where MAX is largest positive representable number for
          int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
        height: integer
        width: integer
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax].
        fast_mode: Optional boolean, if True avoids slower transformations (i.e.
          bi-cubic resizing, random_hue or random_contrast).
        scope: Optional scope for name_scope.
        add_image_summaries: Enable image summaries.
      Returns:
        3-D float Tensor of distorted image used for training with range [-1, 1].
      """
      with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
          bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                             dtype=tf.float32,
                             shape=[1, 1, 4])
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if add_image_summaries:
          # Each bounding box has shape [1, num_boxes, box coords] and
          # the coordinates are ordered [ymin, xmin, ymax, xmax].
          image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                        bbox)
          tf.summary.image('image_with_bounding_boxes', image_with_box)

        distorted_image, distorted_bbox = inception_preprocessing.distorted_bounding_box_crop(image, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        if add_image_summaries:
          image_with_distorted_box = tf.image.draw_bounding_boxes(
              tf.expand_dims(image, 0), distorted_bbox)
          tf.summary.image('images_with_distorted_bounding_box',
                           image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = inception_preprocessing.apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases)

        if add_image_summaries:
          tf.summary.image('cropped_resized_image',
                           tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors. There are 1 or 4 ways to do it.
        if FLAGS.use_fast_color_distort:
          distorted_image = inception_preprocessing.distort_color_fast(distorted_image)
        else:
          num_distort_cases = 1 if fast_mode else 4
          distorted_image = inception_preprocessing.apply_with_random_selector(
              distorted_image,
              lambda x, ordering: distort_color(x, ordering, fast_mode),
              num_cases=num_distort_cases)

        if add_image_summaries:
          tf.summary.image('final_distorted_image',
                           tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


    @staticmethod
    def preprocess_for_eval(image, height, width,
                            central_fraction=0.875, scope=None):
      """Prepare one image for evaluation.

      If height and width are specified it would output an image with that size by
      applying resize_bilinear.

      If central_fraction is specified it would crop the central fraction of the
      input image.

      Args:
        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
          [0, 1], otherwise it would converted to tf.float32 assuming that the range
          is [0, MAX], where MAX is largest positive representable number for
          int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
        height: integer
        width: integer
        central_fraction: Optional Float, fraction of the image to crop.
        scope: Optional scope for name_scope.
      Returns:
        3-D float Tensor of prepared image.
      """
      with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
          image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
          # Resize the image to the specified height and width.
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(image, [height, width],
                                           align_corners=False)
          image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image.set_shape([height, width, 3])
        return image

    @staticmethod
    def preprocess_image(image, output_height, output_width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         add_image_summaries=False):
      """Pre-process one image for training or evaluation.

      Args:
        image: 3-D Tensor [height, width, channels] with the image. If dtype is
          tf.float32 then the range should be [0, 1], otherwise it would converted
          to tf.float32 assuming that the range is [0, MAX], where MAX is largest
          positive representable number for int(8/16/32) data type (see
          `tf.image.convert_image_dtype` for details).
        output_height: integer, image expected height.
        output_width: integer, image expected width.
        is_training: Boolean. If true it would transform an image for train,
          otherwise it would transform it for evaluation.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        fast_mode: Optional boolean, if True avoids slower transformations.
        add_image_summaries: Enable image summaries.

      Returns:
        3-D float Tensor containing an appropriately scaled image

      Raises:
        ValueError: if user does not provide bounding box
      """
      if is_training:
        return inception_preprocessing.preprocess_for_train(image,
                                    output_height, output_width, bbox,
                                    fast_mode,
                                    add_image_summaries=add_image_summaries)
      else:
        return inception_preprocessing.preprocess_for_eval(image,
                                    output_height, output_width)


def preprocess_raw_bytes(image_bytes, is_training=False, bbox=None):
  """Preprocesses a raw JPEG image.

  This implementation is shared in common between train/eval pipelines,
  and when serving the model.

  Args:
    image_bytes: A string Tensor, containing the encoded JPEG.
    is_training: Whether or not to preprocess for training.
    bbox:        In inception preprocessing, this bbox can be used for cropping.

  Returns:
    A 3-Tensor [height, width, RGB channels] of type float32.
  """

  image = tf.image.decode_jpeg(image_bytes, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if FLAGS.preprocessing == 'vgg':
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        resize_side_min=_RESIZE_SIDE_MIN,
        resize_side_max=_RESIZE_SIDE_MAX)
  elif FLAGS.preprocessing == 'inception':
    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        bbox=bbox)
  else:
    assert False, 'Unknown preprocessing type: %s' % FLAGS.preprocessing
  return image


class InputPipeline(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
  """

  def __init__(self, is_training, data_dir, use_bfloat16):
    self.is_training = is_training
    self.data_dir = data_dir
    self.use_bfloat16 = use_bfloat16

  def dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None
    if FLAGS.use_annotated_bbox:
      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

    image = features['image/encoded']
    image = preprocess_raw_bytes(image, is_training=self.is_training, bbox=bbox)
    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    if self.use_bfloat16:
      image = tf.cast(image, tf.bfloat16)

    return image, label

  def dataset_iterator(self, batch_size, shuffle):
    """Constructs a real-data iterator over batches for train or eval.

    Args:
      batch_size: The effective batch size.
      shuffle: Whether or not to shuffle the data.

    Returns:
      A tf.data iterator.
    """
    file_pattern = os.path.join(self.data_dir, 'train-*'
                                if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(
          filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length=FLAGS.num_files_infeed, sloppy=True))

    if shuffle and FLAGS.followup_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(buffer_size=FLAGS.followup_shuffle_buffer_size)

    dataset = dataset.map(
        self.dataset_parser, num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training

    return dataset.make_one_shot_iterator()

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A (images, labels) tuple of `Tensor`s for a batch of samples.
    """
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      images, labels = self.dataset_iterator(batch_size,
                                             self.is_training).get_next()
    else:
      images = tf.random_uniform(
          [batch_size, FLAGS.height, FLAGS.width, 3], minval=-1, maxval=1)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)

    images = tensor_transform_fn(images, params['output_perm'])
    return images, labels


def image_serving_input_fn():
  """Serving input fn for raw images.

  This function is consumed when exporting a SavedModel.

  Returns:
    A ServingInputReceiver capable of serving MobileNet predictions.
  """

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      preprocess_raw_bytes, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def tensor_transform_fn(data, perm):
  """Transpose function.

  This function is used to transpose an image tensor on the host and then
  perform an inverse transpose on the TPU. The transpose on the TPU gets
  effectively elided thus voiding any associated computational cost.

  NOTE: Eventually the compiler will be able to detect when this kind of
  operation may prove beneficial and perform these types of transformations
  implicitly, voiding the need for user intervention

  Args:
    data: Tensor to be transposed
    perm: New ordering of dimensions

  Returns:
    Transposed tensor
  """
  if FLAGS.transpose_enabled:
    return tf.transpose(data, perm)
  return data


def inception_model_fn(features, labels, mode, params):
  """Inception v3 model using Estimator API."""
  num_classes = FLAGS.num_classes
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  is_eval = (mode == tf.estimator.ModeKeys.EVAL)

  if isinstance(features, dict):
    features = features['feature']

  features = tensor_transform_fn(features, params['input_perm'])

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network():
    if FLAGS.precision == 'bfloat16':
      with tf.contrib.tpu.bfloat16_scope():
        logits, end_points = inception.inception_v3(
            features,
            num_classes,
            is_training=is_training)
      logits = tf.cast(logits, tf.float32)
    elif FLAGS.precision == 'float32':
      logits, end_points = inception.inception_v3(
          features,
          num_classes,
          is_training=is_training)
    return logits, end_points

  if FLAGS.clear_update_collections:
    # updates_collections must be set to None in order to use fused batchnorm
    with arg_scope(inception.inception_v3_arg_scope(
        weight_decay=0.0,
        batch_norm_decay=BATCH_NORM_DECAY,
        batch_norm_epsilon=BATCH_NORM_EPSILON,
        updates_collections=None)):
      logits, end_points = build_network()
  else:
    with arg_scope(inception.inception_v3_arg_scope(
        batch_norm_decay=BATCH_NORM_DECAY,
        batch_norm_epsilon=BATCH_NORM_EPSILON)):
      logits, end_points = build_network()

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  if mode == tf.estimator.ModeKeys.EVAL and FLAGS.display_tensors and (
      not FLAGS.use_tpu):
    with tf.control_dependencies([
        tf.Print(
            predictions['classes'], [predictions['classes']],
            summarize=FLAGS.eval_batch_size,
            message='prediction: ')
    ]):
      labels = tf.Print(
          labels, [labels], summarize=FLAGS.eval_batch_size, message='label: ')

  one_hot_labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.int32)

  if 'AuxLogits' in end_points:
    tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=tf.cast(end_points['AuxLogits'], tf.float32),
        weights=0.4,
        label_smoothing=0.1,
        scope='aux_loss')

  tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)

  losses = tf.add_n(tf.losses.get_losses())
  l2_loss = []
  for v in tf.trainable_variables():
    if 'BatchNorm' not in v.name and 'weights' in v.name:
      l2_loss.append(tf.nn.l2_loss(v))
  loss = losses + WEIGHT_DECAY * tf.add_n(l2_loss)

  initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
  if FLAGS.use_learning_rate_warmup:
    # Adjust initial learning rate to match final warmup rate
    warmup_decay = FLAGS.learning_rate_decay**(
        (FLAGS.warmup_epochs + FLAGS.cold_epochs) /
        FLAGS.learning_rate_decay_epochs)
    adj_initial_learning_rate = initial_learning_rate * warmup_decay

  final_learning_rate = 0.0001 * initial_learning_rate

  host_call = None
  train_op = None
  if is_training:
    batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    global_step = tf.train.get_or_create_global_step()
    current_epoch = tf.cast(
        (tf.cast(global_step, tf.float32) / batches_per_epoch), tf.int32)

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=int(FLAGS.learning_rate_decay_epochs * batches_per_epoch),
        decay_rate=FLAGS.learning_rate_decay,
        staircase=True)

    if FLAGS.use_learning_rate_warmup:
      wlr = 0.1 * adj_initial_learning_rate
      wlr_height = tf.cast(
          0.9 * adj_initial_learning_rate /
          (FLAGS.warmup_epochs + FLAGS.learning_rate_decay_epochs - 1),
          tf.float32)
      epoch_offset = tf.cast(FLAGS.cold_epochs - 1, tf.int32)
      exp_decay_start = (FLAGS.warmup_epochs + FLAGS.cold_epochs +
                         FLAGS.learning_rate_decay_epochs)
      lin_inc_lr = tf.add(
          wlr, tf.multiply(
              tf.cast(tf.subtract(current_epoch, epoch_offset), tf.float32),
              wlr_height))
      learning_rate = tf.where(
          tf.greater_equal(current_epoch, FLAGS.cold_epochs),
          (tf.where(tf.greater_equal(current_epoch, exp_decay_start),
                    learning_rate, lin_inc_lr)),
          wlr)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, final_learning_rate, name='learning_rate')

    if FLAGS.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    elif FLAGS.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'RMS':
      tf.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)
    if FLAGS.moving_average:
      ema = tf.train.ExponentialMovingAverage(
          decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
      variables_to_average = (
          tf.trainable_variables() + tf.moving_average_variables())
      with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
        train_op = ema.apply(variables_to_average)

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(loss, [1])
    lr_t = tf.reshape(learning_rate, [1])
    ce_t = tf.reshape(current_epoch, [1])

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, loss, lr, ce):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide them as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        with summary.create_file_writer(FLAGS.model_dir).as_default():
          with summary.always_record_summaries():
            summary.scalar('loss', tf.reduce_mean(loss), step=gs)
            summary.scalar('learning_rate', tf.reduce_mean(lr), step=gs)
            summary.scalar('current_epoch', tf.reduce_mean(ce), step=gs)

            return summary.all_summary_ops()

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  eval_metrics = None
  if is_eval:
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch, ]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'accuracy': top_1_accuracy,
          'accuracy@5': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


class LoadEMAHook(tf.train.SessionRunHook):
  """Hook to load exponential moving averages into corresponding variables."""

  def __init__(self, model_dir):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


def main(unused_argv):
  del unused_argv  # Unused

  start = time.time()

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
      'Invalid value for --precision flag; must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', FLAGS.precision)

  params = {
      'input_perm': [0, 1, 2, 3],
      'output_perm': [0, 1, 2, 3],
  }

  batch_axis = 0
  if FLAGS.transpose_enabled:
    params['input_perm'] = [3, 0, 1, 2]
    params['output_perm'] = [1, 2, 3, 0]
    batch_axis = 3

  if FLAGS.eval_total_size > 0:
    eval_size = FLAGS.eval_total_size
  else:
    eval_size = _NUM_EVAL_IMAGES
  eval_steps = eval_size // FLAGS.eval_batch_size

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                FLAGS.iterations)

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     FLAGS.eval_batch_size)

  per_host_input_for_training = (
      FLAGS.num_shards <= 8 if FLAGS.mode == 'train' else True)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=per_host_input_for_training))

  inception_classifier = tf.contrib.tpu.TPUEstimator(
      model_fn=inception_model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  use_bfloat16 = FLAGS.precision == 'bfloat16'
  imagenet_train = InputPipeline(
      is_training=True,
      data_dir=FLAGS.data_dir,
      use_bfloat16=use_bfloat16)
  imagenet_eval = InputPipeline(
      is_training=False,
      data_dir=FLAGS.data_dir,
      use_bfloat16=use_bfloat16)

  if FLAGS.moving_average:
    eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
  else:
    eval_hooks = []

  if FLAGS.mode == 'eval':
    # Run evaluation when there is a new checkpoint
    for checkpoint in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # Includes compilation time
        eval_results = inception_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info(
            'Eval results: %s. Elapsed seconds: %d', eval_results, elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(checkpoint).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break
      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', checkpoint)

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=imagenet_train.input_fn, steps=FLAGS.train_steps_per_eval)

      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    tf.logging.info('Starting training ...')
    inception_classifier.train(
        input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)

  if FLAGS.export_dir is not None:
    tf.logging.info('Starting to export model.')
    inception_classifier.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=image_serving_input_fn)

  end = time.time()
  print("Elapsed time = %lf s" % (end - start))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
