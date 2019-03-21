#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
import random
import IPython.display as display

tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Before you start any training, you'll need a set of images to teach the network about the new classes you want to recognize.
# We've created an archive of creative-commons licensed flower photos to use initially.
data_root = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_root = pathlib.Path(data_root)
print(data_root)

# After downloading 218MB, you should now have a copy of the flower photos available:
for item in data_root.iterdir():
  print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count

all_image_paths[:10]

# Now let's have a quick look at a couple of the images, so we know what we're dealing with:
attributions = (data_root/"LICENSE.txt").read_text(encoding="utf8").splitlines()[4:] # Issues with read_text
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
    
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()
