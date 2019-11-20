import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


import time
start = time.time()

data = np.random.rand(128,244,244,3)

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4")
])
m.build([128,244,244,3])

for i in range(0, 16):
    m.predict(data)

end = time.time()
print("in-program elapsed time = %lf ms" % ((end - start) * (10 ** 3)))
