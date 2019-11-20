"""A demo to classify image."""
import argparse
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import time

import sys, os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../../../guestlib/python'))
#from hook_tflite import *


# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument(
      '--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=True)
  args = parser.parse_args()

  start = time.time()

  # Prepare labels.
  labels = ReadLabelFile(args.label)
  # Initialize engine.
  engine = ClassificationEngine(args.model)
  # Run inference.
  img = Image.open(args.image)
  for result in engine.ClassifyWithImage(img, top_k=3):
    print ('---------------------------')
    print (labels[result[0]])
    print ('Score : ', result[1])

  end = time.time()
  print("Elapsed time = %lf ms" % ((end - start) * (10 ** 3)))

if __name__ == '__main__':
  main()
