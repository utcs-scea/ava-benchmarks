#!/bin/bash

python3 demo/object_detection.py \
--model test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite \
--input test_data/face.jpg \
--output ~/detection_results.jpg
