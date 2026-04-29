
# Samsung MoE Vision System

Modular on-device computer vision system using:

- Mixture of Experts (MoE)
- Scene Context Awareness
- TFLite + OpenCV / MediaPipe

Supports:
TV, Refrigerator, AC, Washing Machine, Microwave,
Dishwasher, Robot Vacuum, Air Purifier

## Project Structure (High-Level)
- apps/: MediaPipe and OpenCV demos
- datasets/: TFRecord datasets and label maps for climate, display, kitchen
- models/: backbone, experts, router, scene_classifier
- pipelines/: training and inference modules
- postprocess/: postprocessing utilities
- scripts/: project bootstrap utilities
- outputs/: training/eval outputs
- tf_env/: Python virtual environment
- tf_models/: TensorFlow Models repository (Object Detection API)
- utils/: preprocessing, NMS, and shared helpers

## Work Log (What Has Been Done)
- Verified virtual environment in tf_env/ and configured Python 3.10.11
- Confirmed TensorFlow 2.13.0 and TF Object Detection API dependencies in the venv
- Verified backbone checkpoint exists at:
	- models/backbone/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0
- Created expert training configs:
	- models/experts/kitchen/pipeline.config (3 classes)
	- models/experts/display/pipeline.config (1 class)
	- models/experts/climate/pipeline.config (2 classes)

## Exact Models and Pipelines
### Backbone
- Model family: SSD MobileNet V2 FPNLite 320x320 (COCO17)
- Checkpoint used for all experts:
  - models/backbone/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0

### Expert Pipelines (TF Object Detection API)
Common model settings:
- model: ssd
- feature_extractor.type: ssd_mobilenet_v2_fpn_keras
- input size: 320x320
- optimizer: momentum with cosine decay (base 0.04, 5000 steps, warmup 200)

Kitchen expert:
- pipeline: models/experts/kitchen/pipeline.config
- num_classes: 3
- train TFRecord: datasets/kitchen/train.tfrecord
- eval TFRecord: datasets/kitchen/valid.tfrecord
- label map: datasets/kitchen/label_map.pbtxt

Display expert:
- pipeline: models/experts/display/pipeline.config
- num_classes: 1
- train TFRecord: datasets/display/train.tfrecord
- eval TFRecord: datasets/display/valid.tfrecord
- label map: datasets/display/label_map.pbtxt

Climate expert:
- pipeline: models/experts/climate/pipeline.config
- num_classes: 2
- train TFRecord: datasets/climate/train.tfrecord
- eval TFRecord: datasets/climate/valid.tfrecord
- label map: datasets/climate/label_map.pbtxt

## Training Runs
### Kitchen Expert
- Training completed to step 5000
- Final training total loss: 0.37666106
- One-time eval completed on 96 validation images
- Eval results saved in models/experts/kitchen/README.md

### Display Expert
- Training started using models/experts/display/pipeline.config
- Completion not yet observed

### Climate Expert
- Training started using models/experts/climate/pipeline.config
- Completion not yet observed

## Commands Used
### Train (example)
```powershell
$env:PYTHONPATH="$PWD\tf_models\research;$PWD\tf_models\research\slim"; c:/Users/admin/Desktop/SAM/SAMSUNG/projects/tf_env/Scripts/python.exe tf_models/research/object_detection/model_main_tf2.py --model_dir=models/experts/kitchen --pipeline_config_path=models/experts/kitchen/pipeline.config
```

### Eval (example)
```powershell
$env:PYTHONPATH="$PWD\tf_models\research;$PWD\tf_models\research\slim"; c:/Users/admin/Desktop/SAM/SAMSUNG/projects/tf_env/Scripts/python.exe tf_models/research/object_detection/model_main_tf2.py --model_dir=models/experts/kitchen --pipeline_config_path=models/experts/kitchen/pipeline.config --checkpoint_dir=models/experts/kitchen --run_once=true
```
