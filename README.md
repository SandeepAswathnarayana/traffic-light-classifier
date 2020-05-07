# traffic-light-classifier  

## Overview
The car would be operated on a test track and required to follow waypoints in a large circle. If the light is green, then the car is required to continue driving around the circle. If the light is red, then the car is required to stop and wait for the light to turn green. This is a part of the **Perception** process, one among the three major steps in the system integration project.  

For traffic light detection and classification we decided to use an SSD (Single Shot MultiBox Detector) network as the purpose of an SSD is detect the location and classify the detected object in one pass through the network.  

Due to the limited amount of data available to train the network the decision was made to take a pre-trained network and transfer learn the network on the available simulated and real datasets provided by Udacity. The chosen network was pre-trained with the COCO dataset.

Transfer learning was achieved using the Object Detection API provided by Tensorflow. For simulated data the network was trained on the provided data by Udacity, however real data provided by Udacity was supplemented with a dataset of labelled traffic lights provided by Bosch. This dataset can be found [here](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset).  

Get models from tensorflows model repository that are compatible with tensorflow 1.4
```
git clone https://github.com/tensorflow/models.git
cd models
git checkout f7e99c0
```

Test the installation
```
cd research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

## Training the model
### Data

#### Carla
Images with labeled traffic lights can be found on

1.  [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
2.  [LaRA Traffic Lights Recognition Dataset](http://www.lara.prd.fr/benchmarks/trafficlightsrecognition)
3.  Udacity's ROSbag file from Carla
4.  Traffic lights from Udacity's simulator

#### Simulation
Training images for simulation can be found downloaded from Vatsal Srivastava's dataset and Alex Lechners's dataset. The images are already labeled and a  [TFRecord file](https://github.com/alex-lechner/Traffic-Light-Classification#23-create-a-tfrecord-file)  is provided as well:

1.  [Vatsal's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)
2.  [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)

## Model 

The model "SSD Mobilenet V1" was used for classification of the Bosch Small Traffic Lights Dataset. See the performance on this page https://github.com/bosch-ros-pkg/bstld .

The model "SSD Inception V2" seems to perform better at the expense of speed. See [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to see the performance comparison.

## Download
Switch to the models directory and download 
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
```
Extract them here
```
tar -xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz
```

## Model Configuration

```
mkdir config
```

Copy the chosen models to config
```
cp models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config config/
cp models/research/object_detection/samples/configs/ssd_inception_v2_coco.config config/
```

#### Configuration on Udacity Simulation dataset for "SSD Inception V2"

Configuration taken from https://github.com/bosch-ros-pkg/bstld/blob/master/tf_object_detection/configs/ssd_mobilenet_v1.config

1.  Change  `num_classes: 90`  to the number of labels in your  `label_map.pbtxt`. This will be  `num_classes: 4`
2.  Set the default  `max_detections_per_class: 100`  and  `max_total_detections: 300`  values to a lower value for example  `max_detections_per_class: 25`  and  `max_total_detections: 100`
3.  Change  `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`  to the directory where your downloaded model is stored e.g.:  `fine_tune_checkpoint: "models/ssd_inception_v2_coco_2018_01_28/model.ckpt"`
4.  Set  `num_steps: 200000`  down to  `num_steps: 20000`
5.  Change the  `PATH_TO_BE_CONFIGURED`  placeholders in  `input_path`  and  `label_map_path`  to your .record file(s) and  `label_map.pbtxt`

## Train

Start Training with 
```
python train.py --logtostderr --train_dir=./models/train-ssd-inception-simulation --pipeline_config_path=./config/ssd_inception_v2_coco-simulator.config
```

## Freeze

Execute:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_inception_v2_coco-simulator.config --trained_checkpoint_prefix ./models/train-ssd-inception-simulation/model.ckpt-20000 --output_directory models/frozen-ssd_inception-simulation
```

Original:
```
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          optimize_tensor_layout=True)
```

Modified:
```
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
```

You can find the frozen graph `frozen_inference_graph.pb` in the [Google Drive](https://drive.google.com/file/d/1QMiQcQSSGDg1GJ4mLKaypd1uKZV3AhUD/view?usp=sharing)


## Detection
The [object detection tutorial - a jupyter notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) walks you through the steps

Most of the stes from above are incorporated into detector.py

Make sure that the following variables are set according to your needs:

```
MODEL_NAME = 'frozen-ssd_inception-simulation'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'udacity_label_map.pbtxt')
PATH_TO_TEST_IMAGES_DIR = 'test_images/simulation'
PATH_TO_TEST_IMAGES_OUTPUTDIR = 'test_images_results/simulation'
```

Execute 
```
python detector.py
```
The resulting images can be found in the directory `PATH_TO_TEST_IMAGES_OUTPUTDIR`
