## Image Analysis for Detection of Potential Mosquitoes Breeding Sites

### Abstract
The goal is to create a system that uses deep learning to detect whether or not a specific image contains potencial sites for mosquitoes reproduction, using object detection. This can be applied for preventing diseases spread such as Dengue or Zika. 

Risky sites include areas with still water, for example: water tanks, plant pots, empty bottles and other recipients, car tires, rubble areas. 

### Author
Bruno Gabriel Cavalcante Lima, <bgcl@ic.ufal.br>

### Folder contents:
- /db - image database
- /exp - experiments
- /ref - references, papers, links, files, videos, etc
- /rep - report, paper containing all pertinent report
- /res - results, data, graph, etc
- /src - source files, code here

### Specific Objectives
The goal of the project is to train a object detection system using TensorFlow. The project consists of:
1. Collecting images for each class (tire, rubble, bottles, water tank, plant pot)
2. Create the images bounding box (one .xml file for each image collected)
3. Divide each class into training and test
3. Generate the .tfrecord (TensorFlow default extension for dataset).
4. Choose a network model and configuration
5. Train the model and export the frozen inference graph (.pb file)
6. Test the results

### Details for each step
1. Used two main sources for getting images: GoogleImages and [ImageNet](http://www.image-net.org/). The aim was to group approx. 300 images for each class.
   In order to download from Google, there is this [nice script](https://github.com/atif93/google_image_downloader) from atif93. To download from a URL list (to be used with ImageNet), you can use [getImagesFromURLs.py](https://github.com/tfvieira/vazazika/blob/master/src/getImgsFromURLs.py) from tfvieira. I added the functions for directory cleaning and images choosing within a folder.

2. To create the annotations, Tzutalin's [LabelImage](https://github.com/tzutalin/labelImg) software can be used.  After you define the classes you have, it generates the .xml once you specify the bounding boxes at for each image. In later steps, you will need to have your class-labels.pbtxt. Samples of classlabels can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection/data).

3. To split dataset into training_set and test_set folders, we have this [train_and_test_splitter.py](https://github.com/bglima/PDI-Project/blob/master/src/train_and_test_splitter.py). It shuffles the data and divide into two folders, with the fraction you specify for training.

4. The .xml files can be converted to .csv, and then to .record using the scripts present in [Racoon Object Detection](https://github.com/datitran/raccoon_dataset) repository.

5. In order to train your network, you first need to have its configuration. 
Configuration samples: [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). Also, if you choose to use a pre-existing model, you can use some pre-trained checkpoints and its features. Pre-trained models: [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The goal is to reach a loss that is smaller than 1.
   After you get to it, you need to export a frozen inference graph. There is [a script](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) that does this in TensorFlow object detection repository. 

6. In the same repository you can find a [Jupyter Notebook script](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) showing all the necessary steps to test your trained network.

In the configuration samples, you basically need to search for the PATH_TO_BE_CONFIGURED in the .config files. They are:
* fine_tune_checkpoint - If you are using pre-trained models
* train_input_reader (input_path) - Path to the train.record generated in step 4.
* train_input_reader (label_map_path) - Path to the class-labels.pbtxt for your training model.
* eval_input_reader (input_path) - Path to test.record
* eval_input_reader (label_map_path) - Path to the class-labels.pbtxt for your testing model. Can be the same as for train.

### First results

We tested SSD MobileNet model, but we got better results with Faster RCNN Resnet COCO 50. 

Configuration files used for the training can be found in [src/training002](https://github.com/bglima/PDI-Project/tree/master/src/training002) Initially we trained with only one class: tire. This way we could learn how the training proccess works. The first results are avaliable in [res/training002_tire_test](https://github.com/bglima/PDI-Project/tree/master/res/training002_tire_test). Image tire_test_01 show images from test_set, while tire_test_02 show new images, not used in training nor test. 

### Status

Generating annotations for rubble class.

### References

[Pre-trained object-detect model checkpoints](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

[Samples for object-detect configuration files](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

[Samples for object-detect class labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data)

[How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

[Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

[Object detection with TensorFlow: Black and white pawns](https://www.oreilly.com/ideas/object-detection-with-tensorflow)

[LabelImage annotation software](https://github.com/tzutalin/labelImg)
