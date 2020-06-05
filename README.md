# YOLO Fish #
This is a fork of Darknet that aims for object detection of fish under water as a part of a [master thesis](wip) at University of Agder. It utilizes word tree to perform hierarchical classification of fish species. Various changes have been made to darknet to optimze it for this function. The weights and dataset for fish detection can be downloaded from [here](http://dx.doi.org/10.17632/b4kcw9r32n.1). Yolo Fish achives 91.8 mAP on this dataset with an inference time of 26.4ms on a Tesla V100. This was achived with the yolov3\_fish\_tree_42000 weigths file.

The Python folder contains various scripts for visualization. fish_detector.py can be used to detect fish in a video. This also utilizes a modified version of [SORT](https://github.com/abewley/sort) to improve detecions.

The scripts folder contains various scripts, like validation scripts for calulating mAP with hirarchical classification and scripts used for creation of the dataset.

From darknet read me:
# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

