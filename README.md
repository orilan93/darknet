![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# YOLO Fish #
This is a fork of darknet that aims at object detection of fish under water as a part of a [master thesis](wip) at University of Agder. It utalizes word tree to do hirarcical classification of fish species. Various changes have been made to darknet to optimze it for this function. Weights and a new dataset for fish detection can be downloaded from [here](wip). Yolo Fish achives 91.8 mAP on this dataset with an inference time of 26.4ms on a Tesla V100. This was achived with the yolov3\_fish\_tree_42000 weigths.


The Python folder conatins various scripts for visualization. fish_detector.py can be used to detect fish in a video. This also utalizes a modified version of [SORT](https://github.com/abewley/sort) to improve detecions.

The scripts folder contains various scipts, like validation scripts for calulating mAP with hiracical classification and scripts used for creatin of the dataset.

From darknet read me:
# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

