#!/bin/bash
#

docker build -t semantic_segmentation_demo .

docker run $PWD:/opt/semantic_segmentation_demo -it --memory-swap -1 --memory 32768m  semantic_segmentation_demo
