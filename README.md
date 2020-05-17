# Object-Detection-on-Video

In this project we have performed object detection on a video using the single shot multibox detector model in pytorch. We have made use of the pytorch implementation of the ssd model Max deGroot. Github - ![amdegroot](https://github.com/amdegroot).
Pytorch implementation of the ssd model can be found ![here](https://github.com/amdegroot/ssd.pytorch).

To make the predictions on the image frames we have made use of pretrained weights.

## Repository Structure / File Description :-
*  The repository contains the data folder , this folder contains the
   Classes BaseTransform that will do the required transformations
   so that the input images will be compatible with the neural net that is implemented in the ssd.py file.
   
* Layers is a folder that contains tools for detection and multibox parts of the ssd
   
* We have the ssd.py file in the working folder that contains the architecture of the single shot multibox detector this is taken from the above mentioned repository. Its the pytorch implementation of the ssd.

* The pretrained weights file that is used to load the weights can be found ![here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth). We will simply load this file
   to do the object detection using the pretrained model.
   We will transfer these weights to the model we implement.

## Libraries/ Modules Used :-

* **torch.autograd** contains Variable class that will be used to  convert tensors  into some torch variables that will contain both the tensor and the gradient and this torch variable containing both gradient and tensor will be like one element of the graph.
   
* **Base Transform** is a class that will do the required transformations
   so that the input images will be compatible with the neural network
   When we feed the images to the neural network they have to have a 
   certain format. So BaseTransform class from data takes care of that.
   
*   **VOC_CLASSES** is a just a dictionary that will do the encoding of the classes
   Different classes will be encoded as numbers.
   
   
*   **build_ssd** will be the constructor of the neural network
