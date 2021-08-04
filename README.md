# Golem Image Classifier Service

This service was designed for the bounty put out by Golem as seen [here](https://gitcoin.co/issue/golemfactory/yagna/1456/100026046). To run this service a pre-existing requestor node setup is required, but if you don't have one a quick primer can be found [here](https://handbook.golem.network/requestor-tutorials/flash-tutorial-of-requestor-development). 

## Using the Service

Clone this repo into the folder of your choice, the main componet needed for testing is the requestor.py script, but the entire service code is included in the service folder if you need to check something. Next download [Model Weights](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5) and name it as "vgg16.h5", this is a required step as the service requires these weights for initialization. Make sure these weights are in the same folder as the requestor script.

The service responds to 2 main types of requests, Predict and Train. The Requestor script itself sets up a local api so you can easily send lots of requests it then forwards to the service running on the provider. 

## Initialzation

The requestor script requires 2 things upon initialization, a dataset archive in .tar.gz format with a similar format to the one shown in /services/dataset ,and a list of class names.

## Predict

__Required Args__

- a .jpg file in the same directory as the requestor script


## Train

__Required Args__

- A .tar.gz archive containing training images, important to note these images must be directly inside the archive, not a subdirectory within it

- A .tar.gz archive containing validation images, important to note these images must be directly inside the archive, not a subdirectory within it

Returns a message when the model training is completed

## Modifying for Personal/Business Use

If you plant to modify this for personal or business use, use a dataset with the same format as shown in /service/dataset and zip it up in .tar.gz, then use the demo.py script as a example to base your script to off/modify it. 

### Questions?

If you have any extra questions make sure to reach out to Nebula on the Golem Discord!
