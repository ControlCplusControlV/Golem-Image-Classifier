# Golem Image Classifier Service

This service was designed for the bounty put out by Golem as seen [here](https://gitcoin.co/issue/golemfactory/yagna/1456/100026046). To run this service a pre-existing requestor node setup is required, but if you don't have one a quick primer can be found [here](https://handbook.golem.network/requestor-tutorials/flash-tutorial-of-requestor-development). This service defines an requestor.py subsystem which can be given predict tasks. A dataset is required to make predictions, right now the neural nets itself is vgg16, but I am looking into EfficientNetV2.

## Using the Service

Clone this repo into the folder of your choice, the main componet needed for testing is the requestor.py script, but the entire service code is included in the service folder if you need to check something. Next download [Model Weights](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5) and name it as "vgg16.h5", this is a required step as the service requires these weights for initialization. Make sure these weights are in the same folder as the requestor script.

The service responds to 2 main types of requests, Predict and Train. The Requestor script itself is used a subprocess that must be initialized with parameters before being incorporated into a larger process. See demo.py for examples, a couple of files are needed, and all are zipped as .tar.gz in order to reduce time spent sending data, so more can be used for training. Arguments for tasks are seperated by spaces.

# Demo 

Files used in demo.py

[Dataset](https://mega.nz/file/dngA1J6J#uxrI6DOFMzdcr4vmU_9Y3gYLn1axbZ6X_a6imusPgDY)

[Train data](https://mega.nz/file/tnoUjBRS#lC_gRgmHQuokJQSJ3sSx-KsixOby3nbSiuFOkG5p2xk)

[Validation Data](https://mega.nz/file/tj4UHDBb#uqYCN9f9K19oY2kLQEr3YBQkIh_G3-FHug4v_LBL0sw)

[Test Image](https://mega.nz/file/Zih2QBiR#uAm-dKGQutINAq4StWBP2Wqy9hV4QPKJm2Tmpm792sU)


Demo.py showcases how to interact with the requestor in an automated way, running the requestor as a subprocess allows it to interconnect with existing ML implementations without needing to build additional network handling to send and recieve requests, instead handling stdout and stdin directly. Running demo.py does a couple things

- First requestor.py is started with  -d dataset -c dog monkey cat cow as args, this tells it the classes to be used in the demo, and dataset is the name of our dataset so it searches for "dataset.tar.gz"

- Sends the requestor a task once intialized to predict with test1.jpg, this should respond with monkey in the stdout marking the prediction

![predict](predict.png?raw=true)

- Another task is then sent with more validation images and training images, once recieved the neural net then trains on those images and returns "Model Training Success" once complete.

![train](train.png?raw=true)

- Finally, the test image is sent again to verify that the neural net is working, and was trained properly


![final](final.png?raw=true)

The demo is primarly there to showcase how to incorporate the requestor as a subprocess module, I chose this approach for ease into existing neural net implementations, as its similar to using another library, but with an added daemon process.

## Requestor

The requestor script requires 2 things upon initialization, a dataset archive in .tar.gz format with a similar format to the one shown in /services/dataset ,and a list of class names.

Example - requestor.py -d dataset -c dog monkey cat cow

It then prompts the user for input on which task they would like to execute. Tasks are single strings with arguments seperated by a space. The 2 types of tasks the requestor responds to are "predict" or "train".

### Predict

__Required Args__

- a .jpg file in the same directory as the requestor script

Returns a labal inside of stdout.

Example - "predict test1.jpg"

### Train

__Required Args__

- A .tar.gz archive containing training images, important to note these images must be directly inside the archive, not a subdirectory within it

- A .tar.gz archive containing validation images, important to note these images must be directly inside the archive, not a subdirectory within it

Returns "Model Successfully Trained" in stdout

Example - "train train.tar.gz valid.tar.gz"

## Modifying for Personal/Business Use

If you plant to modify this for personal or business use, use a dataset with the same format as shown in /service/dataset and zip it up in .tar.gz, then use the demo.py script as a example to base your script to off/modify it. THe vgg16.h5 weights can be changed for other neural nets with minimal modification as neural net is initialized from the weights, but optimizations are for vgg16 so you will encounter irregularities/errors 

### Questions?

If you have any extra questions make sure to reach out to Nebula on the Golem Discord!
