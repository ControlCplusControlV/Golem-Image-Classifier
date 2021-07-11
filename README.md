# Golem Image Classifier Service

This service was designed for the bounty put out by Golem as seen [here](https://gitcoin.co/issue/golemfactory/yagna/1456/100026046). To run this service a pre-existing requestor node setup is required, but if you don't have one a quick primer can be found [here](https://handbook.golem.network/requestor-tutorials/flash-tutorial-of-requestor-development).

## Using the Service

Clone this repo into the folder of your choice, the main componet needed for testing is the requestor.py script, but the entire service code is included in the service folder if you need to check something. When the script is run it should startup with some yagna negotiation dialog before continuing into a prompt.

```
What task do you wish to run? [predict/train] : 
```
### Predict
if you answer "predict" it will continue asking for an input image, jpg is the current supported format. To run a prediction provide the name of a jpg inside of the current directory, in the example case, "test2.jpg". Once this image name is supplied it the requestor will send it out the provider and you will get a response back showing a label like below in the case of test2.jpg
```
iris
```

### Train

To further train the existing model, 2 things are required. An h5 file containing the labels of your data, and another h5 file to containing the feautures of each label. Move these 2 files to the current working directory, then reply as shown below.
```
What is the name of the h5 data file :
data.h5
What is the name of the h5 labels file :
labels.h5
```
"Model Successfully Trained", will then appear indicating successful training, and you will be re-directed back to the main prompt.
