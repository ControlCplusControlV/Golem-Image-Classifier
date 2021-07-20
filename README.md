# Golem Image Classifier Service

This service was designed for the bounty put out by Golem as seen [here](https://gitcoin.co/issue/golemfactory/yagna/1456/100026046). To run this service a pre-existing requestor node setup is required, but if you don't have one a quick primer can be found [here](https://handbook.golem.network/requestor-tutorials/flash-tutorial-of-requestor-development).

## Using the Service

Clone this repo into the folder of your choice, the main componet needed for testing is the requestor.py script, but the entire service code is included in the service folder if you need to check something. You also need to download the [model](https://mega.nz/file/56wTCQJA#lC6VrHFuC8Sc6xgYcOf4A9ufyGnIsVFofoaUR3ETvUI) and the [additional training set](https://mega.nz/file/gyhxBSJY#EHAdZQygGMlX9etFrubc9zc9ePosQWTh0s-6poZEUns) in order to complete the demo. Leave both of them in their respective .tar.gz's inside the folder where you plan to run your requestor script. When the script is run it should startup with some yagna negotiation dialog before continuing into a prompt. Please note this can take some time as the model is around 550MB and must be transfered to the provider before the service can begin. Then the prompt shown below will appear.

```
What task do you wish to run? [predict/train] : 
```
### Predict
if you answer "predict" it will continue asking for an input image, jpg is the current supported format. To run a prediction provide the name of a jpg inside of the current directory, in the example case, "Bluebell.jpg". Once this image name is supplied it the requestor will send it out the provider and you will get a response back showing a label like below in the case of Bluebell.jpg
```
Bluebell
```

### Train

To further train the existing model, a .tar.gz is needed containing 4 folders named after each of the training labels used, with the photos corresponding to the label inside each folder. Respond to the prompt with the filename of your training data (train.tar.gz in the example)
```
What is the name of the training data folder :
```

After this is input the provider will take some time, and then "Model Successfully Trained" will appear indicating successful training, and you will be re-directed back to the main prompt.

## Modifying for Personal/Business Use

If you plant to modify this for personal or business use, use a saved tensorflow model zipped up in a tar.gz, then imageclassifier.py will need to be modified to correspond to the labels of your new data.

### Questions?

If you have any extra questions make sure to reach out to Nebula on the Golem Discord!
