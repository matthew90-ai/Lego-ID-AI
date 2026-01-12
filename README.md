# Lego Identification AI 

This AI utilizes the imagenet network to identify legos and sort them based on predetermined classes. 

![add image descrition here](direct image link here)

## The Algorithm

The dataset I used to retrain the model can be found at the bottom of the page. 
I retrained the resnet18 network to identify the types of lego based on the 20 classes: <br>  Brick_1x1  Brick_1x2  Brick_1x3  Brick_1x4    Brick_2x2  Brick_2x2_L  Brick_2x2_Slope  Brick_2x3  Brick_2x4  Plate_1x1  Plate_1x1_Round  Plate_1x1_Slope  Plate_1x2   Plate_1x2_Grill  Plate_1x3  Plate_1x4  Plate_2x2  Plate_2x2_L  Plate_2x3  Plate_2x4


The Lego AI program analyzes the given image with the retrained model and outputs where you can find the brick online through 3rd party retailers. 
The program is based off the Imagenet program from Jetson Inference Library on the Nano and is held on the Lego.py file on this repository. 

## Running this project

1. Add steps for running this project.
2. Make sure to include any required libraries that need to be installed for your project to run.

[View a video explanation here](video link)
