# Lego Identification AI 

This AI utilizes the imagenet network to identify legos and sort them based on predetermined classes. 

![Sample Test Image to be used with the lego.py program.](https://github.com/matthew90-ai/Lego-ID-AI/blob/master/outputimage.jpg)

## The Algorithm

The dataset I used to retrain the model can be found at the bottom of the page. 
I retrained the resnet18 network to identify the types of lego based on the 20 classes: <br>  
<br>
Brick_1x1 <br>
Brick_1x2 <br>
Brick_1x3 <br>
Brick_1x4 <br>
Brick_2x2 <br> 
Brick_2x2_L <br> 
Brick_2x2_Slope <br>
Brick_2x3  <br>
Brick_2x4  <br>
Plate_1x1  <br>
Plate_1x1_Round  <br>
Plate_1x1_Slope  <br>
Plate_1x2  <br> 
Plate_1x2_Grill  <br>
Plate_1x3  <br>
Plate_1x4  <br>
Plate_2x2  <br>
Plate_2x2_L  <br>
Plate_2x3  <br>
Plate_2x4 <br>
<br>
<br>
The Lego AI program analyzes the given image with the retrained model and outputs where you can find the brick online through 3rd party retailers. 
The program is based off the Imagenet program from Jetson Inference Library on the Nano and is held on the Lego.py file on this repository. 

## Running this project

1. Download Jetson Inference Libraries
2. Download respository files
3. cd to respository directory (called Lego-ID-AI)
4. run command: `python3 lego.py <inputimage> <outputimage.jpg>`
5. outputs predicted class and supplies link to purchase the identified lego online.

## Resources
[Dataset](https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition) <br>
[Jetson Inference Github](https://github.com/dusty-nv/jetson-inference) <br>
[View a video explanation here](video link)
