#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network


# note: to hard-code the paths to load a model, the following API can be used:
#
net = imageNet(model="resnet18.onnx", labels="labels.txt", 
                 input_blob="input_0", output_blob="output_0")

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=args.topK)

    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID).rstrip()
        
        confidence *= 100.0

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                         x=5, y=5 + n * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
        
        if classLabel == "Brick_1x1":
            print("This is a 1x1 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x1#T=A ")
       
        if classLabel == "Brick_1x2":
            print("This is a 1x2 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x2#T=A ")
        
        if classLabel == "Brick_1x3":
            print("This is a 1x3 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x3%20brick#T=A ")                        
        
        if classLabel == "Brick_1x4":
            print("This is a 1x4 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x4%20brick#T=A ")   
        
        if classLabel == "Brick_2x2":
            print("This is a 2x2 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x2%20brick#T=A ")       
       
        if classLabel == "Brick_2x2_L":
            print("This is a 2x2 L Shaped Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x2%20Brick%20Corner#T=A ")   
        
        if classLabel == "Brick_2x2_Slope":
            print("This is a 2x2 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x2%20Brick%20Slope#T=A ")   
        
        if classLabel == "Brick_2x3":
            print("This is a 2x3 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x3%20Brick#T=A")   

        if classLabel == "Brick_2x4":
            print("This is a 2x4 Lego Brick, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x4#T=A ")   

        if classLabel == "Plate_1x1":
            print("This is a 1x1 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x1%20Plate#T=A ")   

        if classLabel == "Plate_1x1_Round":
            print("This is a 1x1 Round Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x1%20Round#T=A")   

        if classLabel == "Plate_1x1_Slope":
            print("This is a 1x1 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x1%20Slope#T=A ")   
   
        if classLabel == "Plate_1x2":
            print("This is a 1x2 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x2%20Plate#T=A ")   
   
        if classLabel == "Plate_1x2_Grill":
            print("This is a 1x2 Lego Grill Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/catalog/catalogitem.page?P=2412b&name=Tile,%20Modified%201%20x%202%20Grille%20with%20Bottom%20Groove&category=%5BTile,%20Modified%5D#T=C ")   
   
        if classLabel == "Plate_1x3":
            print("This is a 1x3 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x3%20Plate#T=A ")   

        if classLabel == "Plate_1x4":
            print("This is a 1x4 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=1x4%20Plate#T=A ")   

        if classLabel == "Plate_2x2":
            print("This is a 2x2 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x2%20Plate#T=A ")   
        
        if classLabel == "Plate_2x2_L":
            print("This is a 2x2 L Shaped Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x2%20Corner%20Plate%20Plate#T=A ")   
        
        if classLabel == "Plate_2x3":
            print("This is a 2x3 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x3%20Plate#T=A ")   
        
        if classLabel == "Plate_2x4":
            print("This is a 2x4 Lego Plate, you can find this through Bricklink on this url: https://www.bricklink.com/v2/search.page?q=2x4%20Plate#T=A ")   
   
   
   
   
   
   
   
   
   
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break


