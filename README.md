## Self-Supervised Image Colorization Using CNN and L2 Regression Loss.  
This mini-project discusses the implementation of a model for coloring black-and-white images.  

The CIE Lab color model describes all colors visible to the human eye in three parameters: L (for black-to-white brightness intensity with a value between 0 and 100), a (to describe the color position between green and red with a negative value for green and positive for red) and b (to describe the color position between blue and yellow with negative quantification for blue and positive for yellow). This model, unlike the RGB model, is a color space independent of the device and is designed almost close to human vision, and the distances inside it model the perceptual distance. In the application of coloring black-and-white images, you can simply consider the brightness intensity parameter L for the input and the color parameters a and b for the output of the model to produce a color image by adding to the parameter L.  
  
A deep convolutional network that takes the L parameter matrix of each image in the input and after several convolutional layers along with ReLU and BatchNorm without the Pool layer, obtains the a and b parameters of the image pixels was used. The simple model of this network can be seen below.  

<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/03d2b126-75f8-4ee1-9b71-7474ee9216fd.png" width="800"></div>  

Notably, in the test phase, the temperature adjustment steps, the softmax function, the mean function, and the bilinear upsampling step are implemented as subsequent layers in the feed path. Also, the effective dilation used in the conv5 and conv6 layers is the space where the successive elements of the convolution kernel are evaluated relative to the input pixels. This step is calculated by multiplying the accumulated strides in the dilation of the layer. Through each convolutional block from conv1 to conv5, the effective expansion of the convolutional core increases and decreases from conv6 to conv8. The table below shows the details of the model architecture along with all the functions used.  

<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/59fad47f-e0cc-4339-a080-e88075d3d304.png" width="400"></div>  

To make the data loader, the images were resized to 256x256, those that were black-and-white already were changed to three-channel RGB mode, and finally, all the images were taken to the CIELab system. Since the problem is self-supervised, the desired output is known for each input image, and loss L2 can be used as a measure of the distance between the predicted and desired output. Presented below is the diagram for regression loss per epoch.  

<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/a373da8e-9546-4f1a-b7d6-cd8848e505ab.png" width="450"></div>  

Here are some results of this model after 40 epochs compared to the original images.

<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/498c02e9-84d4-4390-9e21-163fcd1fd46c.png" width="400"></div>  
<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/6e06e7ea-b4d7-4f2d-89dd-53a80c6ea824.png" width="400"></div>  
<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/2ab6ef9d-e49c-4a22-a9eb-8d0fa97b7f17.png" width="400"></div>  
<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/abc8bd7d-ffe8-4e15-b5b1-cb81b814fde1.png" width="400"></div>  
<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/812450b3-7e48-4405-b8c9-323fb10b4fa4.png" width="400"></div>  
<img src="https://github.com/smsadjadi/Mini-Project-ImageColorization/assets/62998417/74edc91d-7694-4f74-98dc-0030b3ae4359.png" width="400"></div>  
