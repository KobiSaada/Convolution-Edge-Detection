# Convolution-Edge-Detection

Welcome to EX2 IPCV Curse The purpose of this exercise is to help you understand the concept of the convolution and edged etection by performing simple manipulations on images.


# Prerequisites: :bulb:
**Written in Python on PyCharm workspace** ![22](https://user-images.githubusercontent.com/73976733/104122372-55ae3700-534d-11eb-8de4-492973fca972.jpeg)
                          
**convolution on 1D and 2D arrays** -> convolution of 1 1D discrete signal and 2D discrete signal.

**Performing image derivative and image blurring** ->  function that computes the magnitude and the direction of an image gradient. You should derive
the image in each direction separately (rows and column) using simple convolution with [1, 0, −1]T and
[1, 0, −1] to get the two image derivatives. Next, use these derivative images to compute the magnitude
and direction matrix and also the x and y derivatives.

**Edge detection**->  implement edgeDetectionZeroCrossingLOG OR edgeDetectionZeroCrossingSimple

**Hough Circles**->implement the Hough circles transform I is the intensity image, minradius, maxradius should positive numbers and minradius < maxradius.
Use the Canny Edge detector as the edge detector, you can use the function: cv.Canny()
functions should return a list of all the circles found, each circle will be represented by:(x,y,radius). Circle
center x, Circle center y, Circle radius.
* This function is costly in run time, be sure to keep the min/max Radius values close and the images
small.

**Bilateral filter**->implement the Bilateral filter, compare your implementation with Open CV implementation
cv.bilateralFilter() 
                          

# Some screen shoot outputs of the tasks listed above using the OpenCV and numpy libraries::camera:


**Performing image derivative and image blurring:** 
<img width="509" alt="Screen Shot 2022-04-26 at 23 49 08" src="https://user-images.githubusercontent.com/73976733/165391938-1c628873-44e5-4f1e-9d02-c993f0b1bb71.png">

**Edge detection(LOG):**
<img width="508" alt="Screen Shot 2022-04-26 at 23 59 36" src="https://user-images.githubusercontent.com/73976733/165392074-1b73f5ec-ffed-49f4-ab4e-26e0b6f41e26.png">

**Hough Circles:**
<img width="493" alt="Screen Shot 2022-04-27 at 0 00 09" src="https://user-images.githubusercontent.com/73976733/165392127-c6a4fc8b-c5bf-4ddf-afec-747045b3b33a.png">

**Bilateral filter:**
you can see filtered_image_my and filtered_image_OpenCV jpg files the error is 8

**Enjoy**:smile:


