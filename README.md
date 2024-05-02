# DIY-Video_Filter-Jetson_Nano
This repository is for an Computer Vision project on an Embedded system (Jetson-Nano). The project entails Real-time Video augmentation using techniques like Masking, Segmentation for Background Blur and Replacement, Facial Distortions etc.

The project aims to cover 5 main sub-tasks which have been covered both individually and in a Combined Approach. The tasks are ad follows:
* Background Blur.
* Background Replacement.
* Face Distortion.
* Face Filter/ Creative Masks.
* Creative Filter effect (Cartoonization).

### Details of the project
* The project runs on an embedded system i.e., Jetson-Nano (link: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/). The nano is also attached an Raspberry-pi camera module v2 (Link : https://www.raspberrypi.com/products/camera-module-v2/) to enable live vedio streaming.
* The project contains individual codes for each of the sub-tasks and one final code which combines all the sub-tasks namely "Final_Project.py".
* The project runs on a streamlit server which provides and interactive dropdown for switching between different filters which are applied to the real-time vedio feed shown on the local host server.
* For segmentation which an important apsect of tasks like background blur, background replace and cartoonization i used segmentation model called "PP-HumanSeg" for Human Segmentaion mask from opencv_zoo (link : https://github.com/opencv/opencv_zoo).
* As the project runs on embedded system i used the onnx version which works better on nano's gpu. Morover i used the PP-HumanSeg only for mask extraction and then i wrote custom python script for pixel replacement and overlay so that the filter effects are real-time.
* Finally the whole project uses opencv for image processing and i have used harcascades for face detection and replacement with mask for two tasks namely "Face Distortion" and "Face Masking".

### Files added for project
* Alpha support images of masks to blend over face
* Segmentation model
* results and code
