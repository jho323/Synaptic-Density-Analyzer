# Synaptic-Density-Analyzer
A python-based tool that is able to analyze the spine density on the dendrites of neurons



##Spine Detection Interface

- My project is a program that can make analyzing dendritic spines on neurons easier. It will include features such as an automatic spine detector, a count tool, image editing tools, and a file read, write, and save function.The purpose is to make the process of analyzing spines on a dendrite easier.

##How to Run:

The user should load the "Interface.py" file in an editor. Make sure the folder "Models" is in the working directory. The user needs to first load an image into the window by clicking open and selecting the file needed. Any modifications of the image can be done with the tools on the page.

##How to install libraries:
The three libraries needed are OpenCV, Numpy and Pillow, a version of PIL(python image library).

OpenCV: installing OpenCV will automatically install Numpy as well
python -m pip install opencv-python

Pillow: 
python -m pip install Pillow

##Shortcuts:

Press "T","L", "M", "F", "S" to manually label generic, long thing, mushroom, filopodia, and stubby spines, respectively
Press "r" to reset the window.
Press "select" and then click on a marker and then backspace to delete that marker.
Press "show" and "hide" under Spine counter to display specific markers and manually placed markers. 
Press "-" (minus) or "+"(plus) keys to shrink or enlarge the image.
Use arrow keys to move the iamge around

When using autocount, press the "auto-counter" button for generic spine labeling first before pressing "spine counter".
When saving, press "save as" to select the folder in the working directory and then press "save" to save the image.
Click on the filename when opening a file.


