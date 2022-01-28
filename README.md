# CSLP Project | Group 1

Lego Blocks Color and Dimensions Detection

This project is able to detect an aproximation to the color of a Lego Block, based on an official color pallet, and it's dimensions (2x8, p.e.) using a camera.
The code is in c++ with the OpenCV library in order to manipulate and capture the frames.

We didn't manage to make the project working 100%, because the environment conditions shoul've been controled since the luminosity, wich we expected we could control, had way more influence than we thought.
With that, we may now control the environment and make the project more reliable

Project files:
    - colors.hpp : Dataset with the colors
    - main.cpp  : Main program containing all code and functions


## How to install:

In order to run the program, the Raspberry Pi needs OpenCV library.
This is the guide we used : https://solarianprogrammer.com/2019/09/17/install-opencv-raspberry-pi-raspbian-cpp-python-development/

## How to run
To run the program, after having OpenCV library installed:
 
###Compile:
    ```
    $ g++ main.cpp -o readLego `pkg-config --cflags --libs opencv
    ```


###Execute:
    ```
    $ ./readLego
    ```

To see our setup and indepth explanation check the report.


## Authors:
    - Tiago Matos nmec 98134
    - Vítor Dias nmec 98396
