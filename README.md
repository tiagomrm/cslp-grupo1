# CSLP Project - Group 1

## Lego Blocks Color and Dimensions Detection

This project is able to detect an approximation to the color of a Lego Block, based on an official color palette, and 
it's dimensions (e.g., 2x8) using a camera.

The code was implemented in C++ with the OpenCV 4 library in order to manipulate and capture the frames.

We did not manage to make the project work very accurately since the environment conditions played such a big part on 
how the colors and shadows were displayed. Although our approach was to handle this problem algorithmically, a better 
and easier solution would be to control these variables with artificial lighting which would allow for more reliable 
results.

Project files:
- `colors.hpp` : Dataset with the colors
- `main.cpp`  : Main program containing all code and functions


## How to install:

In order to run the program, the Raspberry Pi needs OpenCV library.
This is the guide we used : https://solarianprogrammer.com/2019/09/17/install-opencv-raspberry-pi-raspbian-cpp-python-development/

## How to run
To run the program, after having OpenCV library installed:

Compile:
```
$ g++ main.cpp -o readLego `pkg-config --cflags --libs opencv
```


Execute:
```
$ ./readLego
```

To see our setup and inept explanation check the report.


## Authors:
| Name        | NMEC  |
|-------------|-------|
| Tiago Matos | 98134 |
| Vítor Dias  | 98396 |
