
#include "opencv2/opencv.hpp"

 

using namespace cv;

 

int main(int argc, char **argv)

{

    // Read image

    Mat im_in = imread("../Legos/image2.jpg", IMREAD_GRAYSCALE);

    resize(im_in, im_in, Size(), 0.2, 0.2, INTER_LINEAR);

    imshow("asd", im_in);

    // Threshold.

    // Set values equal to or above 220 to 0.

    // Set values below 220 to 255.image

    Mat im_th;
    threshold(im_in, im_th, 220, 255, THRESH_BINARY_INV | THRESH_OTSU);

 

    // Floodfill from point (0, 0)
    Mat im_floodfill = im_th.clone();
    floodFill(im_floodfill, Point(0,0), Scalar(255),Scalar(0,0,0), Scalar(20,20,20));
 
    // Invert floodfilled image
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);
 
    // Combine the two images to get the foreground.
    Mat im_out = (im_th | im_floodfill_inv);
 
    // Display images
    imshow("Thresholded Image", im_th);
    imshow("Floodfilled Image", im_floodfill);
    imshow("Inverted Floodfilled Image", im_floodfill_inv);
    imshow("Foreground", im_out);
    waitKey(0);

}
