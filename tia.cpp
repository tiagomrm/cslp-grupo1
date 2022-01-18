#include <iostream>
#include <opencv2/opencv.hpp>

#include "colors.h"
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);


void thresh_callback(int, void* );

int main(int argc, char** argv)
{
    // Declare the output variables
    Mat src, dst, dst2, th, masked, img_gray, result;
    const char* default_file = "legos.jpg";
    //default_file = "lego3.png";
    const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    Mat original = imread( samples::findFile( filename ), IMREAD_COLOR );

    float scale = 1;

    if (strcmp(default_file,"lego3.png") != 0)
        scale = 0.6;

    cv::resize(original, src, Size(), scale, scale, INTER_LINEAR);
    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }

    bilateralFilter(src, img_gray, 9, 301, 301, BORDER_DEFAULT);
    cvtColor(img_gray, img_gray, COLOR_RGB2GRAY);

    cv::threshold(img_gray, th, 0, 255, THRESH_TOZERO_INV | THRESH_OTSU);
    cv::bitwise_and(img_gray, img_gray, img_gray, th);

    cv::bitwise_and(src, src, masked, th);
    cv::pyrMeanShiftFiltering(masked, dst2, 40, 50, 2);

    // Show results
        namedWindow("Original",WINDOW_NORMAL);
    resizeWindow("Original", 1000,1000);
        namedWindow("Mean-Shift Segmentation",WINDOW_NORMAL);
    resizeWindow("Mean-Shift Segmentation", 1000,1000);
        namedWindow("blurred gray",WINDOW_NORMAL);
    resizeWindow("blurred gray", 1000,1000);
        namedWindow("Threshold",WINDOW_NORMAL);
    resizeWindow("Threshold", 1000,1000);
        namedWindow("Result",WINDOW_NORMAL);
    resizeWindow("Result", 1000,1000);
    imshow("Original", src);
    imshow("Mean-Shift Segmentation", dst2);
    imshow("blurred gray", img_gray);
    imshow("Threshold", th);
    src_gray = th;
    imshow("Result", masked);


    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", "Threshold", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    // Wait and Exit
    waitKeyEx();
    return 0;
}

void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours_poly, (int)i, color );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
        circle( drawing, centers[i], (int)radius[i], color, 2 );
    }
    namedWindow("Contours",WINDOW_NORMAL);
    resizeWindow("Contours", 1000,1000);

    imshow( "Contours", drawing );
}