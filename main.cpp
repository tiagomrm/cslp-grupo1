// 98134 - Tiago Matos && 98396 - Vítor Dias @ CSLP

#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/core/types.hpp>

//Import color dataset
#include "colors.h"

//Use std and opencv namespaces
using namespace cv;
using namespace std;

//Global Variables
Mat src_gray;
int thresh = 100;
RNG rng(12345);
int width;
int height;

//----
//Functions Declarations

//Get Lego Brick size
void getDimensions(vector<Vec3f> circles, int mX, int mY);

//Get Bounding Rectangles from objects in the image
vector<Rect> getContours(Mat m);

//Check if circles are inside one of the detected rectangles
int insideRect(vector<Vec3f> circles, vector<Rect> rects);

//Get points from arrow equation
double f(double x, double slope, double b);

//
void white_balance(Mat& src, Mat& dst, Mat& mask);
//-----


int main(int argc, char** argv)
{

    // Declare the output variables
    Mat src, dst, dst2, th, masked, img_gray, result, end, color, color_seg, grey, white_balanced;

    //Get Camera instance
    VideoCapture video(0);

    //Make sure the Camera is working
    if (!video.isOpened()){
        cout << "Camera not detected" << endl;
        return -1;
    }

    //Capture Camera frame
    video >> src;

    //Convert color from BGR to Gray Scale
    cvtColor(src, img_gray, COLOR_BGR2GRAY);

    //
    cv::threshold(img_gray, th, 0, 255, THRESH_TOZERO | THRESH_OTSU);

    white_balance(src, white_balanced, th);

    src = white_balanced;

    bilateralFilter(src, img_gray, 9, 301, 301, BORDER_DEFAULT);
    cv::pyrMeanShiftFiltering(src, img_gray, 40, 50, 2);
    cvtColor(img_gray, img_gray, COLOR_RGB2GRAY);

    cv::threshold(img_gray, th, 0, 255, THRESH_TOZERO_INV | THRESH_OTSU);
    cv::bitwise_and(img_gray, img_gray, img_gray, th);

    cvtColor(src, src, COLOR_BGR2HSV);
    cv::bitwise_and(src, src, masked, th);
    cv::pyrMeanShiftFiltering(masked, dst2, 40, 55, 2);

 
    //Clone masked image to find the pins
    Mat pins = masked.clone();

    //
    fastNlMeansDenoising(pins,pins,10.1,10,7);

    //Convert image color from HSV to Gray Scale
    cvtColor(pins, pins, COLOR_HSV2BGR);
    cvtColor(pins,src_gray, COLOR_BGR2GRAY);

    //
    medianBlur(src_gray, src_gray, 5);
    
    //
    equalizeHist( src_gray, src_gray );


    vector<Rect> rectangles;
    rectangles = getContours(src);


    // Find Circles in the image and save them in "circles"
    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1,
                 50,  
                 10, 15, 15, 25
    );

    // Save circles found as "circle" structure
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( pins, center, 1, Scalar(0,100,100), 2, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( pins, center, radius, Scalar(255,0,255), 2, LINE_AA);
    }

    // Get the dimensions of the Lego Brick
    getDimensions(circles, pins.cols, pins.rows);  

    // Find the rectangle that has more circles inside him
    int rect_piece = insideRect( circles , rectangles);

    //Cut the rectangle that has most circles, used to visualize the cricles found
    //pins = pins(rectangles.at(rect_piece));

    // Cut the image that is going to be analyzed to find the colour, in order to get only the part that contains most circles
    dst2 = dst2(rectangles.at(rect_piece));

    // See detected circles in the Lego Brick
    //namedWindow("detected circles", WINDOW_NORMAL);
    //resizeWindow("detected circles", 1000,1000);
    //imshow("detected circles", pins);



    // -- Color Detection --
    vector<int> maxes;
    int max = 0, index;

    //-----------
    for(int i = 0; i < colors_names.size() - 1; i++){
            cv::inRange(
                    dst2,
                    Scalar(colors[i][0] - 5, colors[i][1] / 100 * 0.7 * 255 , colors[i][1] / 100 * 0.7 * 255 + 3 ),
                    //Scalar(colors[i][0] - 5, 0 , 1),
                    //Scalar(colors[i][0] - 15, 0, 5),
                    Scalar(colors[i][0] + 5, colors[i][1] / 100 * 1.3 * 255 , colors[i][1] / 100 * 1.3 * 255 ),
                    //Scalar(colors[i][0] + 5, 255, 150 ),
                    //Scalar(colors[i][010] + 15, 100, 95),
                    end);
            int count = cv::countNonZero(end);
            if(count >= max) {
                cout << colors_names[i] << " ( count : " << count << " )" << endl;
                if (strcmp("Dark Stone Grey", colors_names[i].c_str()) == 0 )
                    grey = end.clone();
                max = count;
                index = i;
                maxes.push_back(i);
                color = end.clone();
            }
    }

    Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 21, 21 ),
                                         Point( 10, 10 ) );


    cv::dilate(color, color, element);
    cv::erode(color, color, element);

    cv::erode(color, color, element);
    cv::dilate(color, color, element);


    //Console output
    cout << "max = " << max << ", index = " << index << endl;
    cout << "color is " << colors_names[index] << endl;

    if(!grey.empty()) {
        //imshow("Grey", grey);
    }


    // Convert source image back to BGR
    cv::cvtColor(src, src, COLOR_HSV2BGR);
    
    
    // Insert data found in the original image to provide output    
    putText(src, "color: ", Point(30,30), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0,0,0), 1);
    putText(src, max > 0 ? toUpperCase(colors_names[index]) : "Unknown", Point(30,60), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0,0,0), 2);
    putText(src, "dimensions : ", Point(30,90), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0,0,0), 1);
    putText(src, to_string(width) + "x" + to_string(height) , Point(30,120), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0,0,0), 1);

    // Show the final result
    imshow("Final Output", src);
    
    // Wait and Exit
    waitKeyEx();
    return 0;
}

void getDimensions(vector<Vec3f> circles, int mX, int mY)
{
    // Set the extreme points to find maximum X and Y and to find minimum Y
    Point2f minX(mX,0), minY(0,mY), maxY(0,0);
    // Search the circles to set the extremes
    for(int i = 0; i < circles.size(); i++)
    {
        Point2f p(circles.at(i)[0], circles.at(i)[1]);
        if(p.x < minX.x){
            Point2f pf(p.x,p.y);
            minX = pf;
        }
        if(p.y < minY.y){
            Point2f pf(p.x,p.y);
            minY = pf;
        }
        if(p.y > maxY.y){
            Point2f pf(p.x,p.y);
            maxY = pf;
        }
    }

    // Get arrow parameters
    double d1 = (minX.y - maxY.y)/(minX.x - maxY.x);
    double d2 = (minX.y - minY.y)/(minX.x - minY.x);
    double b1 = minX.y - (d1 * minX.x);
    double b2 = minX.y - (d2 * minX.x);
    double d;

    // Count the number of circles where one point intersepts the arrow at the x where the arrow intersepts the circle center
    int l1 = 0, l2 = 0;
    for(int i = 0; i < circles.size(); i++){
        int y = f(circles.at(i)[0], d1, b1);
        if( pow( circles.at(i)[0] - circles.at(i)[0], 2) + pow( y - circles.at(i)[1], 2) < pow( circles.at(i)[2], 2) ){
            l1++;
        }
        
        y = f(circles.at(i)[0], d2, b2);
        if( pow( circles.at(i)[0] - circles.at(i)[0], 2) + pow( y - circles.at(i)[1], 2) < pow(circles.at(i)[2], 2) ){
            l2++;
        }
    }

    // Set Lego dimensions
    width = l1;
    height = l2;
}

// Calculate arrow Y's
double f(double x, double slope, double b){
    return x * slope + b;
}

// Get the objects found at the image
vector<Rect> getContours(Mat m){
    Mat src = m;
    Mat th;

    Canny( src_gray, th, 200, 200*2 );
    vector<vector<Point> > contours;
    findContours( th, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
    }
    //Mat drawing = Mat::zeros( th.size(), CV_8UC3 );
    //for( size_t i = 0; i< contours.size(); i++ )
    //{
    //    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    //    drawContours( drawing, contours_poly, (int)i, color );
    //    rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
    //}

    return boundRect;
}

// Check which bounding rectangle has more circles and return his index
int insideRect(vector<Vec3f> circles, vector<Rect> rects){
    int index = 0, max = 0;
    int counter;
    for(int i = 0; i < rects.size(); i++){
        counter = 0;
        for(int j = 0; j < circles.size(); j++){
            if(rects.at(i).contains( Point2f (circles.at(j)[0], circles.at(j)[1]) )){
                counter++;
            }
        }
        if (counter > max){
            max = counter;
            index = i;
        }
    }
    return index;
}

void white_balance(Mat& src, Mat& dst, Mat& mask) {
    vector<Mat> imageBGR;
    split(src, imageBGR);
    Scalar mean_value = mean(src, mask > 0);

    cout << mean_value << endl;

    double b_transform = 255.0 / mean_value[0];
    double g_transform = 255.0 / mean_value[1];
    double r_transform = 255.0 / mean_value[2];

    Mat blue = Mat::zeros(src.rows, src.cols, imageBGR.at(0).type());
    Mat green = Mat::zeros(src.rows, src.cols, imageBGR.at(0).type());
    Mat red = Mat::zeros(src.rows, src.cols, imageBGR.at(0).type());

    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            double temp_blue = imageBGR.at(0).at<uchar>(y, x) * b_transform;
            double temp_green = imageBGR.at(1).at<uchar>(y, x) * g_transform;
            double temp_red = imageBGR.at(2).at<uchar>(y, x) * r_transform;
            blue.at<uchar>(y, x) = temp_blue > 255 ? 255 : temp_blue;
            green.at<uchar>(y, x) = temp_green > 255 ? 255 : temp_green;
            red.at<uchar>(y, x) = temp_red > 255 ? 255 : temp_red;
        }
    }

    vector<Mat> channels;

    channels.push_back(blue);
    channels.push_back(green);
    channels.push_back(red);

    merge(channels, dst);
}
