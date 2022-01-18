#include <opencv2/opencv.hpp>
#include <iostream>
#include <colors.h>
#include <vector>

using namespace std;
using namespace cv;


RNG rng(12345);

const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static int checkColor(Mat mat){
    int counter = 0;
    for(int i = 0; i< mat.rows; i++){
        for(int j = 0; j<mat.cols; j++){
            if( mat.at<uchar>(i, j) > 0)
                counter++;
        }
    }
    cout << "Contador : " << counter << endl;
    return counter;
}


int main(int argc, char** argv)
{

    CommandLineParser parser( argc, argv, "{@input |  lena.jpg | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
    if( src.empty() )
    {
        return EXIT_FAILURE;
    }


    Mat hsv_image;
    cvtColor(src,hsv_image, COLOR_BGR2HSV);


    namedWindow(window_capture_name, WINDOW_NORMAL);
    namedWindow(window_detection_name, WINDOW_NORMAL);

    const String window_reverse = "Reverse";
    namedWindow(window_reverse, WINDOW_NORMAL);
    imshow(window_reverse,src);


    Mat blured;
    //for ( int i = 1; i < 30; i = i + 2 )
    //{
    //    GaussianBlur( hsv_image, blured, Size( i, i ), 0, 0)    ;
//
    //}

    


    Mat th, masked;
    cv::pyrMeanShiftFiltering(src, blured, 40, 50, 2);

    cvtColor(blured,blured, COLOR_BGR2GRAY);

    cv::threshold(blured, th, 0, 255, THRESH_TOZERO_INV | THRESH_OTSU);
    cv::bitwise_and(blured, blured, blured, th);
    cv::bitwise_and(src, src, masked, th);

    const String poggerws = "Bruv";
    namedWindow(poggerws, WINDOW_NORMAL);
    resizeWindow(poggerws, 1000,1000);
    imshow( poggerws, masked );

    

//    Mat canny_output;
//    Canny( blured2, canny_output, thresh, thresh*2 );
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;
//
//    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
//    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
//    vector<Rect> boundRect( contours.size() );
//    vector<vector<Point> > contours_poly( contours.size() );
//
//    for( size_t i = 0; i < contours.size(); i++ )
//    {
//        approxPolyDP( contours[i], contours_poly[i], 3, true );
//        boundRect[i] = boundingRect( contours_poly[i] );
//    }
//
//    float area, maxArea;
//    int maxIdx;
//
//    for( size_t i = 0; i< contours.size(); i++ )
//    {
//        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
//        drawContours( drawing, contours, int(i), color, 2, LINE_8, hierarchy, 0 );
//        float area= contourArea(contours[i]);
//        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
//
//        cout << area << endl;    
//        if(area > maxArea)
//        {
//            maxArea = area;
//            maxIdx = i;
//        }
//    }
//
//
//    namedWindow("dra", WINDOW_NORMAL);
//    resizeWindow("dra", 1000,1000);
//    imshow( "dra", drawing );
//
//    if(maxIdx < 0)
//    {
//        std::cout << "no contour found" << std::endl;
//        return 1;
//    }
//
//    cout << "max " << maxArea << endl;
//
//
//    const String cont = "contours";
//    namedWindow(cont, WINDOW_NORMAL);
//    resizeWindow(cont, 1000,1000);
//    imshow( cont, drawing );



    //---------


    //Mat res;
    //int erosion_size = 1;
    //Mat element = getStructuringElement( 0,
    //                Size( 2*erosion_size + 1, 2*erosion_size+1 ),
    //                Point( erosion_size, erosion_size ) );
//
    //dilate( src, res, element );
    //const String win_erosion = "Result";
    //namedWindow(win_erosion, WINDOW_NORMAL);
    //imshow(win_erosion, res);


    Mat end;
    int max = 0;
    int index = -1;
    Mat top;

    vector<int> maxes;
    //-----------
    for(int i = 0; i < colors_names.size() - 1; i++){
        cout <<  i << endl;
        inRange(blured, Scalar(colors[i][0] - 5, 20, 1), Scalar(colors[i][0] + 5, 255,  255),  end);
        int count = checkColor(end);
        if(count > 1000 ){//&& count >= max){
            max = count;
            index = i;
            top = end.clone();
            maxes.push_back(i);
            const String maxes = "Result";
            namedWindow(maxes, WINDOW_NORMAL);
            imshow(maxes, top);
            waitKey(0);
        }

    }
    // Show the frames
    imshow(window_capture_name, blured);
    resizeWindow(window_reverse, 1000,1000);
    imshow(window_reverse, top);


    cout << "max = " << max << ", index = " << index << endl;
    cout << "color is " << colors_names[index] << endl;

    for(int j = 0; j < maxes.size(); j++){
        cout << "   " << colors_names[maxes[j]] << endl;
    }

    char key = (char) waitKey(0);

    return 0;
}