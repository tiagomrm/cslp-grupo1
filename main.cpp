#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/core/types.hpp>


#include "colors.h"
using namespace cv;
using namespace std;

Mat src_gray;
int thresh = 100;
RNG rng(12345);

vector<Point2f> getDimensions(vector<Vec3f> circles, int mX, int mY);

vector<Rect> getContours(Mat m);

int insideRect(vector<Vec3f> circles, vector<Rect> rects);

int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, dst2, th, masked, img_gray, result, end, color, color_seg, grey;
    const char* default_file = "legos.jpg";
    //default_file = "lego3.png";
    const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    Mat src = imread( samples::findFile( filename ), IMREAD_COLOR );

    float scale = 1;

    if (strcmp(default_file,"lego3.png") != 0)
        scale = 0.6;

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }

    VideoCapture video(0);

    video >> src;


    bilateralFilter(src, img_gray, 9, 301, 301, BORDER_DEFAULT);
    cv::pyrMeanShiftFiltering(src, img_gray, 40, 50, 2);
    cvtColor(img_gray, img_gray, COLOR_RGB2GRAY);

    cv::threshold(img_gray, th, 0, 255, THRESH_TOZERO_INV | THRESH_OTSU);
    cv::bitwise_and(img_gray, img_gray, img_gray, th);

    cvtColor(src, src, COLOR_BGR2HSV);
    cv::bitwise_and(src, src, masked, th);
    cv::pyrMeanShiftFiltering(masked, dst2, 40, 55, 2);

    // Show results

    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 1000,1000);
    imshow("Original", src);

    namedWindow("Gaussian Blur", WINDOW_NORMAL);
    resizeWindow("Gaussian Blur", 1000,1000);
    imshow("Gaussian Blur", img_gray);

    namedWindow("Threshold", WINDOW_NORMAL);
    resizeWindow("Threshold", 1000,1000);
    imshow("Threshold", th);

    namedWindow("Masked", WINDOW_NORMAL);
    resizeWindow("Masked", 1000,1000);
    imshow("Masked", masked);

    namedWindow("Mean-Shift Segmentation", WINDOW_NORMAL);
    resizeWindow("Mean-Shift Segmentation", 1000,1000);
    imshow("Mean-Shift Segmentation", dst2);


    //Get actual piece position
    Mat pins = masked.clone();

    fastNlMeansDenoising(pins,pins,10.1,10,7);

    //resize(pins, pins, Size(), scale, scale, INTER_LINEAR);

    namedWindow("denoise", WINDOW_NORMAL);
    resizeWindow("denoise", 1000,1000);
    imshow("denoise", pins);





    cvtColor(pins, pins, COLOR_HSV2BGR);


    Mat dest = Mat::zeros( pins.size(), pins.type() );
    double alpha = 4; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
    for( int y = 0; y < pins.rows; y++ ) {
        for( int x = 0; x < pins.cols; x++ ) {
            for( int c = 0; c < pins.channels(); c++ ) {
                dest.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*pins.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }

    pins = dest;


    cvtColor(pins,src_gray, COLOR_BGR2GRAY);


    medianBlur(src_gray, src_gray, 5);

    equalizeHist( src_gray, src_gray );
    namedWindow("hist", WINDOW_NORMAL);
    resizeWindow("hist", 1000,1000);
    imshow("hist", src_gray);

    vector<Rect> rectangles;
    rectangles = getContours(src);


    //namedWindow("cont", WINDOW_NORMAL);
    //resizeWindow("cont", 1000,1000);
    //imshow("cont", drawing);

    // 50, 35, 10, 25, 35

    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1,
                 150  ,  // change this value to detect circles with different distances to each other
                 20, 20, 60, 70 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( pins, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( pins, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }


    vector<Point2f> extr = getDimensions(circles, pins.cols, pins.rows);  



    line( pins,
        extr.at(0),
        extr.at(2),
        Scalar( 255, 0, 0 ),
        2,
        LINE_8 );

    line( pins,
        extr.at(0),
        extr.at(3),
        Scalar( 255, 0, 0 ),
        2,
        LINE_8 );

    int rect_piece = insideRect( circles , rectangles);

    cout << "Index = " << rect_piece << endl;

    pins = pins(rectangles.at(rect_piece));

    //for(int i = 0; i < circles.size(); i++)
    //{
    //    //cout << circles.at(i);
    //    cout << circles.at(i)[0] << "," << circles.at(i)[1] << endl;
    //    insideRect(Point2f (circles.at(i)[0], circles.at(i)[1]) , rectangles);
    //    //insideRect(Point2f (1071, 956), rectangles);
    //}

    //cout << src.cols;
    //cout << src.rows;


    namedWindow("detected circles", WINDOW_NORMAL);
    resizeWindow("detected circles", 1000,1000);
    imshow("detected circles", pins);


    //Color Detection
    vector<int> maxes;
    int max = 0, index;
    //-----------
    for(int i = 0; i < colors_names.size() - 1; i++){
        cv::inRange(
                dst2,
                Scalar(colors[i][0] - 5, colors[i][1] / 100 * 0.7 * 255 , colors[i][1] / 100 * 0.7 * 255 + 3 ),
                //Scalar(colors[i][0] - 15, 0, 5),
                Scalar(colors[i][0] + 5, colors[i][1] / 100 * 1.3 * 255 , colors[i][1] / 100 * 1.3 * 255 ),
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

    cv::cvtColor(dst2, dst2, COLOR_HSV2BGR);
    imshow("Mean-Shift Segmentation", dst2);


    Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 21, 21 ),
                                         Point( 10, 10 ) );
    namedWindow("Color segmentation", WINDOW_NORMAL);
    resizeWindow("Color segmentation", 1000,1000);
    imshow("Color segmentation", color);

    cv::dilate(color, color, element);
    cv::erode(color, color, element);

    cv::erode(color, color, element);
    cv::dilate(color, color, element);

    bitwise_and(src, src, color_seg, color);

    cout << "max = " << max << ", index = " << index << endl;
    cout << "color is " << colors_names[index] << endl;
    namedWindow("Segmentation", WINDOW_NORMAL);
    resizeWindow("Segmentation", 1000,1000);
    imshow("Segmentation", color_seg);
    if(!grey.empty()) {
        imshow("Grey", grey);
    }


    // Wait and Exit
    waitKeyEx();
    return 0;
}

vector<Point2f> getDimensions(vector<Vec3f> circles, int mX, int mY)
{


    Point2f minX(mX,0), maxX(0,0), minY(0,mY), maxY(0,0);
    for(int i = 0; i < circles.size(); i++)
    {
        Point2f p(circles.at(i)[0], circles.at(i)[1]);
        if( p.x > maxX.x){
            Point2f pf(p.x,p.y);
            maxX = pf;
        }
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

    vector<Point2f> vPoints(4);
    vPoints[0] = minX;
    vPoints[1] = maxX;
    vPoints[2] = minY;
    vPoints[3] = maxY;


    cout << minX.y << ","<< minX.x << endl;
    cout << maxY.y << "," << maxY.x << endl;
    double d1 = (minX.y - maxY.y)/(minX.x - maxY.x);
    double d2 = (minX.y - minY.y)/(minX.x - minY.x);
    double d;
    int l1 = 0, l2 = 0;
    for(int i = 0; i < circles.size(); i++){
        cout << circles.at(i)[0] << "  " << circles.at(i)[1] << "   "<< circles.at(i)[2] << endl;
        d = (minX.y - circles.at(i)[1])/(minX.x - circles.at(i)[0]);
        cout << "declive " << d << endl;
        if( d > 0 and d <= d1 + 0.01 and d >= d1 - 0.01){
            l1++;
        }
        else if(d > 0 and d <= d2 + 0.01 and d >= d2 - 0.01){
            l2++;
        }
        else if(d >= d1 - 0.01 and d <= d1 + 0.01){
            l1++;
        }
        else if(d >= d2 - 0.01 and d <= d2 + 0.01){
            l2++;
        }
    }

    cout << d1 << endl;
    cout << d2 << endl;
    cout << l1 << endl;
    cout << l2 << endl;

    return vPoints;
}



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
    Mat drawing = Mat::zeros( th.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours_poly, (int)i, color );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
    }

    namedWindow("draw", WINDOW_NORMAL);
    resizeWindow("draw", 1000,1000);
    imshow("draw", drawing);
    return boundRect;
}

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
