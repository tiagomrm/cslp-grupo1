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

double f(double x, double slope, double b);

void white_balance(Mat& src, Mat& dst, Mat& mask);

int main(int argc, char** argv)
{
    
    // Declare the output variables
    Mat dst, dst2, th, masked, img_gray, result, end, color, color_seg, grey, white_balanced;
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

    //VideoCapture video(0);

    //video >> src;

    cvtColor(src, img_gray, COLOR_BGR2GRAY);
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

    // Show results

    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 1000,1000);
    imshow("Original", src);
//
//    namedWindow("Gaussian Blur", WINDOW_NORMAL);
//    resizeWindow("Gaussian Blur", 1000,1000);
//    imshow("Gaussian Blur", img_gray);
//
//    namedWindow("Threshold", WINDOW_NORMAL);
//    resizeWindow("Threshold", 1000,1000);
//    imshow("Threshold", th);
//
    namedWindow("Masked", WINDOW_NORMAL);
    resizeWindow("Masked", 1000,1000);
    imshow("Masked", masked);
//
    namedWindow("Mean-Shift Segmentation", WINDOW_NORMAL);
    resizeWindow("Mean-Shift Segmentation", 1000,1000);
    imshow("Mean-Shift Segmentation", dst2);


    //Get actual piece position
    Mat pins = masked.clone();

    fastNlMeansDenoising(pins,pins,10.1,10,7);

    //resize(pins, pins, Size(), scale, scale, INTER_LINEAR);

//    namedWindow("denoise", WINDOW_NORMAL);
//    resizeWindow("denoise", 1000,1000);
//    imshow("denoise", pins);





    cvtColor(pins, pins, COLOR_HSV2BGR);


//    Mat dest = Mat::zeros( pins.size(), pins.type() );
//    double alpha = 4; /*< Simple contrast control */
//    int beta = 0;       /*< Simple brightness control */
//    for( int y = 0; y < pins.rows; y++ ) {
//        for( int x = 0; x < pins.cols; x++ ) {
//            for( int c = 0; c < pins.channels(); c++ ) {
//                dest.at<Vec3b>(y,x)[c] =
//                  saturate_cast<uchar>( alpha*pins.at<Vec3b>(y,x)[c] + beta );
//            }
//        }
//    }
//
//    pins = dest;


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

    Mat other;
    //dst2 = dst2(rectangles.at(rect_piece));
    imshow("aaaaaaaa", dst2);

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

    imshow("aaa depression", color);

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

    //src = src(rectangles.at(rect_piece));
    bitwise_and(src, src, color_seg, color);

    cout << "max = " << max << ", index = " << index << endl;
    cout << "color is " << colors_names[index] << endl;
    namedWindow("Segmentation", WINDOW_NORMAL);
    resizeWindow("Segmentation", 1000,1000);
    imshow("Segmentation", color_seg);
    if(!grey.empty()) {
        imshow("Grey", grey);
    }

    
    int dilation_size = 50;
    Mat dilation_dst;
    Mat element2 = getStructuringElement( MORPH_CROSS,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );
    dilate( color_seg, dilation_dst, element2 );

    namedWindow("Final", WINDOW_NORMAL);
    resizeWindow("Final", 1000,1000);
    imshow("Final", dilation_dst);


    Mat final;
    vector<vector<Point> > contours2;
    cvtColor(color_seg, color_seg, COLOR_RGB2GRAY);
    namedWindow("Final2", WINDOW_NORMAL);
    resizeWindow("Final2", 1000,1000);
    imshow("Final2", color_seg);
    findContours( color_seg, contours2, RETR_TREE, CHAIN_APPROX_SIMPLE );
    cout << contours2.size() << endl;
//    for (int i = 0; i< contours2.size(); i++){
//  
//        if (i == 0){
//            continue;
//        }
//  
//
//        vector<Point> contours_poly;
//        approxPolyDP( contours2.at(i), contours_poly, 3, true );
//
//
//        drawContours(final, contours_poly, 0, (0, 0, 255), 5);
//  
//        Moments mo = moments(contours2.at(i));
//        
//
//
//        if (mo.m00 != 0.0){
//            int x = (int)(mo.m10/mo.m00);
//            int y = (int)(mo.m01/mo.m00);
//        }
//
//        if (contours_poly.size() == 3){
//            //cv2.putText(img, 'Triangle', (x, y),
//            //            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
//            cout << "triangle" << endl;
//        }
//        else if ( contours_poly.size() == 4){
//            //cv2.putText(img, 'Quadrilateral', (x, y),
//            //            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
//            cout << "rectangle" << endl;
//        }
//        else if(contours_poly.size() == 5){
//            //cv2.putText(img, 'Pentagon', (x, y),
//            //            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
//            cout << "pentagon" << endl;
//
//        }
//        else if (contours_poly.size() == 6){
//            //putText(img, 'Hexagon', (x, y),
//            //            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
//            cout << "hexagon" << endl;
//        }
//        else{
//            //putText(img, 'circle', (x, y),
//            //            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
//            cout << "circle" << endl;
//        }
//    }

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
    double b1 = minX.y - (d1 * minX.x);
    double b2 = minX.y - (d2 * minX.x);
    double d;
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

    return vPoints;
}

double f(double x, double slope, double b){
    return x * slope + b;
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