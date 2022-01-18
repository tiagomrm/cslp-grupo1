#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src_gray;
RNG rng(12345);
int thresh = 100;

void thresh_callback(int, void*);

int main(int argc, char ** argv){
    VideoCapture cameraCapture(0);


    const char* camera = "Webcam";

    while (true){

        namedWindow(camera, WINDOW_AUTOSIZE);

        Mat src;

        cameraCapture >> src;

        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        blur( src_gray, src_gray, Size(3,3) );

        namedWindow("blured");
        imshow("blured", src_gray);

        const int max_thresh = 255;
        createTrackbar( "Canny thresh:", camera, &thresh, max_thresh, thresh_callback );
        thresh_callback( 0, 0 );

        //Mat hsv = cvtColor(src, COLOR_BGR2HSV);
        
        // Threshold of blue in HSV space
        //Scalar lower_blue = Scalar(60, 35, 140);
        //Scalar upper_blue = Scalar(180, 255, 255);
    
        // preparing the mask to overlay
        //Mat mask = inRange(hsv, lower_blue, upper_blue);
        
        // The black region in the mask has the value of 0,
        // so when multiplied with original image removes all non-blue regions
        //Mat result = bitwise_and(src, src, mask = mask);

        while(1){
            imshow(camera, src);

            if(waitKey(10) >= 0)
                break;
        }

    }
    return 0;

}

void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
    imshow( "Contours", drawing );
}