#include <opencv2/opencv.hpp>
#include <iostream>
#include <colors.h>
#include <vector>

using namespace std;
using namespace cv;



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

    //for(int i = 0; i< colors_names.size(); i++){
    //    cout << "{" << colors[i][0]/2 << "," << colors[i][1] << "," <<  colors[i][2] << "}," << endl; 
//
    //}

    CommandLineParser parser( argc, argv, "{@input |  lena.jpg | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
    if( src.empty() )
    {
        return EXIT_FAILURE;
    }

    //VideoCapture cameraCapture(0);

    Mat p1;
    //cameraCapture >> p1;

    namedWindow("first");
    imshow("first", p1);
    cout<<"cringe";

    //char key2 = (char) waitKey(0);
    //cameraCapture(0);

    cout << "cringe";
    //cameraCapture >> src;
    //cameraCapture >> src;
    //cameraCapture >> src;
    //cameraCapture >> src;
    //cameraCapture >> src;

    Mat out;
    Mat mask;
    int dtype = -1;
    cv::absdiff(p1,src, out);
    cv::Mat grayscale;
    //cv::cvtColor(out, grayscale, cv::COLOR_BGR2GRAY);

    // create a mask that includes all pixel that changed their value
    cv::Mat mask2 = out>0;

    cv::Mat output;
    out.copyTo(output,mask2);

    Mat c ;
    //bitwise_and(output,src,c);
    absdiff(src, p1, c);

    namedWindow("difference");
    imshow("difference", output);

    namedWindow("difference2");
    imshow("difference2", src);


    namedWindow("bit");
    imshow("bit", c);


    Mat hsv_image;
    cvtColor(src,hsv_image, COLOR_BGR2HSV);

    Mat frame, frame_HSV, frame_threshold;


    //namedWindow(window_capture_name);
    //namedWindow(window_detection_name);

    const String window_reverse = "Reverse";
    namedWindow(window_reverse);
    //imshow(window_reverse,src);

    Mat blured;
    //for ( int i = 1; i < 30; i = i + 2 )
    //{
    //    GaussianBlur( hsv_image, blured, Size( i, i ), 0, 0)    ;
//
    //}

    cv::pyrMeanShiftFiltering(hsv_image, blured, 40, 50, 2);
    const String win_erosion = "Result";
    namedWindow(win_erosion);
    imshow(win_erosion, blured);
    //---------


    //Mat res;
    //int erosion_size = 1;
    //Mat element = getStructuringElement( 0,
    //                Size( 2*erosion_size + 1, 2*erosion_size+1 ),
    //                Point( erosion_size, erosion_size ) );
//
    //dilate( src, res, element );
    //const String win_erosion = "Result";
    //namedWindow(win_erosion);
    //imshow(win_erosion, res);


    Mat end;
    int max = 0;
    int index = -1;
    Mat top;

    vector<int> maxes;
    //-----------
    for(int i = 0; i < colors_names.size() - 1; i++){
        cout <<  i << endl;
        inRange(blured, Scalar(colors[i][0] - 5, colors[i][1] - 50, colors[i][2] - 20), Scalar(colors[i][0] + 5, colors[i][1] + 50,  colors[i][2] + 20),  end);
        int count = checkColor(end);
        if(count > 1000 && count >= max){
            max = count;
            index = i;
            top = end.clone();
            maxes.push_back(i);
        }

    }
    // Show the frames
    //imshow(window_capture_name, blured);
    //imshow(window_reverse, top);


    cout << "max = " << max << ", index = " << index << endl;
    cout << "color is " << colors_names[index] << endl;

    for(int j = 0; j < maxes.size(); j++){
        cout << "   " << colors_names[maxes[j]] << endl;
    }

    char key = (char) waitKey(0);

    return 0;
}