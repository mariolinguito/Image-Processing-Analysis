#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std; 

Mat Binarization(Mat); 
Mat TransformationDistance4(Mat);
Mat TransformationDistance8(Mat);  

Mat Binarization(Mat src) {
    Mat outputSrc = Mat(src.size(), src.type(), Scalar::all(0)); 

    // We need to binarize the image. All the pixel's value that is greater 
    // than 128 is set to 255, otherwise is set to 0.
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(src.at<uchar>(i, j) > 128) {
                outputSrc.at<uchar>(i, j) = 255; 
            }
        }
    }

    // At the end we return the output.
    return outputSrc; 
}

Mat TransformationDistance4(Mat src) {
    Mat outputSrc;  

    // The algorithm is based on two scans, the first from top to bottom and 
    // from left to right. The structurant element is: 
    //
    // N2    N3    N4
	// N1    PX    N5
	// N8    N7    N6
    // 
    // The first element is N3, the second is N1, so we take the min from these 
    // elements (plus one), and assign it to the current element. 
    for(int i = 1; i < src.rows; i++) {
        for(int j = 1; j < src.cols; j++) {
            int firstElement = src.at<uchar>(i-1, j); 
            int secondElement = src.at<uchar>(i, j-1); 
            int minElement = min(firstElement, secondElement); 

            src.at<uchar>(i, j) = minElement+1; 
        }
    }

    // The first element is N7, and the second element is N5 (all plus one), the third
    // element is the current element. So we take the min element between the first element 
    // and the second element. The second min is the min between the first min computed, and 
    // the current element. At the end, we set the current element with the second min plus one. 
    for(int i = src.rows-1; i >= 0; i--) {
        for(int j = src.cols-1; j >= 0; j--) {
            int firstElement = src.at<uchar>(i+1, j)+1;
            int secondElement = src.at<uchar>(i, j+1)+1; 
            int thirdElement = src.at<uchar>(i, j); 

            int firstMin = min(firstElement, secondElement); 
            int secondMin = min(firstMin, thirdElement); 

            src.at<uchar>(i, j) = secondMin+1; 
        }
    }

    // We normalize the result. 
    normalize(src, outputSrc, 0, 255, NORM_MINMAX); 

    return outputSrc; 
}

// The same thing we can do for the algorithm based on 8-connected. Instead of 
// take only two elements, in this case we take four elements, and of these we 
// found the minimum element bewtween these plus one, in the direct scan.
Mat TransformationDistance8(Mat src) {
    Mat outputSrc; 

    for(int i = 1; i < src.rows; i++) {
        for(int j = 1; j < src.cols; j++) {
            int N1 = src.at<uchar>(i, j-1);
            int N2 = src.at<uchar>(i-1, j-1); 
            int N3 = src.at<uchar>(i-1, j);
            int N4 = src.at<uchar>(i-1, j+1);

            int minElement_1 = min(N1, N2);  
            int minElement_2 = min(N3, N4); 

            src.at<uchar>(i, j) = min(minElement_1, minElement_2)+1;  
        }
    }

    // In the inverse scan we take the other four elements, and consequently the 
    // minimum element between these, and after the minimum element between the 
    // absolute minimum element and the previously element (found thanks to the 
    // first scan). 
    for(int i = src.rows-1; i >= 0; i--) {
        for(int j = src.cols-1; j >= 0; j--) {
            int N5 = src.at<uchar>(i+1, j+1);
            int N6 = src.at<uchar>(i+1, j);
            int N7 = src.at<uchar>(i+1, j-1);
            int N8 = src.at<uchar>(i, j+1);

            int minElement_1 = min(N5, N6);  
            int minElement_2 = min(N7, N8); 

            int absoluteMin = min(minElement_1, minElement_2); 

            src.at<uchar>(i, j) = min(absoluteMin, (int)src.at<uchar>(i, j))+1; 
        }
    }

    // At the end we normalize the result. 
    normalize(src, outputSrc, 0, 255, NORM_MINMAX); 

    return outputSrc; 
}

int main(int argc, char *argv[]) {
    // Get the image from command line, in particular the name of the image, 
    // that is the input of imread() function. The image is read in grayscale
    // because we need to binarize it. 
    String inputName = argv[1]; 
    Mat inputImage = imread(inputName, IMREAD_GRAYSCALE); 
    
    // These istructions show image in a window. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    // We need to binarize the input image. 
    Mat binarizedImage = Binarization(inputImage); 

    // These istructions show image in a window.
    namedWindow("Binarized Image", WINDOW_AUTOSIZE); 
    imshow("Binarized Image", binarizedImage); 

    // This function is for the transformation distance with 4-connected element.  
    Mat transformationImage4 = TransformationDistance4(binarizedImage); 

    // These istructions show image in a window.
    namedWindow("Transformation Image 4-Connected", WINDOW_AUTOSIZE); 
    imshow("Transformation Image 4-Connected", transformationImage4); 

    // This function is for the transformation distance with 8-connected element.  
    Mat transformationImage8 = TransformationDistance8(binarizedImage); 

    // These istructions show image in a window.
    namedWindow("Transformation Image 8-Connected", WINDOW_AUTOSIZE); 
    imshow("Transformation Image 8-Connected", transformationImage8);

    waitKey(0); 
    return 0; 
}