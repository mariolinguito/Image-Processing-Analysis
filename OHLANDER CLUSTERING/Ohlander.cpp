#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;

// We describe the functions to use in the algorithm. 
void Ohlander(Mat, int);
pair<int, int> ComputeMidPoint(Mat, Mat, int); 
void ComputeMasks(Mat, Mat, int, int); 

// channels is relative to the three channels of the color RGB, 
// channelsMat is relative to the output of calcHist() function 
// of openCV to compute the histogram of an image (in this case
// the histogram of the three colors).  
Mat channels[3]; 
Mat channelsMat[3]; 

// resultsMask is the vector that contains the mask that correspond
// to the cluster of the image. In particular we push a mask into 
// this vector when we cannot split its histogram into two masks 
// because the histogram has not two peak.  
vector<Mat> resultsMask; 
vector<Mat> maskToApply;

// These variables are relative to the calcHist function that compute 
// the histogram with a particular mask. 
int histSize = 256;
float range[] = { 0, 256 } ;
const float* histRange = { range };
bool uniformHist = true; 
bool accumulateHist = false;

void Ohlander(Mat src, int thr) {
    // The algorithm extract from vector the mask until he is empty, 
    // and then he pop the element from the same vector. 
    while(!maskToApply.empty()) {
        Mat extractedMask = maskToApply.back();
        maskToApply.pop_back(); 
 
        // We use these variables to track the information of the max peak, 
        // the max valley of the histogram R or G or B, and its index. The
        // boolean variable say the fact that the histogram should be splitted 
        // again into two masks. 
        int maxPeak = 0; 
        int maxValley = 0; 
        int index = 0; 
        bool found = 0;  

        // The pair track the valley and the max peak of the current histogram, 
        // so we can choose that histogram to split it into two masks, as we say 
        // previously. 
        pair<int, int> computedMidPoint; 

        for(int i = 0; i < 3; i++) {
            // For every channel we compute the histogram with the mask extracted previously, and put 
            // the result into the channelsMat current variable. 
            calcHist(&channels[i], 1, 0, extractedMask, channelsMat[i], 1, &histSize, &histRange, uniformHist, accumulateHist);
            
            // Then, we need to compute the middle point between two peak, it is called valley, 
            // for the specific histogram. 
            computedMidPoint = ComputeMidPoint(channelsMat[i], src, thr); 

            // If the valley [or the peak] is lower than zero, we can put the mask 
            // computed with the function ComputeMidPoint() into the vector. Otherwise, 
            // we need to track the information about the peak and the valley and choose 
            // the max peak. 
            if(computedMidPoint.first <= 0 || computedMidPoint.second <= 0) {
                resultsMask.push_back(extractedMask); 
            } else if(computedMidPoint.second > maxPeak) {
                maxValley = computedMidPoint.first;
                maxPeak = computedMidPoint.second;  
                index = i; 
                found = true; 
            }
        }

        // If the mask isn't a "result mask" we need to split the mask into two masks, 
        // and restart the whole process because the functions push two new masks into 
        // the vector. 
        if(found) {
            ComputeMasks(channelsMat[index], src, maxValley, index); 
        }
    }
}

pair<int, int> ComputeMidPoint(Mat histogram, Mat src, int thr) {
    // The theory says that might be more than two peak, and at the 
    // same time more than one valley, so we use a blur on the histogram 
    // to smooth the peak and the valley, with an advance in computation time.  
    GaussianBlur(histogram, histogram, Size(55, 55), 0, 0);
     
    // The map track the value of every point into the histogram, and its 
    // index, so we can choose the first max peak and the second max peak, 
    // and conseguently the valley between them. We need to remeber that an 
    // histogram is a vector of integers. 
    map<int, int> histMap; 
    for(int i = 0; i < histogram.rows; i++) {
        histMap.insert(make_pair(histogram.at<float>(i, 0), i)); 
    }

    int firstPeak = 0; 
    int secondPeak = 0; 

    // To search the first peak on the left, we iterate through the histogram's
    // values until there are some value lower than the current peak. 
    int firstPeakIndex = 0; 
    while(histogram.at<float>(firstPeakIndex, 0) >= firstPeak) {
        firstPeak = histogram.at<float>(firstPeakIndex, 0);  
        firstPeakIndex++; 
    }

    // The same thing we make with the second peak on the right, the while 
    // iterate until the histogram value is grater than second peak and the 
    // index is grater than the first peak index. 
    int secondPeakIndex = histogram.rows-1;
    while(histogram.at<float>(secondPeakIndex, 0) >= secondPeak && secondPeakIndex > firstPeakIndex) {
        secondPeak = histogram.at<float>(secondPeakIndex, 0);
        secondPeakIndex--; 
    }

    // The pair refers to the return value of the function, the first element
    // corresponds to the valley, the second to the max peak of the histogram.
    // We create the pair only when the distance between the two peak is lower 
    // than a threshold that we give in input.  
    pair<int, int> toReturn; 

    // We compute the valley and the max of two peaks. 
    int valley = (histMap[firstPeak] + histMap[secondPeak])/2; 
    int peak = max(firstPeak, secondPeak);

    // If the peaks are too close, so if the differences between the two peaks 
    // is lower than the threshold, we should return -1 and -1. Otherwise 
    // we make the pair with the value computed. 
    if(abs(histMap[firstPeak] - histMap[secondPeak]) < thr) {
        toReturn = make_pair(-1, -1); 
    } else { 
        toReturn = make_pair(valley, peak); 
    }

    return toReturn; 
}

void ComputeMasks(Mat histogram, Mat src, int valley, int index) { 
    // We should create two masks, that are the clone of the principal 
    // histogram, because we need to split this. 
    Mat left = histogram.clone(); 
    Mat right = histogram.clone(); 

    // In the first mask we make zero all the value between the valley 
    // and the histogram's rows, so we have the right mask. 
    for(int i = valley; i < histogram.rows; i++) {
        left.at<float>(i, 0) = 0; 
    }

    // We reverse the operation for the right mask, because we start from 
    // the histogram's rows to the valley. 
    for(int i = histogram.rows-1; i >= valley; i--) {
        right.at<float>(i, 0) = 0; 
    }

    Mat leftMask = Mat(src.size(), CV_8UC1, Scalar::all(0)); 
    Mat rightMask = Mat(src.size(), CV_8UC1, Scalar::all(0)); 

    // We iterate only on the histogram (R, G or B) that are the highest 
    // peak. It is identified by the integer index. Remember that the 
    // channels is obtained from the split function. 
    for(int i = 0; i < channels[index].rows; i++) {
        for(int j = 0; j < channels[index].cols; j++) {
            int intensity = channels[index].at<uchar>(i, j); 
            
            // If the intensity of element (i, j) into the indexed channels 
            // is greater than zero, we should set 1 into the leftMask. Otherwise
            // we should set 0, because it will not be considered. 
            if(left.at<float>(intensity, 0) > 0) {
                leftMask.at<uchar>(i, j) = 1; 
            } else {
                leftMask.at<uchar>(i, j) = 0; 
            }

            // With the same logical reasoning we can create the right mask. 
            if(right.at<float>(intensity, 0) > 0) {
                rightMask.at<uchar>(i, j) = 1; 
            } else {
                rightMask.at<uchar>(i, j) = 0; 
            }
        }
    }

    // At the end we push into the vector the two masks founded. 
    maskToApply.push_back(leftMask); 
    maskToApply.push_back(rightMask); 
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        cout << "Usage: ./[program-name] [image-name].[image-format] [threshold]" << endl;
        exit(0);  
    }

    // We read the input image (colored). 
    String inputName = argv[1]; 
    Mat inputImage = imread(inputName, IMREAD_COLOR);

    // To remove the eventual noise we can apply on the input image
    // a GaussianBlur to smooth the original image. 
    GaussianBlur(inputImage, inputImage, Size(5, 5), 8); 

    // This function split the input image into its three component: R, G, B. 
    split(inputImage, channels); 

    // The first mask to push into the vector is the mask that cover all the image.  
    Mat firstMask = Mat(inputImage.size(), CV_8UC1, Scalar::all(1)); 
    maskToApply.push_back(firstMask); 

    // We read the threshold. 
    int thr = atoi(argv[2]);

    // We call the algorithm.  
    Ohlander(inputImage, thr); 

    // This function show the image into a window.  
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inputImage); 
    
    // At the end we declare the output matrix for the image. 
    Mat outputImage = Mat(inputImage.size(), inputImage.type(), Scalar::all(0)); 
    RNG rng(12345); 

    // For the results masks we make a random color and we assign it to the different 
    // regions founded with Ohlander.  
    for(int i = 0; i < resultsMask.size(); i++) {
        Vec3b color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        for(int x = 0; x < inputImage.rows; x++) {
            for(int y = 0; y < inputImage.cols; y++) {
                if(resultsMask.at(i).at<uchar>(x, y) != 0) {
                    outputImage.at<Vec3b>(x, y) = color; 
                }
            }
        }
    }

    // The result is shown into a specific window.  
    namedWindow("Computed Image", WINDOW_AUTOSIZE);
    imshow("Computed Image", outputImage);
    
    waitKey(0);  

    return 0; 
}
