#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv; 
using namespace std; 

Mat Threshold(Mat& src, int thrValue);

Mat Threshold(Mat& src, int thrValue) {
    // La funzione restituisce in output un'immagine che ha le stesse 
    // dimensioni ed è dello stesso tipo dell'immagine di partenza. 
    Mat destImg(src.rows, src.cols, src.type());

    // Si deve scorrere tutta l'immagine per colonne e per righe per 
    // poter confrontare il valore del pixel con il valore di sogliatura 
    // che è stato scelto in input. Se il valore è minore di quello di 
    // sogliatura, allora viene assegnato 0 a quel pixel, altrimenti 
    // gli viene assegnato il valore massimo (255).
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(src.at<uchar>(i, j) < thrValue) {
                destImg.at<uchar>(i, j) = 0; 
            } else {
                destImg.at<uchar>(i, j) = 255; 
            }
        }
    }

    // Alla fine si restituisce l'immagine ottenuta come risultato. 
    return destImg; 
}

int main(int argc, char *argv[]) {
    // Si prende in input il nome dell'immagine da modificare, ed 
    // anche il valore di sogliatura scelto. 
    string inputFile = argv[1]; 
    int thrValue = atoi(argv[2]);     

    // Si legge l'immagine, che ricordiamo essere in bianco e nero. 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);

    // E si richiama la funzione Threshold per poter effettuare
    // l'omonima operazione sull'immagine data in input. 
    Mat resultImage; 
    resultImage = Threshold(inputImage, thrValue);

    // Alla fine si mostrano in apposite finestre sia l'immagine 
    // originale che l'immagine risultante dall'operazione. 
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inputImage);

    namedWindow("Thresholded Image", WINDOW_AUTOSIZE);
    imshow("Thresholded Image", resultImage);

    waitKey(0); 

    return 0; 
}