#include <cstdio>
#include <iostream>
#include <stack>
#include <unistd.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;

// Definiamo una classe per le informazioni riguardanti il punto 
// che sarà utilizzata nel corso dell'elaborazione dell'algoritmo. 
class RegionPoint {
    public: 
    int x; 
    int y; 

    RegionPoint(int x, int y) {
        this->x = x; 
        this->y = y; 
    }
}; 

Mat StartGrow(Mat& src, int thr);

Mat StartGrow(Mat& src, int thr) {
    // La prima matrice conterrà le regioni che sono state trovate, la
    // seconda matrice conterrà i pixel che sono stati visitati o meno. 
    // Nel senso: se un pixel è stato visitato allora viene posto a 1, 
    // se invece non è stato visitato, viene posto a 0. Tutte e due le 
    // matrici sono impostate a 0. 
    Mat clonedSrc = Mat(src.size(), src.type(), Scalar::all(0));
    Mat regionSrc = Mat(src.size(), CV_32F, Scalar::all(0));

    // Lo stack viene utilizzato per tenere traccia di quei punti che 
    // dovranno essere considerati per la regione che cresce. 
    stack<RegionPoint> regionsPoints; 

    // Si scorre tutta l'immagine per poter selezionare il seed, ossia 
    // il seme iniziale che deve dare origine alla regione. 
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // Il punto non è stato visitato, quindi non appartiene ad una 
            // specifica regione, allora viene considerato per essere il 
            // seed della nuova regione. Quindi si crea un nuovo oggetto di 
            // tipo RegionPoint e poi si inizia a far crescere la regione. 
            if(regionSrc.at<float>(i, j) == 0) {
                RegionPoint seed = RegionPoint(i, j);

                // Al punto che viene considerato come seed viene associato lo stesso 
                // valore del pixel che si trova nella stessa posizione del seed. Il 
                // punto in questione viene poi inserito all'interno dello stack. 
                clonedSrc.at<uchar>(seed.x, seed.y) = src.at<uchar>(seed.x, seed.y);
                regionSrc.at<float>(seed.x, seed.y) = 1;                  
                regionsPoints.push(seed);

                // Fintanto che lo stack non è vuoto, quindi non ci sono punti da considerare 
                // e la regione non può più crescere, si deve continuare a fare crescere la 
                // regione considerando i punti che vengono ad essa aggiunti (cioè i punti che 
                // contornano il punto che abbiamo preso in considerazione e che non sono stati 
                // visitati o inseriti in qualche regione). 
                while(!regionsPoints.empty()) {
                    // Si estrae un pixel dallo stack, e lo si inserisce in currentPoint, poi 
                    // viene estratto il punto dallo stack per eliminarlo dalla struttura dati. 
                    RegionPoint currentPoint = regionsPoints.top();
                    regionsPoints.pop();   

                    // I due cicli for vengono impiegati per poter scorrere sui pixel che sono di 
                    // contorno al pixel che stiamo tenendo in considerazione. 
                    for(int x = -1; x <= 1; x++) {
                        for(int y = -1; y <= 1; y++) {
                            // Se le coordinate del pixel non sforano le dimensioni della matrice
                            // allora può essere considerato, altrimenti ovviamente non può essere 
                            // considerato nella regione. 
                            if((currentPoint.x)+x >= 0 && (currentPoint.x)+x < src.rows && (currentPoint.y)+y >= 0 && (currentPoint.y)+y < src.cols) { 
                                // Se il punto di contorno non è stato considerato allora esso è pari a 0, 
                                // altrimenti non deve più essere considerato.   
                                if(regionSrc.at<float>((currentPoint.x)+x, (currentPoint.y)+y) == 0) {
                                    // Il criterio grazie al quale si capisce se un pixel appartiene o meno alla regione è di 
                                    // calcolare la differenza tra il valore del seed e quello del punto che stiamo considerando
                                    // di contorno, e se questa differenza è minore di thr (che è il valore di threshold). 
                                    if(abs(src.at<uchar>(seed.x, seed.y) - src.at<uchar>((currentPoint.x)+x, (currentPoint.y)+y)) < thr) { 
                                        // Se tutti i criteri sono stati soddisfatti, allora viene impostato il pixel come 
                                        // visitato, e poi gli viene assegnato il valore del seed, per poter ovviamente creare
                                        // la regione. 
                                        regionSrc.at<float>((currentPoint.x)+x, (currentPoint.y)+y) = 1; 
                                        clonedSrc.at<uchar>((currentPoint.x)+x, (currentPoint.y)+y) = src.at<uchar>(seed.x, seed.y); 

                                        // Il punto che ha superato tutti i test viene inserito nello stack, in modo tale
                                        // che i pixel che lo circondano e che non sono stati ancora visitati vengono 
                                        // considerati per poter essere inseriti nella regione in questione. 
                                        RegionPoint visitedPoint = RegionPoint((currentPoint.x)+x, (currentPoint.y)+y);
                                        regionsPoints.push(visitedPoint);
                                    }
                                }
                            } 
                        }
                    }  
                }
            }                    
        }
    }

    // Il risultato potrebbe contenere del rumore, quindi è opportuno
    // applicargli un filtro per lo smoothing, e scegliamo quello di 
    // Gauss, utilizzando la funzione contenuta in OpenCV. 
    GaussianBlur(clonedSrc, clonedSrc, Size(3, 3), 3, 3);

    // Alla fine restituiamo il risultato finale. 
    return clonedSrc; 
}

int main(int argc, char *argv[]) {
    // Leggiamo l'immagine da riga di comando. 
    string inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);

    // Si mostra l'immagine originale in una apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage);

    // Si applica la funzione per il Region Growing con in input 
    // l'immagine e il valore di thresholding. 
    Mat regionGrowingImage = StartGrow(inputImage, atoi(argv[2]));  

    // Il risultato dell'operazione viene mostrato in una
    // apposita finestra. 
    namedWindow("Region Growing Image", WINDOW_AUTOSIZE); 
    imshow("Region Growing Image", regionGrowingImage);

    waitKey(0); 

    return 0;  
}