#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv; 

Mat Average(Mat& src);
Mat Median(Mat& src);

Mat Average(Mat& src) {
    // Questa dichiarata è la maschera che conterrà gli elementi selezionati 
    // dall'immagine originale. In questo caso la maschera è 3x3. 
    Mat maskFromSrc; 
    Mat resultImg(src.rows, src.cols, src.type()); 
    int summary = 0; 
    
    // Con i primi due cicli for annidati scorriamo sulle righe e sulle colonne 
    // dell'immagine originale. Ad ogni passo del ciclo for più interno assegnamo 
    // alla maschera solo una specifica porzione dell'immagine originale, definita 
    // dal range esplicitato. 
    for(int i = 0; i < src.rows-2; i++) {
        for(int j = 0; j < src.cols-2; j++) {
            maskFromSrc = src(Range(i, i+3), Range(j, j+3));

            // I due cicli for annidati più interni scorrono sulla maschera, sommano 
            // quindi gli elementi della maschera ed inseriscono il risultato nella 
            // variabile intera summary. 
            for(int z = 0; z < maskFromSrc.rows; z++) {
                for(int x = 0; x < maskFromSrc.cols; x++) {
                    summary += static_cast<int>(maskFromSrc.at<uchar>(z, x)); 
                }
            } 

            // Alla fine summary deve essere divisa per la grandezza della maschera,
            // che in questo caso è 3x3, quindi 9. Il risultato deve essere il pixel 
            // centrale dell'immagine risultante. 
            summary = static_cast<int>(summary/9); 
            resultImg.at<uchar>(i+1, j+1) = static_cast<int>(summary); 
            summary = 0;  
        }
    }

    // Alla fine si restituisce l'immagine risultante. 
    return resultImg; 
}

Mat Median(Mat& src) {
    Mat maskFromSrc; 
    Mat resultImg(src.rows, src.cols, src.type());

    // Nell'applicazione del filtro mediana abbiamo bisogno di una variabile 
    // per il mediano di un vettore, ed appunto un vettore di dimensione pari 
    // alla grandezza della maschera, in questo caso 3x3, quindi 9. 
    int medianValue = 0; 
    vector<int> medianVector(9); 

    // I due cicli for annidati scorrono sull'immagine originale, e creano la 
    // maschera di dimensione 3x3 con gli elementi presi dall'immagine in input. 
    for(int i = 0; i < src.rows-2; i++) {
        for(int j = 0; j < src.cols-2; j++) {
            maskFromSrc = src(Range(i, i+3), Range(j, j+3));

            // A questo punto, con i due cicli annidati si vanno a selezionare 
            // tutti gli elementi della maschera e li si inserisce all'interno 
            // del vettore, dalla testa o dalla coda, non ha importanza dato che 
            // saranno in seguito ordinati. 
            for(int z = 0; z < maskFromSrc.rows; z++) {
                for(int x = 0; x < maskFromSrc.cols; x++) {
                    medianVector.push_back((maskFromSrc.at<uchar>(z, x))); 
                }
            }

            // Per poter scegliere l'elemento mediano abbiamo bisogno di ordinare 
            // il vettore, e lo facciamo con la funzione di c++ sort(). 
            sort(medianVector.begin(), medianVector.end());

            // In seguito l'elemento mediano lo si ottiene dividendo la grandezza 
            // del vettore per 2 e sommandola ad 1. L'elemento selezionato viene 
            // inserito in medianValue. 
            medianValue = medianVector.at((medianVector.size()/2)+1);

            // Il pixel centrale dell'immagine adesso è quell'elemento mediano. 
            resultImg.at<uchar>(i+1, j+1) = static_cast<int>(medianValue); 

            // Si resetta il valore medianValue a 0, cosi come si cancellano tutti 
            // gli elementi presenti nel vettore. 
            medianValue = 0; 
            medianVector.clear();
        }
    }

    // Alla fine si restituisce l'immagine risultante. 
    return resultImg; 
}

int main(int argc, char *argv[]) {
    string inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE); 

    // Richiamiamo la funzione Average su inputImage, ed assegniamo 
    // il suo risultato all'oggetto average. 
    Mat average; 
    average = Average(inputImage); 

    // Richiamiamo la funzione Median su inputImage, ed assegniamo 
    // il suo risultato all'oggetto median. 
    Mat median; 
    median = Median(inputImage); 
    
    // Alla fine mostriamo tutte le immagini (compresa quella originale)
    // in apposite finestre. 
    namedWindow("Average Image", WINDOW_AUTOSIZE);
    imshow("Average Image", average);

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inputImage);

    namedWindow("Median Image", WINDOW_AUTOSIZE);
    imshow("Median Image", median);

    waitKey(0); 

    return 0; 
}