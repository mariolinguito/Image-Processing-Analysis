#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv; 

Mat HoughTransformation(Mat src, Mat rst, int thr); 

// L'input della funzione è praticamente l'immagine che contiene i contorni 
// ottenuti attraverso l'algoritmo di edge detection (come ad esempio Canny). 
// Poi l'immagine originale, ed il threshold con il quale confrontare gli 
// elementi dello spazio dei parametri (matrice dei voti). 
Mat HoughTransformation(Mat src, Mat rst, int thr) {
    // Dobbiamo dare una grandezza adeguata alla matrice dei voti per evitare 
    // qualsiasi tipo di problematica e per permettere alla matrice di contenere 
    // tutte le linee che vengono trovate. Quindi di solito, la grandezza massima R
    // viene calcolata moltiplicando il numero di righe o di colonne (in relazione a
    // quale sia il massimo) per la radice quadrata di 2. 
    double R = max(src.rows, src.cols)*(sqrt(2)); 

    // RHO e THETA sono gli elementi che compongono le coordinate polari, che ricordiamo 
    // sono ottenute con la relazione: RHO = X*sin(THETA) + Y*cos(THETA). Sappiamo che 
    // un punto all'interno dell'immagine corrisponde ad una superficie all'interno dello
    // spazio dei parametri, quindi dei punti che appartengono alla stessa curva nello 
    // spazio dell'immagine corrispondono a più superifici che in un determinato punto si 
    // intersecano tra loro. Il fatto che utilizziamo appunto lo spazio di Hough per 
    // trovare delle linee nelle immagini ci riporta ad un semplice problema di ricerca di 
    // intersezioni tra queste superifici, e di conseguenza si riduce tutto alla ricerca di 
    // massimi locali all'interno della matrice dei voti perchè quelli sicuramente identificano 
    // una curva all'interno dello spazio immagine.
    // Utilizziamo delle coordinate polari perchè una classica retta è caratterizzata dalla 
    // equazione y = mx+q, dove m è il coefficiente angolare della retta e q è la distanza della 
    // stessa dal centro del piano. Questi valori però oscillano da +infinito e -infinito, il 
    // che li rende poco pratici all'interno di un contesto discreto come quello di un'immafine
    // definita da una larghezza e lunghezza precisa. 
    double rho;
    double theta;

    // Dobbiamo prima di tutto definire una matrice dei voti, che sono inizializzate a 0, 
    // perchè le sue celle vengono incrementate ad ogni passaggio di una determinata
    // superficie, questo ci permette di trovare i massimi locali. 
    Mat VotingMatrix = (Mat_<uchar>(R, 180)); 
    VotingMatrix = Mat::zeros(VotingMatrix.size(), CV_8U); 

    // Si scorre tutta l'immagine originale per poter trovare dei punti salienti, ossia 
    // dei punti relativi a dei bordi. Questi hanno valore maggiore di 250 dato che abbiamo 
    // utilizzato Canny per poterli ottenere. Se ci troviamo in quel caso, allora dobbiamo 
    // considerare in quel punto tutte quelle rette (grado per grado) che contengono quel 
    // punto stesso, in modo tale da ottenere per quel punto tutta la superficie che lo 
    // caratterizza all'interno dello spazio dei parametri. 
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(src.at<uchar>(i, j) > 250) {
                for(int x = 0; x < 180; x++) {
                    // Viene calcolato il parametro RHO con la formula esplicitata in 
                    // precedenza, ed otteniamo in questo modo uno dei parametri delle 
                    // coordinate polari. 
                    rho = cvRound(j*cos((x)*(CV_PI/180)) + i*sin((x)*(CV_PI/180)));

                    // In seguito si va ad incrementare la specifica cella nella matrice
                    // dei voti nella posizione RHO e THETA. In quella cella passa la 
                    // superficie che descrive il punto dell'immagine originale. 
                    VotingMatrix.at<uchar>(rho, x)++; 
                }
            }
        }
    } 

    // Ottenuta la matrice dei voti, a questo punto la si deve
    // scorrere per poter andare a trovare quei punti che hanno 
    // un valore uguale o maggiore del threshold, ossia della 
    // soglia che abbiamo dato in input.
    for(int i = 0; i < VotingMatrix.rows; i++) {
        for(int j = 0; j < VotingMatrix.cols; j++) {
            if(VotingMatrix.at<uchar>(i, j) >= thr) {
                // Definiamo delle variabili per conservare il valore 
                // di RHO ed il valore di THETA. 
                double r = i; 
                double t = j*(CV_PI/180); 

                // Questi che abbiamo ottenuto sono i valori relativi alle 
                // coordinate polari. In particolare l'ascissa la si ottiene 
                // moltiplicando RHO per il coseno di THETA, viceversa 
                // l'ordinata la si ottiene moltiplicando RHO per il seno di 
                // THETA. 
                int x = cvRound(r * cos(t)); 
                int y = cvRound(r * sin(t)); 

                // Degli oggetti di tipo Point sono definiti con dei parametri 
                // (x, y) per ogni punto. 
                Point P1, P2; 

                // Nel primo punto andiamo a definire le coordinate cartesiane
                // relative ad esso. 
                P1.x = cvRound(x + 1000 * -(sin(t))); 
                P1.y = cvRound(y + 1000 * (cos(t))); 

                // Facciamo lo stesso con il secondo punto. 
                P2.x = cvRound(x - 1000 * -(sin(t))); 
                P2.y = cvRound(y - 1000 * (cos(t))); 

                // Alla fine si va a disegnare la linea passante per i due punti. 
                // Infatti per poter disegnare una retta bastano due punti, perchè 
                // per due punti passa una ed una sola retta. 
                line(rst, P1, P2, Scalar(0, 0, 255), 1, LINE_AA);
            }
        }
    }

    // Alla fine si restituisce l'immagine finale con le linee disegnate. 
    return rst; 
}

int main(int argc, char *argv[]) {
    // Si prende in input da riga di comando l'immagine sulla quale 
    // applicare l'algoritmo. Questa immagine originale viene clonata 
    // in una nuova immagine sulla quale saranno disegnate le rette.
    String inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE); 
    Mat clonedImage = inputImage.clone(); 

    // Si mostra l'immagine originale in una apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    // Per poter ottenere un risultato ottimale bisogna applicare un filtro 
    // di smoothing all'immagine, come quello di Gauss, che in questo caso è 
    // applicato con una maschera di dimensione 5x5 e con una deviazione 1.4. 
    // Sul risultato del blurring applichiamo l'algoritmo di Canny per poter 
    // trovare i contorni dell'immagine, facilitando il nostro algoritmo. 
    Mat cannyImage = inputImage.clone(); 
    GaussianBlur(inputImage, inputImage, Size(5, 5), 1.4, 1.4); 
    Canny(inputImage, cannyImage, 60, 160, 3); 

    // L'immagine risultante dall'applicazione di Canny viene mostrata in 
    // una finestra apposita. 
    namedWindow("Canny Edge Detector Image", WINDOW_AUTOSIZE); 
    imshow("Canny Edge Detector Image", cannyImage); 
    
    // Alla fine si applica l'algoritmo di Hough per poter trovare le 
    // rette all'interno dell'immagine derivata da Canny. 
    int threshold = atoi(argv[2]);
    Mat resultImage = HoughTransformation(cannyImage, clonedImage, threshold); 

    // Viene mostrato il risultato finale. 
    namedWindow("Hough Transformation Image", WINDOW_AUTOSIZE); 
    imshow("Hough Transformation Image", resultImage);  

    waitKey(0); 

    return 0; 
}