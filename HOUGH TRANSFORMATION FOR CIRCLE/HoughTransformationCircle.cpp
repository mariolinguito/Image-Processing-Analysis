#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv; 

Mat HoughTransformation(Mat src, Mat rst, int thr, int rmax, int rmin); 

// Gli input della trasformata di Hough per i cerchi sono l'immagine originale
// dalla quale estrapolare le informazioni, l'immagine da dare come risultato 
// delle operazioni, il threshold con il quale confrontare i valori della matrice 
// dei voti, e due parametri, uno è il raggio massimo ed uno è il raggio minimo, 
// questo per evitare che vengano fatte troppe operazioni che dal punto di vista 
// computazionale è molto oneroso. Si lavorerà quindi all'interno di questo range 
// di valori. 
Mat HoughTransformation(Mat src, Mat rst, int thr, int rmax, int rmin) {
    // Il range nella quale lavorare quindi è definito dal raggio massimo 
    // e da quello minimo, tale sarà una delle dimensioni attribuite alla 
    // matrice tridimensionale dei voti.
    int R = (rmax-rmin)+1;  

    // La matrice dei voti è tridimensionale perchè se andiamo a considerare 
    // l'equazione classica di una circonferenza essa è: (x-xc)^2+(y-yc)^2=r, 
    // dove xc e yc sono le coordinate del centro della circonferenza di raggio 
    // pari a r. Le coordinate sono tre nello spazio dei parametri. Le dimensioni 
    // quindi sono il range dei due raggi, e la grandezza dell'immagine originale. 
    int matrixSize[3] = {R, src.rows, src.cols}; 

    // Uno dei costruttori della classe Mat prevede anche che venga specificata
    // la dimensione della matrice, che in questo caso è tridimensionale, ed i 
    // size da dare alla matrice stessa, oltre che il tipo delle informazioni in
    // essa contenute. 
    Mat VotingMatrix = Mat(3, matrixSize, CV_64F);

    // Si deve scorrere l'immagine per intera per trovare i punti di edge che possono
    // o meno appartenere ad una circonferenza. Quindi il compito dei due cicli for 
    // annidati è quello di scorrere l'immagine pixel per pixel. A questo punto ci 
    // troviamo di fronte ad un edge solo quando il valore del pixel è maggiore di 
    // una certa soglia per le proprietà dell'Edge Detector Canny. A questo punto, 
    // essendo che non conosciamo il raggio della circonferenza che intendiamo trovare, 
    // dobbiamo provare ogni raggio compreso all'internod del range, ed è questo 
    // il compito del terzo ciclo for annidato. Sappiamo poi che i punti di una circonferenza
    // all'interno dello spazio immagine corrispondono ad una circonferenza nello spazio dei 
    // parametri che si interseca con altre circonferenze generati da altri punti che appartengono
    // alla stessa circonferenza nello spazio delle immagini, quindi non ci resta che 
    // generare tutte le possibili circonferenze per quel raggio specifico ed incrementare 
    // la cella relativa all'interno della matrice dei voti. 
    // Quando scorriamo la matrice dei voti, quella cella con il massimo numero di 
    // intersezione corrisponde al centro di una circonferenza che si trova 
    // all'interno della immagine iniziale. Ovviamente con quel relativo raggio, dato 
    // che le circonferenze all'interno dello spazio dei parametri sono descritte da
    // due parametri (coordinate centro, raggio). 
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(src.at<uchar>(i, j) > 250) {
                
                // Dal punto di vista computazionale è molto oneroso eseguire le operazioni 
                // previste, quindi le eseguiamo in un range di raggi che sono stati definiti 
                // in precedenza. 
                for(int r = rmin; r < rmax; r++) {

                    // Per ognuno dei raggi andiamo a definire le circonferenze grado per grado 
                    // fino a 360 gradi, dato che è il massimo angolo della circonferenza stessa. 
                    for(int d = 0; d < 360; d++) {
                        
                        // Dal punto di vista parametrico sappiamo che le coordinate del centro 
                        // della circonferenza di raggio d attuale sono: 
                        // x = i (coordinata x del punto sulla circonferenza) - Raggio*cos(angolo in radianti); 
                        // y = j (coordinata y del punto sulla circonferenza) - Raggio*sin(angolo in radianti); 
                        int xC = round(i - r*cos(d*(CV_PI/180)));
                        int yC = round(j - r*sin(d*(CV_PI/180))); 

                        // Per sicurezza, prima di incrementare la rrlativa posizione all'interno della matrice 
                        // dei voti, verifichiamo se questo rientra all'interno dei limiti della matrice stessa. 
                        // Se rientra allora andiamo ad incrementare il relativo elemento all'interno della 
                        // matrice tridimensionale dei punti. 
                        if(xC >= 0 && xC < src.cols && yC >= 0 && yC < src.rows) {
                            VotingMatrix.at<float>(r-rmin, xC, yC)++; 
                        } 
                    }
                }
            }
        }
    }

    // Avendo ottenuto la matrice dei voti, a questo punto la si deve scorrere tutta 
    // per poter andare a trovare quei valori che superano o sono uguali al valore 
    // soglia che abbiamo dato in input. Se ci sono valori che rispettano questa 
    // proprietà allora dobbiamo disegnare una circonferenza sull'immagine da date 
    // come risultato, e lo si fa prendendo in esame il centro della circonferenza 
    // ossia l'elemento z e l'elemento j, ed il raggio, ossia l'elemento i. 
    for(int i = 0; i < matrixSize[0]; i++) {
        for(int j = 0; j < matrixSize[1]; j++) {
            for(int z = 0; z < matrixSize[2]; z++) {
                if(VotingMatrix.at<float>(i, j, z) >= thr) {
                    Point center; 
                    center.x = z; 
                    center.y = j; 
                    int radius = i+rmin; 

                    // Si utilizza un'apposita funzione per disegnare il cerchio. 
                    circle(rst, center, radius, Scalar(125, 255, 10), 1, 8, 0);
                }
            }
        }
    }

    // Alla fine si restituisce il risultato. 
    return rst; 
}

int main(int argc, char *argv[]) {
    // Si prende in input il nome dell'immagine da riga di comando, 
    // e la si legge con i colori. In seguito la cloniamo per darla 
    // in input alla funzione. 
    String inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_ANYCOLOR); 
    Mat clonedImage = inputImage.clone(); 

    // Si mostra l'immagine all'interno di un'apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inputImage); 

    // Prima di dare in pasto l'immagine all'algoritmo di Hough per 
    // la ricerca di cerchi, applichiamo uno smoothing per migliorare 
    // la ricerca delle circonferenze. In questo caso applichiamo il 
    // filtro Gaussiano con una maschera di dimensioni 5x5 e una 
    // deviazione standard pari a 1.5. Poi applichiamo su di essa 
    // l'algoritmo di Canny per trovare i bordi degli oggetti all'interno 
    // dell'immagine in modo tale che risulti essere più semplice trovare 
    // le circonferenze all'interno dell'immagine dato che ci soffermiamo 
    // solo su punti salienti che hanno un particolare valore. 
    Mat cannyImage = inputImage.clone(); 
    GaussianBlur(inputImage, inputImage, Size(5, 5), 1.5, 1.5); 
    Canny(inputImage, cannyImage, 80, 150, 3); 

    // Mostriamo il risultato dell'algoritmo di Canny in un'apposita finestra. 
    namedWindow("Canny Image", WINDOW_AUTOSIZE);
    imshow("Canny Image", cannyImage);

    // Prendiamo in input il threshold e i due raggi, uno massimo e uno minimo.
    int threshold = atoi(argv[2]); 
    int rmax = atoi(argv[3]); 
    int rmin = atoi(argv[4]); 

    // Applichiamo sull'immagine risultante da Canny l'algoritmo di Hough per la 
    // ricerca di cerchi, dando in input anche l'immagine originale clonata 
    // ed il valore di threshold e quello dei due raggi. 
    Mat resultImage = HoughTransformation(cannyImage, clonedImage, threshold, rmax, rmin); 

    // Alla fine si mostra il risultato dell'algoritmo in un'apposita immagine. 
    namedWindow("Hough Image", WINDOW_AUTOSIZE);
    imshow("Hough Image", resultImage);

    waitKey(0); 

    return 0; 
}