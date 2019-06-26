#include <iostream>
#include <cstdio>
#include <math.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv; 

Mat GaussianFilter(Mat& src, int size, float sigma); 
pair<Mat, Mat> SobelFilter(Mat& src);
Mat HarrisCornerDetector(Mat& src, Mat& Ix, Mat& Iy, int upper);

// Definiamo una struct per gestire una struttura dati per i punti che hanno come 
// informazioni le coordinate x, y e l'autovalore l, tale struttura sarà utilizzata 
// per la dichiarazione dei punti e del vettore di questi ultimi. 
struct PPoint {
    int x; 
    int y; 
    int l; 
}; 

Mat GaussianFilter(Mat& src, int size, float sigma) {
    // La prima matrice è la maschera Gaussiana da applicare all'immagine 
    // per mezzo di convoluzione. La seconda matrice è l'immagine risultante
    // dall'operazione di applicazione del filtro Gaussiano. 
    Mat kernel = Mat_<double>(size, size);
    Mat result = Mat(src.rows, src.cols, src.type());

    // Vengono calcolati il limite sinistro e quello di destro per poter 
    // calcolare il kernel Gaussiano. 
    int leftLimit = floor((kernel.rows/2)*(-1)); 
    int rightLimit = floor((kernel.cols/2)*(1));

    double summary = 0.0; 
    double sum = 0.0; 
    double pix = 0.0; 

    double multConstant = 1/exp(-(pow(leftLimit, 2) + pow(leftLimit, 2)) / (2*pow(sigma, 2)));

    // I due cicli for annidati permettono di calcolare la maschera da
    // applicare all'immagine e permette di calcolare la somma di tutti 
    // gli elementi che compongono tale maschera. Il risultato di questa 
    // operazione non è normalizzato. 
    for(int i = leftLimit; i <= rightLimit; i++) {
        for(int j = leftLimit; j <= rightLimit; j++) {
            pix = exp(-(pow(i, 2) + pow(j, 2)) / (2*pow(sigma, 2)));
            kernel.at<double>(i+rightLimit, j+rightLimit) = cvRound(pix * multConstant);
            sum += kernel.at<double>(i+rightLimit, j+rightLimit);
        }
    }

    // Per poter normalizzare il risultato dell'operazione allora ogni 
    // elemento che compone il kernel deve essere rapportato alla 
    // somma di tutti gli elementi della maschera. 
    for(int i = 0; i < kernel.rows; i++) {
        for(int j = 0; j < kernel.cols; j++) {
            kernel.at<double>(i, j) /= sum; 
        }
    } 

    // La convoluzione è molto pesante dal punto di vista computazionale, 
    // per cui di solito si sceglie di operare con degli array piuttosto 
    // che con una matrice (non è questo il caso). I primi due cicli for 
    // annidati scorrono sull'immagine di partenza fino ad un limite massimo 
    // definito dalla grandezza della maschera, questo per non sforare i 
    // limiti della matrice che definisce l'immagine di partenza. 
    for(int i = 0; i <= src.rows-kernel.rows; i++) {
        for(int j = 0; j <= src.cols-kernel.cols; j++) {
            
            // I cicli for annidati più interni lavorano sulla maschera (kernel) e 
            // scorrono dall'inizio fino alla fine della matrice del kernel. 
            // L'operazione svolta all'interno dei due cicli for è quella di ottenere 
            // la somma di tutti gli elementi della matrice dell'immagine inquadrati 
            // nella maschera kernel, moltiplicati con i corrispettivi elementi del 
            // kernel stesso. 
            for(int x = 0; x < kernel.rows; x++) {
                for(int y = 0; y < kernel.cols; y++) {
                    summary += static_cast<double>(src.at<uchar>(x+i, y+j)) * kernel.at<double>(x, y);  
                }
            }

            // Al termine dei due cicli for interni si assegna il valore della sommatoria
            // al pixel centrale della porzione di immagine inquadrata dalla maschera. Ed 
            // ovviamente si riporta la sommatoria a 0.  
            result.at<uchar>(i+1, j+1) = static_cast<uchar>((summary));
            summary = 0.0; 

        } 
    }

    // Alla fine si restituisce l'immagine risultante. 
    return result; 
}

pair<Mat, Mat> SobelFilter(Mat& src) {
    // Le due matrici riguardano le maschere di Sobel da applicare all'immagine 
    // per mezzo di convoluzione.
    Mat sobelX = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobelY = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

    // Le due matrici sono quelle che salveranno le derivate parziali calcolate 
    // con l'operatore Sobel di ogni pixel dell'immagine. 
    Mat Ix = Mat(src.rows, src.cols, src.type()); 
    Mat Iy = Mat(src.rows, src.cols, src.type());

    double sumX = 0.0; 
    double sumY = 0.0; 

    // Si scorre tutta l'immagine dall'inizio fino alla fine facendo attenzione 
    // a non sforare i bordi dell'immagine con la maschera che deve essere applicata 
    // sull'immagine stessa.  
    for(int i = 0; i < src.rows-sobelX.rows; i++) {
        for(int j = 0; j < src.cols-sobelY.cols; j++) {

            // I due cicli for più interni applicano il processo di convoluzione all'immagine, 
            // calcolando la derivata parziale rispetto ad x e rispetto ad y. Ricordiamo che 
            // questa viene calcolata in base alla sommatoria di tutti gli elementi che sono 
            // compresi all'interno della maschera moltiplicati con i corrispettivi elementi 
            // che compongono la maschera stessa. 
            for(int x = 0; x < sobelX.rows; x++) {
                for(int y = 0; y < sobelX.cols; y++) {
                    sumX += static_cast<double>(static_cast<int>(src.at<uchar>(x+i, y+j)) * sobelX.at<int>(x, y));
                    sumY += static_cast<double>(static_cast<int>(src.at<uchar>(x+i, y+j)) * sobelY.at<int>(x, y));
                }
            }

            sumX = abs(sumX); 
            sumY = abs(sumY); 

            // I risultati che otteniamo li inseriamo all'interno delle due matrici che 
            // conservano le derivate parziali rispetto ad x e rispetto ad y di ogni 
            // pixel che viene considerato. 
            Ix.at<uchar>(i, j) = static_cast<uchar>(sumX); 
            Iy.at<uchar>(i, j) = static_cast<uchar>(sumY); 

            sumX = 0; 
            sumY = 0; 

        }
    }

    // Quello che vogliamo restituire è una struttura che è composta dalle due matrici 
    // ottenute dalle derivate parziali rispetto ad x e rispetto ad y dei pixel.  
    pair<Mat, Mat> derivates = make_pair(Ix, Iy);
    return derivates; 
}

Mat HarrisCornerDetector(Mat& src, Mat& Ix, Mat& Iy, int upper) {
    // Abbiamo calcolato l'immagine gradiente di ogni pixel attraverso l'operatore di Sobel, 
    // e sono state salvate le derivate parziali rispetto ad x e rispetto ad y in due matrici 
    // che sono chiamate Ix e Iy, che di conseguenza sono state date in input alla funzione. 
    // A questo punto, per ogni pixel: 
    // 1. Calcolare la matrice M in uno specifico intorno del pixel; 
    // 2. Di questa matrice dobbiamo calcolare il più piccolo autovalore; 
    // La matrice M è una matrice composta dalle derivate parziali che sono state calcolate 
    // con Sobel (in questo caso) di quello specifico pixel. Tale matrice è moltiplicata ad 
    // una maschera w(x, y) che è una finestra che assegna peso unitario al pixel che si trova 
    // al suo interno. la dimensione di tale finestra è pari a 3(x3) ed è data dalla variabile wSize.  
    int wSize = 3; 

    // Il vettore L che andiamo a creare è del tipo EidenPoint, una classe che conserva specifiche 
    // informazioni, quali le coordinate x, y e l'autovalore di quel pixel in quella posizione. 
    vector<PPoint> L;

    for(int i = 0; i < src.rows-wSize; i++) {
        for(int j = 0; j < src.cols-wSize; j++) {

            // La matrice M è di dimensioni 2x2, e conterrà quelle che sono le derivate parziali 
            // rispetto ad x e rispetto ad y (ed il loro prodotto) per ogni pixel. La seconda 
            // matrice invece è di dimensione pari a 1 colonna e 2 righe e conserva gli autovalori 
            // calcolati della matrice M. 
            Mat M = Mat::zeros(2, 2, CV_32FC1);
            Mat eigenVal = Mat::zeros(1, 2, CV_32FC1);
            
            // I cicli for più interni servono per poter calcolare gli elementi della matrice M, e
            // quindi scorrono sulle dimensioni della matrice M stessa. Utilizziamo la sommatoria 
            // per poter calcolare le somme dei prodotti delle derivate rispetto ad x e rispetto ad 
            // y dei pixel che sono inquadrati dalla finestra che assegna ad essi peso unitario. 
            for(int x = 0; x < wSize; x++) {
                for(int y = 0; y < wSize; y++) {
                    M.at<float>(0, 0) += static_cast<float>(pow(Ix.at<uchar>(i+x, j+y), 2)); 
                    M.at<float>(0, 1) += static_cast<float>(Ix.at<uchar>(i+x, j+y) * Iy.at<uchar>(i+x, j+y));
                    M.at<float>(1, 0) += static_cast<float>(Ix.at<uchar>(i+x, j+y) * Iy.at<uchar>(i+x, j+y)); 
                    M.at<float>(1, 1) += static_cast<float>(pow(Iy.at<uchar>(i+x, j+y), 2));
                }
            }

            // La funzione va direttamente a calcolare gli autovalori della matrice M, e li va 
            // ad inserire all'interno della matrice eigenVal, che è un vettore. 
            eigen(M, eigenVal); 

            // Dobbiamo estrapolare da questo il più piccolo autovalore, che andiamo a salvare
            // all'interno della variabile minEigen (sfruttiamo la funzione min). 
            float minEigen = min(eigenVal.at<float>(0, 0), eigenVal.at<float>(0, 1));
            
            // Se questo autovalore è più grande di un certo valore che è il valore di threshold, 
            // e se lo è allora il pixel viene inserito con le sue coordinate ed il suo autovalore 
            // all'interno del vettore L. 
            if(minEigen > upper) {
                // Definiamo una variabile di tipo PPoint per poter salvare le informazioni del punto. 
                PPoint point;

                // Inseriamo le informazioni all'interno della struttura dati per il punto. 
                point.x = i; 
                point.y = j; 
                point.l = minEigen; 

                // Poi inseriamo il punto all'interno del vettore dei punti. 
                L.push_back(point); 
            }
        }
    }

    // Si deve ordinare il vettore in modo decrescente in base all'autovalore, in modo che si possa 
    // scandagliare la lista dall'inizio alla fine. 
    sort(L.begin(), L.end(), [](const PPoint& point1, const PPoint& point2) { return point1.l > point2.l; }); 

    // Cloniamo l'immagine iniziale per poter restituire l'immagine di destinazione con i 
    // contorni evidenziati con dei cerchi. 
    Mat dest = src.clone();
	
    // Ci occorre per convertire l'immagine di destinazione a colori. 
    cvtColor(dest, dest, COLOR_GRAY2BGR);

    // Andiamo a sfoltire i punti che si trovano in un intorno ben preciso, 
    // quello cioè definito dalla grandezza della finestra (wSize) utilizzata 
    // per il calcolo degli elementi della matrice M. 
	for(int i = 0; i < L.size(); i++) {
        for(int j = i+1; j < L.size(); j++) {
            if(L.at(j).x <= L.at(i).x+wSize && L.at(j).x >= L.at(i).x-wSize && L.at(j).y <= L.at(i).y+wSize && L.at(j).y >= L.at(i).y-wSize) {
                // Impostiamo a zero tutti i parametri del punto nel caso in cui si trovi all'interno 
                // del range che abbiamo definito di dimensione pari a wSize. 
                L.at(j).x = 0; 
                L.at(j).y = 0; 
                L.at(j).l = 0; 
            }
        }
	}

    // Andiamo a selezionare i punti dalla lista per poter cerchiare gli 
    // angoli dell'immagine, se i parametri del punto sono diversi da 0, 
    // allora quello è un punto da disegnare. 
    for(int i = 0; i < L.size(); i++) {
        if(L.at(i).x != 0 && L.at(i).y != 0 && L.at(i).l != 0) {
            circle(dest, Point(L.at(i).y, L.at(i).x), 5, Scalar(0, 0, 255), 1, 4, 0);
        }
    }

    // Alla fine restituiamo l'immagine finale. 
    return dest; 
}

int main(int argc, char *argv[]) {
    // Leggiamo l'immagine da riga di comando. 
    string inputFile = argv[1];
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE); 

    // Visualizziamo l'immagine originale. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    // Applichiamo il filtro di Sobel per poter calcolare le derivate parziali. 
    pair<Mat, Mat> sobelFilter = SobelFilter(inputImage);
    Mat Ix = GaussianFilter(sobelFilter.first, atoi(argv[2]), atoi(argv[3])); 
    Mat Iy = GaussianFilter(sobelFilter.second, atoi(argv[2]), atoi(argv[3]));  

    // Applichiamo l'algoritmo di Harris per trovare gli spigoli dell'immagine.  
    Mat harrisCornerDetector = HarrisCornerDetector(inputImage, Ix, Iy, atoi(argv[4])); 
    namedWindow("Harris Image", WINDOW_AUTOSIZE); 
    imshow("Harris Image", harrisCornerDetector); 

    waitKey(0); 

    return 0; 
}
