#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std; 
using namespace cv; 

Mat GaussianFilter(Mat& src, int size, int sigma);
Mat SobelApplication(Mat& src, int kernelSize);
Mat Thresholding(Mat& src, int maxG);

// Primo passo per l'algoritmo di Canny: Soppressione del rumore dell'immagine 
// con un filtro, in questo caso usiamo il filtro Gaussiano. 
Mat GaussianFilter(Mat& src, int size, int sigma) {
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

    double multConstant = 1/exp(-(pow(leftLimit, 2) + pow(leftLimit, 2)) / (2*pow(sigma,2)));

    // I due cicli for annidati permettono di calcolare la maschera da
    // applicare all'immagine e permette di calcolare la somma di tutti 
    // gli elementi che compongono tale maschera. Il risultato di questa 
    // operazione non è normalizzato. 
    for(int i = leftLimit; i <= rightLimit; i++) {
        for(int j = leftLimit; j <= rightLimit; j++) {
            pix = exp(-(pow(i, 2) + pow(j, 2)) / (2*pow(sigma,2)));
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

Mat SobelApplication(Mat& src, int kernelSize) {
    // Volendo applicare il filtro di Sobel, allora bisogna creare prima di 
    // tutto le due matrici per poter calcolare il gradiente in ognuno dei 
    // pixel dell'immagine. 
    Mat sobelX = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); 
    Mat sobelY = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

    // Dichiariamo alcune variabili necessarie al corretto funzionamento
    // dell'algoritmo. 
    double modG = 0.0; 
    double arcG = 0.0;  
    double sumX = 0.0; 
    double sumY = 0.0; 
    double dirG = 0.0; 
    double maxG = 0.0; 

    // La prima matrice riguarda tutte le direzioni dei gradienti calcolati, 
    // mentre la seconda matrice riguarda la magnitudine di ogni pixel. Ognuna 
    // delle due matrici ha le stesse dimensioni dell'immagine originale, ed 
    // è dello stesso tipo. 
    Mat gradDirections = Mat(src.rows, src.cols, src.type());
    Mat gradModules = Mat(src.rows, src.cols, src.type());

    // Applicare Sobel all'immagine significa applicare la maschera, sia X che 
    // Y per convoluzione. Cioè, si moltiplicano gli elementi della matrice 
    // inquadrati dalla maschera con gli elementi della maschera stessi, e poi 
    // si sommano per ottenere un unico numero. Dobbiamo stare attenti ai bordi
    // dell'immagine, ed infatti la maschera non deve andare oltre le dimensioni 
    // dell'immagine stessa. 
    for(int i = 0; i < src.rows-3; i++) {
        for(int j = 0; j < src.cols-3; j++) {
            
            for(int x = 0; x < sobelX.rows; x++) {
                for(int y = 0; y < sobelX.rows; y++) {
                    sumX += static_cast<double>(static_cast<int>(src.at<uchar>(x+i, y+j)) * sobelX.at<int>(x, y));
                    sumY += static_cast<double>(static_cast<int>(src.at<uchar>(x+i, y+j)) * sobelY.at<int>(x, y));
                }
            }

            // L'intensità del pixel viene calcolata come la somma dei 
            // valori assoluti ottenuti dai cicli for più interni. Mentre
            // l'angolo della direzione viene calcolato attraverso
            // l'arcotangente del rapporto tra le due somme.  
			sumX = abs(sumX);
			sumY = abs(sumY);
			modG = sumX + sumY;
			arcG = atan2(sumY, sumX);
			dirG = arcG*180/M_PI;

            // Ci occorre conoscere l'intensità massima dei pixel per le successive 
            // elaborazioni, ed in particolare per la fase finale. 
			if(modG > maxG) {
				maxG = modG;
			}
			
            // Dobbiamo stabilizzare il valore del pixel nel caso in cui esso 
            // eccede il valore 255, oppure contestualmente è minore di 0. 
			modG = modG > 255? 255: modG < 0? 0: modG;

            // Dobbiamo stabilizzare, e quindi normalizzare anche la direzione 
            // del gradiente calcolata, nel caso in cui essa superi i 180 gradi o
            // sia minore di 0. 
            if(dirG > 180) {
				while(dirG > 180) dirG -= 180;
			} else if(dirG < 0) {
				while(dirG < 0) dirG += 180;
			}

            // Dobbiamo verificare che la direzione del gradiente rientri in uno di 
            // questi quattro casi, perchè potrebbe appartenere ad un bordo che va
            // in orizzontale, uno che va in verticale, uno che va in obliquo positivo
            // e uno che va in obliquo negativo. Questo è un processo approssimativo.  
            if((dirG >= 0 && dirG < 22.5) || (dirG >= 157.5 && dirG <= 180)) {
                // Caso studiato::Orizzontale.  
                dirG = 0; 
            } else if(dirG >= 22.5 && dirG < 67.5) {
                // Caso studiato::Obliguo. 
                // Da destra verso sinistra. 
                dirG = 45; 
            } else if(dirG >= 67.5 && dirG < 112.5) {
                // Caso studiato::Verticale.
                dirG = 90; 
            } else if(dirG >= 112.5 && dirG < 157.5) {
                // Caso studiato::Obliquo. 
                // Da sinistra verso destra. 
                dirG = 135; 
            } 

            // Gli elementi calcolati vengono poi inseriti all'interno delle matrici
            // che sono state create appositamente per contenerli. 
            gradDirections.at<uchar>(i, j) = dirG;
            gradModules.at<uchar>(i, j) = modG; 

            // Si riportano i valori delle variabili a 0 per la prossima iterazione. 
            sumX = 0.0; 
            sumY = 0.0; 
            modG = 0.0; 
            arcG = 0.0; 
            dirG = 0.0; 

        }
    }

    // Un pixel fa parte di un contorno se esso ha intensità massima rispetto ai suoi vicini 
    // più prossimi, quindi i pixel che si trovano a nord, a sud, a est, a ovest, oppure a 
    // nord-ovest, nord-est, sud-est, sud-ovest. Questi corrispondono ai quattro casi possibili. 
    // Per evitare di sforare la matrice dell'immagine si deve creare una nuova immagine con uno 
    // spessore (pad) in modo tale che i pixel sui bordi abbiano i vicini più prossimi pari a 0. 
    // Le dimensioni delle due nuove matrici quindi saranno pari a quelle delle matrici precedenti 
    // con l'aggiunta del pad. 
    int pad = floor(kernelSize/2);
    Mat paddedModules = Mat(gradModules.rows+pad, gradModules.cols+pad, gradModules.type());
    Mat paddedDirections = Mat(gradDirections.rows+pad, gradDirections.cols+pad, gradDirections.type());

    // Gli elementi delle due matrici vengono copiati all'interno delle due nuove matrici, facendo 
    // attenzione ad inserirli in modo tale che vi sia un bordo di elementi pari a 0 su ogni lato 
    // delle matrici, per poter effettuare i controlli successivi. 
    for(int i = 0; i < gradModules.rows; i++) {
        for(int j = 0; j < gradModules.cols; j++) {
            paddedModules.at<uchar>(i+1, j+1) = gradModules.at<uchar>(i, j);
            paddedDirections.at<uchar>(i+1, j+1) = gradDirections.at<uchar>(i, j); 
        }
    }

    // I due cicli for procedono sulla matrice con il pad. Ad ogni iterazione del ciclo più interno 
    // si estrapolano i pixel più prossimi al pixel corrente, oltre che la sua direzione. 
    for(int i = (pad-1); i < paddedModules.rows-(pad-1); i++) {
        for(int j = (pad-1); j < paddedModules.cols-(pad-1); j++) {
            int nord = paddedModules.at<uchar>(i-1, j);
            int sud = paddedModules.at<uchar>(i+1, j); 
            int est = paddedModules.at<uchar>(i, j+1);
            int ovest = paddedModules.at<uchar>(i, j-1);
            int nest = paddedModules.at<uchar>(i-1, j+1);
            int sest = paddedModules.at<uchar>(i+1, j+1);
            int sovest = paddedModules.at<uchar>(i+1, j-1);
            int novest = paddedModules.at<uchar>(i-1, j-1); 

            int currentDirection = paddedDirections.at<uchar>(i, j);
            int currentModules = paddedModules.at<uchar>(i, j);

            // Se il valore del pixel che teniamo in considerazione 
            // è minore del valore dei pixel vicini, allora non deve 
            // essere considerato come bordo dell'immagine. Significa 
            // che l'intensità del pixel non è quella massima, quindi 
            // deve essere soppressa. 
            if(currentDirection == 0) {
                // Caso studiato::Orizzontale. 
                if(currentModules < est || currentModules < ovest) {
                    paddedModules.at<uchar>(i, j) = 0; 
                }
            } else if(currentDirection == 90) {
                // Caso studiato::Verticale. 
                if(currentModules < nord || currentModules < sud) {
                    paddedModules.at<uchar>(i, j) = 0; 
                }
            } else if(currentDirection == 45) {
                // Caso studiato::Obliquo. 
                if(currentModules < novest || currentModules < sest) {
                    paddedModules.at<uchar>(i, j) = 0; 
                }
            } else if(currentDirection == 135) {
                // Caso studiato::Obliquo. 
                if(currentModules < nest || currentModules < sovest) {
                    paddedModules.at<uchar>(i, j) = 0; 
                }
            }
        }
    }

    // La fase finale dell'algoritmo consiste nel fare il threshold 
    // sull'immagine ottenuta dall'operazione di Sobel. La funzione 
    // quindi prende in input la matrice delle intensità dei pixel 
    // dell'immagine e l'intensità massima. 
    Mat thresholding = Thresholding(paddedModules, maxG); 

    // Si restituisce il risultato finale delle operazioni. 
    return thresholding; 
}

Mat Thresholding(Mat& src, int maxG) {
    Mat resultImage = Mat(src.rows, src.cols, src.type()); 

    // Ci occorrono le due soglie per effettuare l'ultima operazione 
    // dell'algoritmo, vengono calcolate entrambe considerando l'intensità
    // massima dei pixel dell'immagine. 
    int tLow = cvRound(maxG*0.05);
    int tHigh = cvRound(maxG*0.1);

    // I cicli for annidati sono scorrono sull'immagine data in input 
    // per poter verificare pixel per pixel se rispettano i vincoli 
    // imposti, ossia che l'intensità del pixel corrente sia maggiore
    // della soglia tHigh, se non supera questa soglia, allora deve 
    // essere almeno superiore a tLow e sarà considerato solo se supera 
    // la soglia tLow e sia collegato ad un pixel del primo tipo. 
    for(int i = 0; i < resultImage.rows; i++) {
        for(int j = 0; j < resultImage.cols; j++) {
            if(src.at<uchar>(i, j) > tHigh) {
                resultImage.at<uchar>(i, j) = 255; 
            } else if(src.at<uchar>(i, j) < tLow) {
                resultImage.at<uchar>(i, j) = 0; 
            } else if(src.at<uchar>(i, j) <= tHigh && src.at<uchar>(i, j) >= tLow) {
                // Il flag booleano ci dirà se il pixel deve essere considerato come
                // bordo oppure no. 
                bool canCheck = false; 

                /*
                for(int x = -1; x <= 1; x++) {
                    for(int y = -1; y <= 1; y++) {
                        if(src.at<uchar>(i+x, j+y) > tHigh) {
                            canCheck = true; 
                            break; 
                        }
                    }
                }
                */

                // Si devono prendere in considerazione i pixel che circondano il pixel 
                // che stiamo tenendo in considerazione. 
                int nord = src.at<uchar>(i-1, j); 
                int sud = src.at<uchar>(i+1, j); 
                int est = src.at<uchar>(i, j+1); 
                int ovest = src.at<uchar>(i, j-1); 
                int nest = src.at<uchar>(i-1, j+1); 
                int sest = src.at<uchar>(i+1, j+1); 
                int sovest = src.at<uchar>(i+1, j-1); 
                int novest = src.at<uchar>(i-1, j-1); 

                // Se uno di questi è un pixel che rispetta la prima condizione allora può essere 
                // considerato come un bordo dell'immagine. 
                if(nord > tHigh || sud > tHigh || est > tHigh || ovest > tHigh || nest > tHigh || sest > tHigh || sovest > tHigh || novest > tHigh) {
                    canCheck = true; 
                }

                // Se il flag è true, allora viene considerato. Altrimenti 
                // semplicemente viene ignorato. 
                if(canCheck) {
                    resultImage.at<uchar>(i, j) = 255;
                } else { 
                    resultImage.at<uchar>(i, j) = 0; 
                }
            } 
        }
    }

    // Alla fine si restituisce il risultato finale. 
    return resultImage; 
}

int main(int argc, char *argv[]) {
    // Si prende in input l'immagine iniziale sulla quale applicare l'algoritmo 
    // di Canny Edge Detector. 
    string inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);

    // Utilizzo della funzione di soppressione del rumore con filtro 
    // Gaussiano scritta da capo. 
    Mat gaussianFilter = GaussianFilter(inputImage, 3, 3); 

    // Per un confronto, utilizzo della funzione di soppressione del rumore 
    // con filtro Gaussiano di OpenCv. 
    Mat workGBlur = Mat(inputImage);
    workGBlur = inputImage.clone();
    cv::GaussianBlur(inputImage, workGBlur, cv::Size(3, 3), 3);

    // Utilizzo della funzione di applicazione del filtro di Sobel, 
    // oltre che l'applicazione del threshold sul risultato finale. 
    Mat sobelApplication = SobelApplication(gaussianFilter, 3);

    // Si mostrano i risultati di ogni passo in apposite finestre. 
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inputImage);

    namedWindow("Custom Gauss", WINDOW_AUTOSIZE);
    imshow("Custom Gauss", gaussianFilter);

    namedWindow("OpenCV Gauss", WINDOW_AUTOSIZE);
    imshow("OpenCV Gauss", workGBlur);

    namedWindow("Custom Sobel Application", WINDOW_AUTOSIZE);
    imshow("Custom Sobel Application", sobelApplication);

    waitKey(0); 

    return 0; 
}
