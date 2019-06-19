#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define L 256

using namespace std; 
using namespace cv;

// Definiamo i prototipi delle funzioni che saranno utilizzati nei 
// vari passaggi per l'equalizzazione di un istogramma. 
Mat            GetHistoImg(Mat src);
vector<double> GetProbabilities(Mat src); 
Mat            GetEqualization(vector<double> probs, Mat src);

// La funzione è necessaria per poter troncare i numeri alla seconda 
// cifra decimale. 
double round(double number) {
    return floor(number * 100 + 0.5) / 100; 
}

Mat GetHistoImg(Mat src) {
    vector<int> histogram(L-1);
    
    // Dobbiamo scansionare tutta l'immagine di partenza per
    // poter generare l'istogramma. Quindi si utilizzano due
    // cicli for annidati, uno che scorre sulle righe, l'altro
    // che invece scorre sulle colonne della matrice che
    // definisce l'immagine. Poi si incrementa la posizione
    // dell'array relativa al valore che si è incontrato.
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            histogram.at((int)src.at<uchar>(i, j))++;
        }
    }
    
    // destImgHist è l'immagine dell'istogramma che alla fine sarà
    // restituita dalla funzione in questione. Essa ha dimensione
    // pari a 256x256. Viene generata attraverso un ciclo for che
    // ovviamente dovrà tenere in considerazione l'array che è stato
    // creato poco fa.
    Mat destImgHist(L, L, CV_8UC1, Scalar(L-1));
    float max = 0;
    
    // Attraverso un ciclo for andiamo a selezionare il massimo
    // dell'istogramma, per poter creare l'immagine relativa ad esso.
    for(int i = 0; i < histogram.size(); i++) {
        if(max < histogram.at(i)) {
            max = histogram.at(i);
        }
    }
    
    // Per poter rappresentare le linee all'interno dell'immagine, scaliamo
    // l'intensità di un fattore scale, per poi disegnarle nell'immagine.
    float scale = (0.9*256)/max;

    // Il ciclo for viene utilizzato per poter disegnare l'istogramma
    // cosi come lo conosciamo dal punto di vista grafico. 
    for(int i = 0; i < histogram.size(); i++) {
        int intensity = (histogram.at(i)*scale);
        line(destImgHist, Point(i, 255), Point(i, 255-intensity), Scalar(0));
    }
    
    // Ritorniamo l'immagine dell'istogramma risultante.
    return destImgHist;
}

vector<double> GetProbabilities(Mat src) {
    // Dichiariamo i due vettori, uno per le occorrenze del valore presente
    // nell'immagine, e l'altro invece tiene conto di quanta probabilità c'è
    // che quel pixel venga osservato nell'immagine. 
    vector<int> occorrences(L-1);
    vector<double> probabilities(L-1);
    
    // Per poter calcolare le occorrenze sfruttiamo due cicli for annidati, 
    // uno che scorre sulle righe dell'immagine, e uno che scorre sulle 
    // colonne della stessa immagine. Quindi alla posizione definita dal 
    // valore del pixel nel vettore delle occorrenze si va ad incrementare 
    // il valore di una unità. 
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            occorrences.at((int)src.at<uchar>(i, j))++;
        }
    }

    // La probabilità viene calcolata con la formula (numero di occorrenze)/M*N, 
    // dove M sono le righe dell'immagine ed N sono le colonne. 
    for(int i = 0; i < occorrences.size(); i++) {
        probabilities.at(i) = round(static_cast<double>(occorrences.at(i))/(src.rows*src.cols)); 
    }
    
    // Alla fine si restituisce il vettore delle probabilità calcolato. 
    return probabilities;
}

Mat GetEqualization(vector<double> probs, Mat src) {
    // Il valore del pixel viene calcolato attraverso la 
    // sommatoria dei valori dei pixel precedenti. Tale 
    // valore viene poi inserito nell'apposita posizione
    // del vettore equalization. 
    double pixelValue = 0; 
    vector<double> equalization(L-1); 

    // Il ciclo for viene utilizzato per poter determinare il 
    // valore del pixel per ogni posizione del vettore equalization. 
    for(int i = 0; i < probs.size(); i++) {
        pixelValue += (L-1)*(probs.at(i)); 
        equalization.at(i) = pixelValue; 
    }

    // L'oggetto Mat di output è delle stesse dimensioni dell'immagine originale, e dello 
    // stesso tipo anche. Tutti i valori sono però pari a 0, all'inizio. 
    Mat destImage = Mat::zeros(src.rows, src.cols, src.type());

    // Il ciclo for serve per poter popolare l'immagine con i valori appositamente 
    // arrotondati nell'immagine finale. 
    for(int i = 0; i < destImage.rows; i++) {
        for(int j = 0; j < destImage.cols; j++) {
            destImage.at<uchar>(i, j) = floor(equalization.at((int)src.at<uchar>(i, j)));  
        }
    }

    // Alla fine si ritorna l'immagine finale equalizzata. 
    return destImage; 
}

int main(int argc, char *argv[]) {
    // In questo modo leggiamo da riga di comando nel momento del lancio
    // le due immagini di partenza, quella di input e quella di output.
    string in_file = argv[1];
    string out_file = argv[2];
    
    // Assegniamo a due stringhe i nomi delle immagini date in input
    // con il prefisso hist_, per una migliore comprensione dei risultati.
    string in_hist_file = "hist_" + in_file;
    string out_hist_file = "hist_" + out_file;
    
    // Leggiamo l'immagine che ci è stata data in input.
    Mat in_img = imread(in_file, IMREAD_GRAYSCALE);
    
    // Istanziamo un nuovo oggetto di tipo Mat, per l'immagine dell'istogramma.
    Mat in_hist_img;
    in_hist_img = GetHistoImg(in_img);
    
    // Alla fine scriviamo l'immagine in un nuovo file, che riposta il nome
    // definito dalla stringa in_hist_file.
    imwrite(in_hist_file, in_hist_img);
    
    // Il vettore dichiarato conterrà ciò che sarà restituito dalla funzione
    // GetProbabilities che prende in input l'immagine data da riga di comando. 
    vector<double> probs(L-1); 
    probs = GetProbabilities(in_img); 

    // L'immagine equalizzata sarà restituita dalla funzione GetEqualization che 
    // prende in input il vettore delle probabilità (calcolato in precedenza) e 
    // l'immagine data in input. 
    Mat equalizedImg; 
    equalizedImg = GetEqualization(probs, in_img); 

    // Possiamo confrontare il risultato con ciò che è restituito dalla funzione che 
    // si occupa di equalizzare l'immagine direttamente grazie alla libreria OpenCV. 
    Mat equalizedImageOpenCV = Mat::zeros(in_img.rows, in_img.cols, in_img.type());
    equalizeHist(in_img, equalizedImageOpenCV);

    // Visualizziamo l'immagine dell'equalizzazione in una finestra apposita. 
    namedWindow("OpenCV Equalization", WINDOW_AUTOSIZE);
    imshow("OpenCV Equalization", equalizedImageOpenCV);

    // Visualizziamo l'immagine dell'equalizzazione in una finestra apposita. 
    namedWindow("Custom Equalization", WINDOW_AUTOSIZE);
    imshow("Custom Equalization", equalizedImg);
    
    // Visualizziamo l'immagine dell'istogramma in una finestra apposita.
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", in_hist_img);

    // Visualizziamo l'immagine originale. 
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", in_img);
    
    waitKey(0);
    
    return 0;
}
