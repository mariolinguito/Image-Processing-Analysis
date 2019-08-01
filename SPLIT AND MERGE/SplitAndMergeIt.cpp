#include <iostream>
#include <cstdio>
#include <stack>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std; 

// Definiamo la struttura dati per le informazioni relative alle 
// regioni dell'immagine. In particolare abbiamo: 
// regionRec: rettangolo che definisce la regione sull'immagine 
//            originale; 
// regionSrc: porzione dell'immagine che andiamo a considerare; 
// regionCl:  colore della regione, che sarà utilizzato per colorare 
//            appunto il rettangolo che stiamo considerando; 
struct Region {
    Rect regionRec; 
    Mat regionSrc; 
    Scalar regionCl;
}; 

stack<Region> Split(Mat src, Rect rec, int thr); 
bool CheckHomogeneity(Mat src, int thr); 
bool MergeRegion(Mat src, Region R1, Region R2, int thr); 
void Merge(Mat src, stack<Region> regionList, int thr); 

stack<Region> Split(Mat src, Rect rec, int thr) {
    // Abbiamo bisogno di due stack, uno per le regioni che devono essere 
    // considerate per essere divise in 4 o meno, ed uno per quelle regioni 
    // che rispettano il criterio di omogeneità previsto. 
    stack<Region> processList; 
    stack<Region> regionList; 

    // La regione che per prima viene inserita all'interno dello stack è 
    // quella che riguarda l'intera immagine. Quindi vengono aggiornate 
    // le informazioni dei campi della struttura dati e poi si inzia con 
    // il ciclo while. 
    Region currentRegion; 
    currentRegion.regionRec = rec;
    currentRegion.regionSrc = src; 

    processList.push(currentRegion); 

    // Il ciclo while continua fino a che ci sono regioni nello stack, 
    // quindi per ogni regione viene applicato il criterio di omogeneità,
    // e se sono omogenee vengono inserite all'interno di un altro stack. 
    while(!processList.empty()) {
        // La regione corrente viene estratta come primo elemento dello 
        // stack, poi viene eliminato dallo stack stesso. 
        currentRegion = processList.top(); 
        processList.pop(); 

        // Se sulla regione il criterio di omogeneità non è verificato, allora
        // la regione viene divisa in 4 sottoregioni di ugual dimensioni. 
        if(!(CheckHomogeneity(currentRegion.regionSrc, thr))) {
            // Si calcola l'altezza e la larghezza delle regioni, saranno 
            // utilizzate per creare le sottoregioni. Ovviamente sia l'altezza
            // che la larghezza è la metà dell'attuale regione. 
            int h = floor(currentRegion.regionSrc.rows/2); 
            int w = floor(currentRegion.regionSrc.cols/2); 

            // Per ogni regione viene creata una variabile di tipo Region, in 
            // modo tale che ognuna conserva le informazioni della specifica 
            // regione. La prima regione ad esempio va dalla coordinata x e 
            // dalla coordinata y del rettangolo della regione corrente, ed 
            // ha le dimensioni che sono state in precedenza calcolate. 
            // La regione relativa all'immagine invece va dal punto (0, 0) ed 
            // ha le stesse dimensioni specificate in precedenza. 
            // Una volta calcolate queste informazioni, la relativa regione viene 
            // inserita all'interno dello stack processList per essere elaborata 
            // successivamente, e lo stesso viene fatto per le altre tre regioni. 
            Region Region_1; 
            Region_1.regionRec = Rect(currentRegion.regionRec.x, currentRegion.regionRec.y, w, h);
            Region_1.regionSrc = currentRegion.regionSrc(Rect(0, 0, w, h));
            processList.push(Region_1);   

            Region Region_2; 
            Region_2.regionRec = Rect(currentRegion.regionRec.x+w, currentRegion.regionRec.y, w, h);
            Region_2.regionSrc = currentRegion.regionSrc(Rect(w, 0, w, h));
            processList.push(Region_2); 

            Region Region_3; 
            Region_3.regionRec = Rect(currentRegion.regionRec.x, currentRegion.regionRec.y+h, w, h);
            Region_3.regionSrc = currentRegion.regionSrc(Rect(0, h, w, h));
            processList.push(Region_3);  

            Region Region_4; 
            Region_4.regionRec = Rect(currentRegion.regionRec.x+w, currentRegion.regionRec.y+h, w, h);
            Region_4.regionSrc = currentRegion.regionSrc(Rect(w, h, w, h));
            processList.push(Region_4);          
        } else {
            // Se invece siamo arrivati al punto in cui la regione non può essere
            // suddivisa o perchè rispetta il criterio di omogeneità o perchè è 
            // troppo piccola per essere divisa, allora si calcola il colore della 
            // stessa attraverso la funzione per il calcolo della deviazione standard 
            // e della media all'interno di un set di dati, e si attribuisce al colore 
            // la media. In seguito la regione viene inserita all'interno dello 
            // stack regionList, per poter essere elaborata nella fase di Merg. 
            Scalar mean, std; 
            meanStdDev(currentRegion.regionSrc, mean, std); 
            currentRegion.regionCl = mean; 
            regionList.push(currentRegion);  
        }
    }

    // Alla fine si restituisce lo stack delle regioni per il Merge. 
    return regionList; 
}

// La funzione per verificare se una regione è omogenea o meno fa 
// uso della funzione di OpenCV per il calcolo della deviazione 
// standard e della media, quindi si verifica che la deviazione 
// standard sia minore di una certa quantità numerica oppure che le 
// dimensioni della regione stessa non siano minore di un valore 
// che è stato dato in input. Se è così allora viene restituito un 
// true, altrimenti viene restituito un false. 
bool CheckHomogeneity(Mat src, int thr) {
    bool response = false; 
    Scalar mean, std; 
    meanStdDev(src, mean, std); 

    if(std[0] <= 5.8 || (src.rows*src.cols) <= thr) {
        response = true; 
    }

    // La risposta viene restituita alla fine. 
    return response; 
}

// La funzione prende in input l'immagine iniziale, le due regioni che 
// devono essere fuse e per il quale si devono ricalcolare le informazioni 
// della regione nel caso in cui essa sia stata unita, e poi il valore di 
// threshold con la quale verificare le dimensioni della regione. 
bool MergeRegion(Mat src, Region R1, Region R2, int thr) {
    bool response = false; 

    // Si definiscono i due rettangoli da fondere relativi alle 
    // due regioni che sono state date in input. In seguito si 
    // definisce un nuovo rettangolo che è l'unione dei due 
    // precedenti. Su qusto rettangolo si verifica il criterio di 
    // omogeneità e se è vero, allora il rettangolo che è stato 
    // appena creato viene inserito nel relativo campo della 
    // prima regione data in input, e viene disegnato un rettangolo 
    // riempito con il colore della prima regione (è indifferente). 
    // Ovviamente il responso in questo caso è vero. Altrimenti 
    // sarebbe falso. 
    Rect Region1 = R1.regionRec; 
    Rect Region2 = R2.regionRec; 
    Rect Region12 = Region1|Region2;

    if(CheckHomogeneity(src(Region12), thr)) {
        R1.regionRec = Region12; 
        response = true; 
        rectangle(src, R1.regionRec, R1.regionCl, FILLED);
    }

    // Il risultato viene dato in input. 
    return response; 
}

void Merge(Mat src, stack<Region> regionList, int thr) {
    // Il ciclo while prosegue fino a quando non si arriva 
    // ad avere solamente 4 elementi all'interno dello stack. 
    // Quindi si estraggono i quattro "nodi" dallo stack relativi 
    // alle regioni che sono state ottenute dal processo di 
    // splitting. Ovviamente poi devono essere eliminati dallo 
    // stack.      
    while(regionList.size() >= 4) {
        Region currentRegion_1 = regionList.top(); 
        regionList.pop(); 

        Region currentRegion_2 = regionList.top(); 
        regionList.pop(); 

        Region currentRegion_3 = regionList.top(); 
        regionList.pop(); 

        Region currentRegion_4 = regionList.top(); 
        regionList.pop();

        // I valori booleani vengono utilizzati per capire se si 
        // possono unire le regioni per righe o per colonne. 
        // Inizialmente vegono posti a false. 
        bool row_1, row_2, col_1, col_2; 
        row_1 = row_2 = col_1 = col_2 = false;
        
        // Si richiama la funzione per il Merge delle regioni sulle regioni 
        // delle righe. Quindi la regione 1 e la regione 2 e poi sulla regione 
        // 3 e sulla regione 4. 
        row_1 = MergeRegion(src, currentRegion_1, currentRegion_2, thr);
        row_2 = MergeRegion(src, currentRegion_3, currentRegion_4, thr); 

        // Se non si può unire per righe, allora si tenta di unire per 
        // colonne, utilizzando la stessa funzione, prima sulle regioni 1 
        // e 3 e poi sulle regioni 2 e 4. 
        if(!row_1 && !row_2) {
            col_1 = MergeRegion(src, currentRegion_1, currentRegion_3, thr); 
            col_2 = MergeRegion(src, currentRegion_2, currentRegion_4, thr); 

            // Le regioni che sono state modificate (il campo regionRec) vengono 
            // inserite all'interno dello stack per poter essere rielaborate con 
            // altre regioni vicine. 
            regionList.push(currentRegion_1); 
            regionList.push(currentRegion_2);
        } else {
            // Lo stesso viene fatto per quanto riguarda il caso in cui siano 
            // le righe ad essere unite. 
            regionList.push(currentRegion_1); 
            regionList.push(currentRegion_3);
        }
    }
}

int main(int argc, char *argv[]) {
    // Si legge l'immagine da riga di comando, e la si converte
    // in grigio. 
    String inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);

    // Per poter ottenere un risultato ottimale si applica un filtro Gaussiano 
    // sull'immagine di input, la maschera è di dimensione 5x5, mentre la 
    // deviazione standard è pari a 0. 
    GaussianBlur(inputImage, inputImage, Size(5, 5), 0, 0);

    // L'immagine originale viene mostrata in un'apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    int inputCols = inputImage.cols; 
    int inputRows = inputImage.rows; 
    int threshold = atoi(argv[2]); 
    Mat resultImage = inputImage.clone(); 

    Mat resultSrc = inputImage(Rect(0, 0, inputCols, inputRows)); 
    Rect resultRect = Rect(0, 0, inputCols, inputRows);

    // Viene richiamata la funzione di Split sull'immagine originale
    // ed il risultato, essendo uno stack delle regioni ottenute viene 
    // salvato in un apposito stack. Su tale stack viene effettuata 
    // l'operazione di Merge.  
    stack<Region> regionList = Split(resultSrc, resultRect, threshold); 
    Merge(resultImage, regionList, threshold); 

    // Si effettua sul risultato finale un processo di post-elaborazione 
    // per poter smussare i contorni delle regioni calcolate. 
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilate(resultImage, resultImage, kernel, Point(-1,-1), 2);
    erode(resultImage, resultImage, kernel, Point(-1,-1), 2);
    erode(resultImage, resultImage, kernel, Point(-1,-1), 2);
    dilate(resultImage, resultImage, kernel, Point(-1,-1), 2);  

    // Alla fine si mostra il risultato in un'apposita finestra. 
    namedWindow("Output Image", WINDOW_AUTOSIZE); 
    imshow("Output Image", resultImage);

    waitKey(0);  

    return 0; 
}