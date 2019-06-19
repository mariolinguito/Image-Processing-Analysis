#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;

// Definiamo una struttura per contenere le informazioni relative
// ad una particolare regione. 
// regionRect:   il rettangolo che corrisponde alla porzione di 
//               regione da considerare (relativa alla immagine);  
// regionSrc:    l'immagine relativa alla porzione di regione che 
//               viene considerata; 
// colorSrc:     il colore del pixel che viene calcolato con la 
//               funzione di OpenCV meanStdDev() che calcola la 
//               deviazione standard e la media di un set di dati; 
// regionChilds: è la lista dei nodi figli dell'attuale regione, 
//               ossia le regioni che sono state ottenute dalla 
//               suddivisione della regione attuale in quattro. 
// regionActive: è un flag che indica che se la regione che siamo 
//               elaborando deve essere considerata per la
//               colorazione finale per l'ottenimento dell'immagine 
//               finale. 
struct Region {
    Rect regionRect;
    Mat regionSrc;
    Scalar colorSrc; 
    vector<Region> regionChilds; 

    bool regionActive;  
}; 

Region Split(Mat src, Rect rec, int thr);
bool CheckHomogeneity(Mat region, int thr); 
bool MergeRegion(Mat src, Region R1, Region R2, int thr);  
void Merge(Mat src, Region R, int thr); 
void Fill(Mat src, Region region); 

Region Split(Mat src, Rect rec, int thr) {
    // La funzione Split è una funzione ricorsiva, ogni volta che viene richiamata viene 
    // creata una regione corrente che dovrà o meno essere divisa in quattro sottoregioni. 
    Region currentRegion; 
    currentRegion.regionRect = rec;
    currentRegion.regionSrc = src;
    currentRegion.regionActive = true; 

    // Per poter capire se dobbiamo dividere la regione in quattro sottoregioni richiamiamo 
    // la funzione CheckHomogeneity() che se è restituisce un false, allora la regine deve 
    // essere divisa. 
    if(!CheckHomogeneity(src, thr)) {
        // La regione deve essere divisa in quattro sottoregioni tutte di egual dimensioni, 
        // quindi calcoliamo la metà del numero di righe e la metà del numero di colonne. 
        int h = floor(src.rows/2);
        int w = floor(src.cols/2); 

        // Creiamo le quattro regioni, su ognuna delle quali si richiama ricorsivamente la 
        // funzione Split con gli appositi parametri. 
        // La prima regione parte dal punto (0, 0) ed ha grabndezza pari a w e h. 
        // La seconda regione parte dal punto (w, 0), è cioè la seconda regione in alto a 
        // destra (le dimensioni sono le stesse). 
        // La terza regione parte dal punto (0, h), riguarda cioè la regione in basso a 
        // sinistra (le dimensioni sono le stesse). 
        // La quarta regione parte dal punto (w, h), cioè riguarda la region in basso a 
        // destra, ed ha grandezza pari a quella delle altre tre regioni, ossia w e h. 
        Region firstRegion = Split(src(Rect(0, 0, w, h)), Rect(rec.x, rec.y, w, h), thr);
        Region secondRegion = Split(src(Rect(w, 0, w, h)), Rect(rec.x+w, rec.y, w, h), thr);
        Region thirdRegion = Split(src(Rect(0, h, w, h)), Rect(rec.x, rec.y+h, w, h), thr);
        Region fourthRegion = Split(src(Rect(w, h, w, h)), Rect(rec.x+w, rec.y+h, w, h), thr);

        // L'attuale regione ha come figli (considerando il quadtree) proprio le quattro regioni 
        // che sono state create in precedenza, quindi vengono inseriti all'interno del vettore. 
        currentRegion.regionChilds.push_back(firstRegion); 
        currentRegion.regionChilds.push_back(secondRegion); 
        currentRegion.regionChilds.push_back(thirdRegion); 
        currentRegion.regionChilds.push_back(fourthRegion);          
    } else {
        // Arrivati alla fine della ricorsione, per ogni regione che è stata creata viene calcolata 
        // la media e la deviazione standard con l'apposita funzione di OpenCV, quindi viene assegnato 
        // al campo relativo al colore la media calcolata. 
        Scalar mean, std; 
        meanStdDev(src, mean, std); 
        currentRegion.colorSrc = mean;  
    }

    // Ai fini della ricorsione si restituisce la regione corrente. 
    return currentRegion; 
}

// Per la verifica della omogeneità di due regioni viene confrontata la 
// deviazione standard con un valore numerico. Se questo è verificato 
// allora viene restituito un valore booleano true, vicevenrsa viene 
// restituito un false. Lo stesso viene fatto però anche nel caso in cui 
// la regione risulta essere troppo piccola per essere suddivisa ancora. 
// Si confronta a tal proposito il prodotto tra le colonne e le righe per 
// un valore dato in input. 
bool CheckHomogeneity(Mat src, int thr) {
    bool response = false; 
    Scalar mean, std; 
    meanStdDev(src, mean, std); 

    if(std[0] <= 5.8 || (src.rows*src.cols) <= thr) {
        response = true; 
    }

    // Il risultato deve essere restituito dopo i dovuti controlli. 
    return response; 
}

// I parametri di input della funzione sono la matrice data in input 
// e sulla quale si deve verificare la condizione di omogeneità. Poi 
// le due regioni da fondere, ed infine il valore threshold. 
bool MergeRegion(Mat src, Region R1, Region R2, int thr) {
    bool response = false; 

    // Possiamo effettuare il merge tra due regioni solo quando queste 
    // sono ovviamente le più piccole ottenute o quelle che rispettano 
    // il predicato di omogeneità, quindi si verifica se entrambe le 
    // regioni non hanno figli, se è così allora si possono fondere le 
    // due regioni, ottenendo una terza regione di tipo Rect, sulla quale 
    // viene ricalcolato il predicato di omogeneità. Se questo viene
    // verificato allora nel campo regionRect della prima regione viene 
    // inserita la regione fusa, mentre viene posto il flag regionActive 
    // della seconda regione a false, in modo da non essere calcolata 
    // quando le regioni saranno colorate. 
    if(R1.regionChilds.size() == 0 && R2.regionChilds.size() == 0) {
        Rect Region1 = R1.regionRect; 
        Rect Region2 = R2.regionRect; 
        Rect Region12 = Region1|Region2;

        if(CheckHomogeneity(src(Region12), thr)) {
            R1.regionRect = Region12; 
            R2.regionActive = false; 
            response = true; 
        }
    }

    // Ovviamente se le regioni sono state fuse, ma non rispettano il 
    // predicato di omogeneità, allora viene restituito un false, viceversa
    // viene restituito un true. 
    return response; 
}

void Merge(Mat src, Region R, int thr) {
    // Le variabili booleane vengono utilizzate per indicare se i tentativi 
    // di fusione tra le righe o le colonne sono andati a buon fine o meno. 
    bool row1 = false; 
    bool row2 = false; 
    bool col1 = false; 
    bool col2 = false;

    // Essendo una funzione ricorsiva, tale viene richiamata fintanto che non 
    // si arriva al nodo il cui vettore dei nodi figli è vuoto, cioè fintanto 
    // che non arriviamo alle foglie dell'albero. 
    if(R.regionChilds.size() < 1) {
        return; 
    }

    // Proviamo a fare il merge delle righe tra le due regioni che sono figlie 
    // dello stesso nodo. 
    row1 = MergeRegion(src, R.regionChilds[0], R.regionChilds[1], thr); 
    row2 = MergeRegion(src, R.regionChilds[2], R.regionChilds[3], thr);

    // Se una delle due fusioni non è andata a buon fine, allora viene fatto un 
    // tentativo di fusione per quanto riguarda le colonne delle stesse regioni 
    // figlie dello stesso nodo. 
    if(!row1 && !row2) {
        col1 = MergeRegion(src, R.regionChilds[0], R.regionChilds[2], thr); 
        col2 = MergeRegion(src, R.regionChilds[1], R.regionChilds[3], thr); 
    } 

    // Attraverso il ciclo for si richiama ricorsivamente la funzione di merge 
    // su ognuno dei nodi figli del nodo corrente. Ovviamente solo se il nodo 
    // ha dei figli.  
    for(int i = 0; i < R.regionChilds.size(); i++) {
        if(R.regionChilds[i].regionChilds.size() > 0) {
            Merge(src, R.regionChilds[i], thr); 
        }
    }  
}

void Fill(Mat src, Region region) {
    // La funzione Fill() viene utilizzata per colorare le regioni che sono 
    // quelle risultanti dalle elaborazioni fatte in precedenza. Viene disegnato 
    // un rettangolo riempito del colore della regione solo quando la regione che 
    // stiamo considerando è una regione che non ha figli (quindi non è stata divisa) 
    // ed è una regione attiva. 
    if(region.regionChilds.size() == 0 && region.regionActive == true) {
        rectangle(src, region.regionRect, region.colorSrc, FILLED);
    } 

    // Per ogni figlio della regione corrente viene richiamata ricorsivamente la
    // stessa funzione Fill(). 
    for(int i = 0; i < region.regionChilds.size(); i++) {
        Fill(src, region.regionChilds[i]); 
    }
}

int main(int argc, char *argv[]) {
    // Viene letta l'immagine da riga di comando, ed inserita nella 
    // variabile inputImage. 
    String inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);

    // Quindi l'immagine originale viene mostrata all'interno di 
    // un'apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    int inputCols = inputImage.cols; 
    int inputRows = inputImage.rows; 
    int threshold = atoi(argv[2]); 
    Mat resultImage = inputImage.clone(); 

    // La prima regione da considerare è ovviamente l'immagine
    // per intera, che viene data in input alla funzione di Split. 
    Mat resultSrc = inputImage(Rect(0, 0, inputCols, inputRows)); 
    Rect resultRect = Rect(0, 0, inputCols, inputRows); 

    // Il primo passo da compiere sull'immagine risultante è quello di
    // eseguire lo Split in sottoregioni. 
    Region regionSplitted = Split(resultSrc, resultRect, threshold);
    
    // Una volta eseguito lo Split in sottoregioni, queste devono essere 
    // fuse secondo i criteri descritti per ogni funzione utilizzata.  
    Merge(inputImage, regionSplitted, threshold);

    // Il passo finale è quello di colorare le regioni attive con i 
    // relativi colori, per ottenere il risultato finale. 
    Fill(resultImage, regionSplitted);

    // Eseguiamo la post-elaborazione dell'immagine per rendere 
    // le regioni che sono state ottenute più smussate. 
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilate(resultImage, resultImage, kernel, Point(-1, -1), 2);
    erode(resultImage, resultImage, kernel, Point(-1, -1), 2);
    erode(resultImage, resultImage, kernel, Point(-1, -1), 2);
    dilate(resultImage, resultImage, kernel, Point(-1, -1), 2); 

    // L'immagine risultante viene mostrato in un'apposita finestra. 
    namedWindow("Output Image", WINDOW_AUTOSIZE); 
    imshow("Output Image", resultImage);

    waitKey(0);  

    return 0; 
}