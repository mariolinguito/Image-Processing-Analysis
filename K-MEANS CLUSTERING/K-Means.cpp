#include <cstdio>
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv; 

// La classe Cluster tiene traccia di tutte le informazioni relative ad 
// un particolare Cluster, in particolare: 
// 1. La lista dei pixel che sono in esso contenuti; 
// 2. Il centroide attuale del Cluster; 
// 3. Il numero dei pixel che sono in esso inseriti; 
// 4. Le variabili intere per la sommatoria delle componenti dell'oggetto 
//    Vec3b per il calcolo del nuovo centroide; 
class Cluster {
    public: 
    vector<Point> pixelsCoordsList;
    Vec3b clusterCentroyd;  
    int pixelNum = 1;

    int sumB = 0; 
    int sumG = 0; 
    int sumR = 0; 

    Cluster() {} 

    // Il metodo InsertPixel(...) prende in input le coordinate del punto da 
    // inserire all'interno del Cluster e l'oggetto Vec3b dello stesso pixel
    // in modo tale da poter sommare le sue componenti con quelle degli altri 
    // pixel ivi contenuti. Le coordinate dei pixel che compongono il Cluster 
    // sono gestiti in una apposita lista, e contestualmente si aumenta il 
    // contatore dei pixel che sono contenuti nella regione. 
    void InsertPixel(int x, int y, Vec3b currentPixel3b) {
        this->pixelsCoordsList.push_back(Point(x, y));

        this->sumB += currentPixel3b.val[0]; 
        this->sumG += currentPixel3b.val[1]; 
        this->sumR += currentPixel3b.val[2]; 

        this->pixelNum++; 
    }

    // Il metodo SetCentroyd(...) ricalcola il centroide della regione che è dato 
    // dalla media ottenuta semplicemente come la somma di tutte le componenti dei 
    // pixel diviso il numero di pixel, e questo per tutte e tre i canali del punto. 
    void SetCentroyd() {
        Vec3b newCentroyd;

        newCentroyd.val[0] = this->sumB/this->pixelNum;
        newCentroyd.val[1] = this->sumG/this->pixelNum;
        newCentroyd.val[2] = this->sumR/this->pixelNum;

        // Il nuovo centroide va a sostituire il precedente centroide della regione. 
        this->clusterCentroyd = newCentroyd;

        // Tutte le variabili di lavoro vengono resettate per le successive iterazioni. 
        this->pixelNum = 1; 
        this->sumB = 0; 
        this->sumG = 0; 
        this->sumR = 0; 
    }
};

Mat K_Means(Mat src, int k, int iterations, int thr); 

// Gli input della funzione K_Means sono l'immagine originale, il numero 
// di Cluster da creare, il numero di iterazioni da effettuare ed il numero 
// relativo al threshold con il quale confrontare la distanza tra il vecchio 
// centroide della regione ed il nuovo ricalcolato nell'iterazione attuale. 
Mat K_Means(Mat src, int k, int iterations, int thr) {
    // I due vettori vengono utilizzati uno per i centroidi dei Cluster e 
    // uno per i Cluster stessi dell'immagine. 
    vector<Vec3b> centroyds; 
    vector<Cluster> clusters; 

    int it = 0; 
    bool breakable; 

    // 1. Selezioniamo dall'immagine i primi centroidi dei cluster 
    // in maniera random, vengono poi inseriti all'interno della 
    // lista dei centroidi, mentre il Cluster appena creato viene 
    // inserito all'interno della lista dei Cluster.  
    for(int i = 0; i < k; i++) {
        int randX = rand() % src.rows; 
        int randY = rand() % src.cols; 
        
        Vec3b currentPixel = src.at<Vec3b>(randX, randY); 
        Cluster currentCluster = Cluster(); 

        centroyds.push_back(currentPixel); 
        clusters.push_back(currentCluster); 
    }

    for(int it = 0; it < iterations; it++) {
        // 2. Per tutti i punti dell'immagine dobbiamo trovare quei pixel
        // che sono più vicini ad uno dei centroidi selezionati in precedenza.
        // Per fare questo si scorre tutta l'immagine e per ogni pixel si 
        // calcola la distanza da tutti i centroidi. Queste distanze vengono 
        // inserite all'interno di una lista.  
        for(int i = 0; i < src.rows; i++) {
            for(int j = 0; j < src.cols; j++) {
                vector<double> distances; 
                Vec3b pixelValue = src.at<Vec3b>(i, j);

                // 3. Per ognuno dei cluster calcoliamo la distanza che intercorre
                // tra il punto in questione ed il centroide del cluster. 
                for(int x = 0; x < k; x++) {
                    Vec3b currentCentroyd = centroyds.at(x); 

                    // La distanza che viene calcolata è semplicemente quella Euclidea, adattata 
                    // al contesto in cui ci sono tre componenti. 
                    double distance0 = currentCentroyd.val[0]-pixelValue.val[0]; 
                    double distance1 = currentCentroyd.val[1]-pixelValue.val[1];
                    double distance2 = currentCentroyd.val[2]-pixelValue.val[2];  
                    double currentDistance = sqrt(pow(distance0, 2) + pow(distance1, 2) + pow(distance2, 2));

                    distances.push_back(currentDistance); 
                }

                // 5. Troviamo per il punto corrente la minima distanza tra quelle calcolate. Quindi 
                // si scorre tutta la lista delle distanze e si trova quella minima. Il Cluster che 
                // dovrà contenere quel pixel ha indice index.  
                int index = 0; 
                for(int i = 0; i < k; i++) {
                    if(distances.at(i) < distances.at(index)) {
                        index = i; 
                    }
                }

                // 6. Inseriamo il punto nel relativo Cluster, utilizzando l'apposito metodo della classe. 
                clusters.at(index).InsertPixel(i, j, pixelValue); 
                distances.clear(); 
            } 
        }

        centroyds.clear(); 
        breakable = false; 

        // Per ognuno dei Cluster si deve ricalcolare il centroide essendo 
        // adesso composti da una serie di pixel. Il centroide viene ricalcolato 
        // con l'apposito metodo della funzione, ma prima viene salvato il vecchio 
        // centroide per poterlo confrontare (in quanto a distanza) con il nuovo 
        // centroide. Se uno dei centroidi non è stato modificato sostanzialmente, 
        // ed infatti a questo serve il threshold, allora si deve bloccare il ciclo 
        // for e terminare l'algoritmo. 
        for(int i = 0; i < clusters.size(); i++) {
            Vec3b prevCentroyd = clusters.at(i).clusterCentroyd; 

            clusters.at(i).SetCentroyd(); 
            centroyds.push_back(clusters.at(i).clusterCentroyd); 

            // Si deve calcolare la distanza tra il vecchio centroide e quello nuovo, 
            // e questo viene fatto attraverso la norma. 
            double distance = norm(prevCentroyd, clusters.at(i).clusterCentroyd);
            
            // Se la distanza è maggiore del threshold che abbiamo scelto, allora 
            // l'algoritmo può continuare, altrimenti deve fermarsi. 
            if(distance > thr) {
                breakable = false;
            }
        }

        // Se la variabile breakable è vera allora si deve terminare l'algoritmo. 
        if(breakable) {
            break; 
        }
    }

    // Si deve creare l'immagine risultante dalle operazioni precedenti, quindi una
    // volta usciti dal ciclo for si deve scorrere su tutti i Cluster e per ogni 
    // Cluster scorrere la lista dei pixel per poter assegnare all'immagine finale 
    // le cui coordinate coincidono con quelle del pixel scelto il valore del pixel 
    // della lista. 
    Mat resultImage = Mat(src.size(), src.type(), Scalar::all(0));  

    for(int cluster = 0; cluster < k; cluster++) {
        for(int pixel = 0; pixel < clusters.at(cluster).pixelsCoordsList.size(); pixel++) {
            int x = clusters.at(cluster).pixelsCoordsList.at(pixel).x;
            int y = clusters.at(cluster).pixelsCoordsList.at(pixel).y; 

            resultImage.at<Vec3b>(x, y) = clusters.at(cluster).clusterCentroyd;  
        }
    }

    // Alla fine si restituisce il risultato finale. 
    return resultImage; 
}

int main(int argc, char *argv[]) {
    // Si legge l'immagine di input da riga di comando, che deve 
    // essere a colori. 
    String inputFile = argv[1]; 
    Mat inputImage = imread(inputFile, IMREAD_ANYCOLOR);

    // Si mostra l'immagine all'interno di una apposita finestra. 
    namedWindow("Original Image", WINDOW_AUTOSIZE); 
    imshow("Original Image", inputImage); 

    // Da riga di comando si prendono anche i parametri: 
    // k:          numero di Cluster da creare, 
    // iterations: numero di iterazione da fare al massimo, 
    // threshold:  il valore che la distanza deve superare per poter 
    //             permettere all'algoritmo di continuare. 
    int k = atoi(argv[2]); 
    int iterations = atoi(argv[3]); 
    int threshold = atoi(argv[4]); 

    Mat resultImage = K_Means(inputImage, k, iterations, threshold);

    // Il risultato finale viene mostrato in un'apposita finestra. 
    namedWindow("K-Means Image", WINDOW_AUTOSIZE); 
    imshow("K-Means Image", resultImage); 

    waitKey(0); 

    return 0; 
}