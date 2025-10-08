#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 28    //piksel boyutu
#define PIXELS (N*N)     
#define DATA_SIZE 2500     //Veri seti boyutu
#define TRAIN_RATIO 0.8    //Eğitim/Test oranı
#define TRAIN_SIZE (int)(DATA_SIZE*TRAIN_RATIO)
#define TEST_SIZE (DATA_SIZE-TRAIN_SIZE)
#define LEARNING_RATE 0.3
#define EPOCHS 100
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
double data1[DATA_SIZE][PIXELS+1];  
double labels1[DATA_SIZE];                     
double weights_gd[PIXELS+1];      //GD için ağırlıklar
double weights_sgd[PIXELS+1];     //SGD için ağırlıklar
double weights_adam[PIXELS+1];    //Adam için ağırlıklar
double m[PIXELS+1];               //hareketli ortalama
double v[PIXELS+1];               //kare ortalama
double t = 0;                       //zaman 

double **allocate_2d_array(int rows, int cols) {
    double **array = malloc(rows*sizeof(double*));
    if (!array) {
        perror("Bellek ayirma hatasi (satirlar)");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        array[i] = malloc(cols*sizeof(double));
        if (!array[i]) {
            perror("Bellek ayirma hatasi (sutunlar)");
            exit(EXIT_FAILURE);
        }
    }
    return array;
}

double activation(double z) {
    return tanh(z);
}

double activation_derivative(double z) {
    return 1.0 - tanh(z) * tanh(z);
}

void load_data(const char *filename, double data[DATA_SIZE][PIXELS + 1], double labels[DATA_SIZE]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("CSV dosyasi acilamadi");
        exit(EXIT_FAILURE);
    }
    char line[8192];
    fgets(line, sizeof(line), file); // İlk satırı atla (başlıklar)
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < DATA_SIZE) {
        char *token = strtok(line, ",");
        int original_label = atoi(token);
        if (original_label == 3) {
            labels[row] = 1.0;
        } else if (original_label == 7) {
            labels[row] = -1.0; 
        } else {    
            continue; 
        }
        for (int col = 0; col < PIXELS; col++) {
            token = strtok(NULL, ",");
            data[row][col] = atof(token) / 255.0; 
        }
        data[row][PIXELS] = 1.0; 
        row++;
    }
    fclose(file);
}

void split_data(double **train_data, double **test_data, double *train_labels, 
double *test_labels,double data[DATA_SIZE][PIXELS+1],double labels[DATA_SIZE]) {
    int i;
    for (i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j <= PIXELS; j++) {
            train_data[i][j] = data[i][j];
        }
        train_labels[i] = labels[i];
    }
    for (; i < DATA_SIZE; i++) {
        for (int j = 0; j <= PIXELS; j++) {
            test_data[i - TRAIN_SIZE][j] = data[i][j];
        }
        test_labels[i - TRAIN_SIZE] = labels[i];
    }
}

void initialize_weights(double *weights) {  //Eğer tüm ağırlıklar 0 veya aynı sabit bir değerle başlatılırsa, modelin öğrenme sürecinde farklı özellikler arasında bir fark yaratılmaz. Bu durum, özellikle sinir ağlarında, simetri problemi olarak bilinir ve öğrenmeyi engeller.
    for (int i = 0; i <= PIXELS; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; 
    }
}

double forward(double *weights, double *x) {
    double sum = 0.0;
    for (int i = 0; i <= PIXELS; i++) {
        sum += weights[i] * x[i];
    }
    return activation(sum);
}

// GD güncellemesi
void gradient_descent_update(double *weights, double *x, double y) {
    double output = forward(weights, x);
    double error = y - output;
    double gradient = error * activation_derivative(output);

    for (int i = 0; i <= PIXELS; i++) {
        weights[i] += LEARNING_RATE * gradient * x[i];
    }
}

// SGD güncellemesi
void stochastic_gradient_descent_update(double *weights, double **data, double *labels, int data_size) {
    int random_index = rand() % data_size; 
    double *x = data[random_index];       
    double y = labels[random_index];      
    gradient_descent_update(weights, x, y); 
}

// Adam güncellemesi
void adam_update(double *weights, double *x, double y) {
    t += 1; // Zaman adımını artır
    double output = forward(weights, x);
    double error = y - output;
    double gradient = error * activation_derivative(output);

    for (int i = 0; i <= PIXELS; i++) {
        m[i] = BETA1 * m[i] + (1 - BETA1) * gradient * x[i];
        v[i] = BETA2 * v[i] + (1 - BETA2) * gradient * x[i] * gradient * x[i];

        double m_hat = m[i] / (1 - pow(BETA1, t));
        double v_hat = v[i] / (1 - pow(BETA2, t));

        weights[i] += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
    }
}

// Eğitim döngüsü
void train_and_log(double **train_data, double *train_labels, double *weights, 
                   void (*update_fn)(double *, double *, double), const char *log_filename) {
    FILE *log_file = fopen(log_filename, "w");
    if (!log_file) {
        perror("Log dosyasi yazma hatasi");
        exit(EXIT_FAILURE);
    }
    fprintf(log_file, "Epoch,Loss\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        for (int i = 0; i < TRAIN_SIZE; i++) {
            double output = forward(weights, train_data[i]);
            double error = train_labels[i] - output;
            loss += error * error; 
            update_fn(weights, train_data[i], train_labels[i]); // Ağırlıkları güncelle
        }
        loss /= TRAIN_SIZE;
        fprintf(log_file, "%d,%.4f\n", epoch + 1, loss);
        printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, EPOCHS, loss);
        if (loss < 0.0001) {
            break;
        }
    }
    fclose(log_file);
}


void train_sgd_and_log(double **train_data, double *train_labels, double *weights, 
                       int train_size, const char *log_filename) {
    FILE *log_file = fopen(log_filename, "w");
    if (!log_file) {
        perror("Log dosyasi yazma hatasi");
        exit(EXIT_FAILURE);
    }
    fprintf(log_file, "Epoch,Loss\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        for (int i = 0; i < train_size; i++) {
            stochastic_gradient_descent_update(weights, train_data, train_labels, train_size);
        }
        for (int i = 0; i < train_size; i++) {
            double output = forward(weights, train_data[i]);
            double error = train_labels[i] - output;
            loss += error * error; // MSE
        }
        loss /= train_size;
        fprintf(log_file, "%d,%.4f\n", epoch + 1, loss);
         printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, EPOCHS, loss);
    }
    fclose(log_file);
}


void evaluate(double **test_data, double *test_labels, double *weights) {
    int correct = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        double output = forward(weights, test_data[i]);
        int prediction = (output >= 0.0) ? 1 : -1;
        if (prediction == (int)test_labels[i]) {
            correct++;
        }
    }
    printf("Test Dogrulugu: %.2f%%\n", (correct / (double)TEST_SIZE) * 100);
}
void write_results_to_csv(const char *filename, double **test_data, double *test_labels, double *weights) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("CSV dosyasi yazma hatasi");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "Gercek Etiket,Tahmin\n");
    int correct = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        double output = forward(weights, test_data[i]);
        int prediction = (output >= 0.0) ? 1 : -1;
        if (prediction == (int)test_labels[i]) {
            correct++;
        }
        fprintf(file, "%.0f,%d\n", test_labels[i], prediction);
    }
    double accuracy = (correct / (double)TEST_SIZE) * 100;
    fprintf(file, "\nDoğruluk: %.2f%%\n", accuracy);

    fclose(file);
}


int main() {
    srand(time(NULL));

    // Veri kümeleri için bellek ayırma
    double **train_data1 = allocate_2d_array(TRAIN_SIZE, PIXELS + 1);
    double **test_data1 = allocate_2d_array(TEST_SIZE, PIXELS + 1);
    double *train_labels1 = malloc(TRAIN_SIZE * sizeof(double));
    double *test_labels1 = malloc(TEST_SIZE * sizeof(double));
    load_data("filtered_data.csv", data1, labels1);
    
    split_data(train_data1, test_data1, train_labels1, test_labels1, data1, labels1);
   

    printf("Gradient Descent\n");
    srand(time(NULL));
    initialize_weights(weights_gd);
    train_and_log(train_data1, train_labels1, weights_gd, gradient_descent_update, "loss_gd.csv");
    evaluate(test_data1, test_labels1, weights_gd);

    printf("\nStochastic Gradient Descent (SGD)\n");
    srand(time(NULL));
    initialize_weights(weights_sgd);
    train_sgd_and_log(train_data1, train_labels1, weights_sgd, TRAIN_SIZE, "loss_sgd.csv");
    evaluate(test_data1, test_labels1, weights_sgd);

    printf("\nAdam Optimizasyonu\n");
    srand(time(NULL));
    initialize_weights(weights_adam);
    train_and_log(train_data1, train_labels1, weights_adam, adam_update, "loss_adam.csv");
    evaluate(test_data1, test_labels1, weights_adam);
    
    
    for (int i = 0; i < TRAIN_SIZE; i++) 
        free(train_data1[i]);

    for (int i = 0; i < TEST_SIZE; i++) 
        free(test_data1[i]);
    
    free(train_data1);
    free(test_data1);
    free(train_labels1);free(test_labels1);
    return 0;
}
