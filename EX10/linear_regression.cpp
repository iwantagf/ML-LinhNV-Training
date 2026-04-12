#include <bits/stdc++.h>
using namespace std;
double learning_rate = 0.01;
int training_steps = 1000;
int display_step = 50;

double X[] = {3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1};
double y[] = {1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3};
double w, b;

double calc(double x) {
    return w * x + b;
}

double loss() {
    double total_loss = 0;
    for (int i = 0; i < sizeof(X) / sizeof(double); ++i) {
        double y_pred = calc(X[i]);
        total_loss += (y_pred - y[i]) * (y_pred - y[i]);
    }
    return total_loss / (sizeof(X) / sizeof(double));
}

double dw() {
    double total_dw = 0;
    for (int i = 0; i < sizeof(X) / sizeof(double); ++i) {
        double y_pred = calc(X[i]);
        total_dw += (y_pred - y[i]) * X[i];
    }
    return total_dw / (sizeof(X) / sizeof(double));
}

double db() {
    double total_db = 0;
    for (int i = 0; i < sizeof(X) / sizeof(double); ++i) {
        double y_pred = calc(X[i]);
        total_db += (y_pred - y[i]);
    }
    return total_db / (sizeof(X) / sizeof(double));
}

void backward() {
    double dw_ = dw();
    double db_ = db();
    w -= learning_rate * dw_;
    b -= learning_rate * db_;
}


int main() {
    vector<double> losses;
    for (int step = 0; step < training_steps; ++step) {
        double l = loss();
        losses.push_back(l);
        backward();
        if (step % display_step == 0) {
            cout << "Step: " << step << ", Loss: " << l << ", w: " << w << ", b: " << b << endl;
        }
    }
}