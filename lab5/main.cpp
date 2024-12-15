#include <iostream>
#include "Header.h"

auto test = [](double x) { 
    //return (x - 0.1) * (x - 0.22) * (x - 0.55) * (x - 0.7) * (x - 0.75);
    //return sqrt(x + 1) - 1; 
    //return 35 * x * x * x - 67 * x * x - 3 * x + 3; 
    return cos(x * x * x * x * x) - x + 3 + pow(2, 1 / 3) + atan((x * x * x - 5 * sqrt(2) * x - 4) / (sqrt(6) * x + sqrt(3))) + 1.8;
    };

auto test_1 = [](double x1, double x2) { 
    return x1 * x1 - x2 * x2 - 15; 
    //return x1 * x1 + x2 * x2 + x1 + x2 - 8;
    //return exp(x1) - x2 + 4;
};
auto test_2 = [](double x1, double x2) { 
    return x1 * x2 + 4; 
    //return x1 * x1 + x2 * x2 + x1 * x2 - 7;
    //return sqrt(x1) - 2 * x1 - 3;
};

int main(){
    setlocale(LC_ALL, "Russian");
    vector<double> localized_interval = localize(test, -1, 10, 10);
    bisection(test, localized_interval[0], localized_interval[1], 1e-06);
    newton(test, localized_interval[0], localized_interval[1]);
    newton_2(test_1, test_2, 1e-6, -5., -6.);
}
