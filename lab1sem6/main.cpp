#include <iostream>
#include "Header.h"
#include <vector>
using namespace std;

int main()
{
    double h = 0.01; // шаг
    double t0 = 0;
    double tn = 4; // отрезок
    vector<double> u0 = { 1, 0 }; // задача Коши

    EulerExplicit(*testM, u0, h, t0, tn);
    EulerImplicit(*testM, u0, h, t0, tn);
    SymmetricScheme(*testM, u0, h, t0, tn);
    RungeKutta2(*testM, u0, h, t0, tn);
    RungeKutta4(*testM, u0, h, t0, tn);
    PredictorCorrector(*testM, u0, h, t0, tn);
    AdamsBashforth(*testM, u0, h, t0, tn);
}
