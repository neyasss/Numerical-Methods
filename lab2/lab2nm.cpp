#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "Header.h"
using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");
    vector<vector<T>> A;
    vector<T> b;
    readSLAE("test.txt", A, b);
    int n = A.size();

    cout << "Исходная СЛАУ:" << endl;
    printSLAE(A, b, n);
    cout << endl;

    vector<T> x0 = b; // в качестве начального приближения берем вектор правой части

    Params resultSimpleIter = SimpleIterationMethod(A, b, x0, 0.015, 1e-4, n, 1);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultSimpleIter.x[i] << endl;
    cout << "Количество итераций: " << resultSimpleIter.iterCount << endl;
    // cout << resultSimpleIter.normC1 << endl;

    Params resultJacobi = JacobiMethod(A, b, x0, 1e-4, n, 1);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultJacobi.x[i] << endl;
    cout << "Количество итераций: " << resultJacobi.iterCount << endl;
}