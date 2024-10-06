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
    const T eps = 1e-4;
    const T tau = 0.015;
    const T omega = 1.2;
    const int norm = 1; // 1 - октаэдрическая, 0 - кубическая (inf)

    cout << "МЕТОД ПРОСТОЙ ИТЕРАЦИИ:" << endl;
    Params resultSimpleIter = SimpleIterationMethod(A, b, x0, tau, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultSimpleIter.x[i] << endl;
    cout << "Количество итераций: " << resultSimpleIter.iterCount << endl;
    // cout << resultSimpleIter.normC1 << endl;

    cout << endl << "МЕТОД ЯКОБИ:" << endl;
    Params resultJacobi = JacobiMethod(A, b, x0, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultJacobi.x[i] << endl;
    cout << "Количество итераций: " << resultJacobi.iterCount << endl;

    cout << endl << "МЕТОД ЗЕЙДЕЛЯ:" << endl;
    Params resultSeidel = SeidelMethod(A, b, x0, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultSeidel.x[i] << endl;
    cout << "Количество итераций: " << resultSeidel.iterCount << endl;

    cout << endl << "МЕТОД РЕЛАКСАЦИИ:" << endl;
    Params resultRelaxation = RelaxationMethod(A, b, x0, omega, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultRelaxation.x[i] << endl;
    cout << "Количество итераций: " << resultRelaxation.iterCount << endl;
}
