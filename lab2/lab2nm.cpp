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
    const T tau = 0.05;
    const T omega = 0.5;
    const int norm = 0; // 1 - октаэдрическая, 0 - кубическая (inf)

    cout << "МЕТОД ПРОСТОЙ ИТЕРАЦИИ:" << endl;
    Params resultSimpleIter = SimpleIterationMethod(A, b, x0, tau, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultSimpleIter.x[i] << endl;
    cout << "Количество итераций: " << resultSimpleIter.iterCount << endl;
    cout << resultSimpleIter.normC1 << endl;
    cout << resultSimpleIter.normCInf << endl;
    cout << ResidualVectorNorm(A, b, resultSimpleIter.x, n, norm) << endl;

    cout << endl << "МЕТОД ЯКОБИ:" << endl;
    Params resultJacobi = JacobiMethod(A, b, x0, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultJacobi.x[i] << endl;
    cout << "Количество итераций: " << resultJacobi.iterCount << endl;
    cout << resultJacobi.normC1 << endl;
    cout << resultJacobi.normCInf << endl;
    cout << ResidualVectorNorm(A, b, resultJacobi.x, n, norm) << endl;

    cout << endl << "МЕТОД ЗЕЙДЕЛЯ:" << endl;
    Params resultSeidel = SeidelMethod(A, b, x0, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultSeidel.x[i] << endl;
    cout << "Количество итераций: " << resultSeidel.iterCount << endl;
    cout << resultSeidel.normC1 << endl;
    cout << resultSeidel.normCInf << endl;
    cout << ResidualVectorNorm(A, b, resultSeidel.x, n, norm) << endl;

    cout << endl << "МЕТОД РЕЛАКСАЦИИ:" << endl;
    Params resultRelaxation = RelaxationMethod(A, b, x0, omega, eps, n, norm);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << resultRelaxation.x[i] << endl;
    cout << "Количество итераций: " << resultRelaxation.iterCount << endl;
    cout << resultRelaxation.normC1 << endl;
    cout << resultRelaxation.normCInf << endl;
    cout << ResidualVectorNorm(A, b, resultRelaxation.x, n, norm) << endl;

    int N = 208;
    cout << endl << "Трехдиагональная матрица большой размерности: " << endl;
    vector<T> a(N, 1), bb(N, 4), c(N, 1), d(N);
    vector<T> solution(N), x0diag(N, 0);
    for (int i = 0; i < N - 1; i++)
    {
        d[i] = 10 - 2 * (i % 2);
        solution[i] = 2 - (i % 2);
    }
    d[0] = 6;
    d[N - 1] = 9 - 3 * (N % 2);
    Params xDiag = SeidelMethodTriDiagonal(a, bb, c, d, x0diag, 1e-4, N, norm);
    cout << ResidualVectorNormTriDiagonal(a, bb, c, d, xDiag.x, N, norm);
}
