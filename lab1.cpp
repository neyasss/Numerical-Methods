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

    cout << "Решение СЛАУ методом Гаусса:" << endl;
    vector<T> solution1 = GaussianMethod(A, b, n);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << solution1[i] << endl;

    ResidualVectorNorm(A, b, solution1, n);

    cout << endl << "Решение СЛАУ методом QR-разложения:" << endl;
    vector<T> solution2 = QRMethod(A, b, n);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << solution2[i] << endl;

    ResidualVectorNorm(A, b, solution2, n);

    vector<vector<T>> Ainv = InvLU(A, n);
    cout << endl << "Обратная матрица для матрицы системы A:" << endl;
    printMatrix(Ainv, n);

    cout << endl << "Число обусловленности матрицы A при использовании" << endl;
    cout << "октаэдрической нормы: " << cond1(A, n) << endl;
    cout << "кубической нормы: " << condInf(A, n) << endl;

    vector<T> disturb = { 0.01, 0.01, 0.01, 0.01 }; // возмущение
    // condEstimation(A, b, disturb, n);

    cout << endl << "Оценка числа обусловленности:";
    condEstimationLower(A, b, n);

    cout << endl << "Результат умножения A^(-1) на A:" << endl;
    vector<vector<T>> M = MatrixMult(A, Ainv, n);
    printMatrix(M, n);
}
