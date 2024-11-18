#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "Header.h"
using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");

    int n = 16; // количество узлов
    int n1 = 200; // для точных значений функции и интерполяции
    int n2 = 1000; // оценка ошибки

    double A = -1;
    double B = 1;

    func fk[8] = { f1, f2, f3, f4, f5, f6, f7, f8 };
    func f = fk[0];

    ofstream file1, file2;

    // сетки
    file1.open("uniformCoord.txt");
    vector<double> uniformX = uniformGrid(A, B, n);
    vector<double> uniformY(n);
    for (int i = 0; i < n; i++)
    {
        uniformY[i] = f(uniformX[i]);
        file1 << uniformX[i] << " " << uniformY[i] << endl;
    }
    file1.close();

    file2.open("chebyshevCoord.txt");
    vector<double> chebyshevX = chebyshevGrid(A, B, n);
    vector<double> chebyshevY(n);
    for (int i = 0; i < n; i++)
    {
        chebyshevY[i] = f(chebyshevX[i]);
        file2 << chebyshevX[i] << " " << chebyshevY[i] << endl;
    }
    file2.close();

    // точные значения функции
    file1.open("function.txt");
    vector<double> XX1 = uniformGrid(A, B, n1);
    vector<double> YY1(n1);
    for (int i = 0; i < n1; i++)
    {
        YY1[i] = f(XX1[i]);
        file1 << XX1[i] << " " << YY1[i] << endl;
    }
    file1.close();

    // интерполяция полиномом Лагранжа
    file1.open("UniformLagrange.txt");
    for (int i = 0; i < n1; i++)
    {
        YY1[i] = polynomLagrange(uniformX, uniformY, n, XX1[i]);
        file1 << XX1[i] << " " << YY1[i] << endl;
    }
    file1.close();

    file2.open("ChebyshevLagrange.txt");
    for (int i = 0; i < n1; i++)
    {
        YY1[i] = polynomLagrange(chebyshevX, chebyshevY, n, XX1[i]);
        file2 << XX1[i] << " " << YY1[i] << endl;
    }
    file2.close();
    
    // сплайн-интерполяция
    file1.open("UniformSpline.txt");
    for (int i = 0; i < n1; i++)
    {
        YY1[i] = cubicSpline(uniformX, uniformY, n, XX1[i]);
        file1 << XX1[i] << " " << YY1[i] << endl;
    }
    file1.close();
    
    /*
    file2.open("ChebyshevSpline.txt");
    for (int i = 0; i < n1; i++)
    {
        YY1[i] = cubicSpline(chebyshevX, chebyshevY, n, XX1[i]);
        file2 << XX1[i] << " " << YY1[i] << endl;
    }
    file2.close();
    */

    vector<double> XX2 = uniformGrid(A, B, n2);
    vector<double> YY2(n2);
    for (int i = 0; i < n2; i++)
    {
        YY2[i] = polynomLagrange(uniformX, uniformY, n, XX2[i]);
    }
    double error = errNorm(f, XX2, YY2, n2);
    cout << error << endl;
}