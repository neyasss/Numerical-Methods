#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // обычная точность 
#define T double                            // повышенная точность

// лр1

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b); // чтение СЛАУ из файла

void printSLAE(const vector<vector<T>>& A, const vector<T>& b, int n); // вывод СЛАУ на экран
void printMatrix(const vector<vector<T>>& A, int n); // вывод матрицы на экран

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B, int n); // умножение матриц
vector<vector<T>> Transpose(const vector<vector<T>>& A, int n); // транспонирование

T ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, int n, int norm); // норма вектора невязки

vector<vector<T>> InvLU(const vector<vector<T>>& A, int n); // нахождение обратной матрицы с помощью LU-разложения

T vectorNorm1(const vector<T>& b, int n); // векторная октаэдрическая норма
T vectorNormInf(const vector<T>& b, int n); // векторная кубическая норма

T matrixNorm1(const vector<vector<T>>& A, int n); // матричная октаэдрическая норма
T matrixNormInf(const vector<vector<T>>& A, int n); // матричная кубическая норма

// Число обусловленности для различных матричных норм
T cond1(const vector<vector<T>>& A, int n);
T condInf(const vector<vector<T>>& A, int n);

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// 
// лр2

struct Params
{
	vector<vector<T>> C;
	vector<T> y;
	vector<T> x; // решение
	int iterCount; // количество итераций
	T normC1, normCInf; // нормы матрицы C
};

// vector<T> GenDiagVec(vector<T>& a, int num, int ind); // генерация диагональных векторов
// vector<T> GenResultVec(vector<T>& d); // генерация диагональных векторов
T ResidualVectorNormTriDiagonal(vector<T> a, vector<T> b, vector<T> c, vector<T> d, const vector<T>& x, int n, int norm); // норма вектора невязки для тридиагональной матрицы

void LDU(vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U, int n);

Params SimpleIterationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);

Params JacobiMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& eps, int n, int norm);

Params SeidelMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& eps, int n, int norm);

Params RelaxationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& omega, const T& eps, int n, int norm);

Params SeidelMethodTriDiagonal(vector<T> a, vector<T> b, vector<T> c, vector<T> d, vector<T> x0, const T& eps, int n, int norm);

Params RelaxationMethodTriDiagonal(vector<T> a, vector<T> b, vector<T> c, vector<T> d, vector<T> x0, const T& omega, const T& eps, int n, int norm);
