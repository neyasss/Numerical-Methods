#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // обычная точность 
#define T double                            // повышенная точность

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b); // чтение СЛАУ из файла

void printSLAE(const vector<vector<T>>& A, const vector<T>& b, int n); // вывод СЛАУ на экран
void printMatrix(const vector<vector<T>>& A, int n); // вывод матрицы на экран

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B, int n); // умножение матриц
vector<vector<T>> Transpose(const vector<vector<T>>& A, int n); // транспонирование (для QR-разложения)

vector<T> GaussianMethod(vector<vector<T>>& A, vector<T>& b, int n); // Метод Гаусса с частичным выбором главного элемента (по столбцу)

vector<T> QRMethod(vector<vector<T>>& A, vector<T>& b, int n); // Решение методом QR-разложения

void ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, int n); // норма вектора невязки (обе нормы)

vector<vector<T>> InvLU(const vector<vector<T>>& A, int n); // нахождение обратной матрицы с помощью LU-разложения

T vectorNorm1(const vector<T>& b, int n); // векторная октаэдрическая норма
T vectorNormInf(const vector<T>& b, int n); // векторная кубическая норма

T matrixNorm1(const vector<vector<T>>& A, int n); // матричная октаэдрическая норма
T matrixNormInf(const vector<vector<T>>& A, int n); // матричная кубическая норма

// Число обусловленности для различных матричных норм
T cond1(const vector<vector<T>>& A, int n);
T condInf(const vector<vector<T>>& A, int n);

void condEstimation(vector<vector<T>>& A, vector<T>& b, const vector<T>& disturb, int n); // вектор возмущения как аргумент, вычисляет 1 раз
void condEstimationLower(vector<vector<T>>& A, vector<T>& b, int n, int disturbCount = 5); // решение несколько раз и оценка
