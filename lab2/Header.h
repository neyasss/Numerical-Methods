#pragma once

#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // îáû÷íàÿ òî÷íîñòü 
#define T double                            // ïîâûøåííàÿ òî÷íîñòü

// ëð1

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b); // ÷òåíèå ÑËÀÓ èç ôàéëà

void printSLAE(const vector<vector<T>>& A, const vector<T>& b, int n); // âûâîä ÑËÀÓ íà ýêðàí
void printMatrix(const vector<vector<T>>& A, int n); // âûâîä ìàòðèöû íà ýêðàí

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B, int n); // óìíîæåíèå ìàòðèö
vector<vector<T>> Transpose(const vector<vector<T>>& A, int n); // òðàíñïîíèðîâàíèå

T ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, int n, int norm); // íîðìà âåêòîðà íåâÿçêè

vector<vector<T>> InvLU(const vector<vector<T>>& A, int n); // íàõîæäåíèå îáðàòíîé ìàòðèöû ñ ïîìîùüþ LU-ðàçëîæåíèÿ

T vectorNorm1(const vector<T>& b, int n); // âåêòîðíàÿ îêòàýäðè÷åñêàÿ íîðìà
T vectorNormInf(const vector<T>& b, int n); // âåêòîðíàÿ êóáè÷åñêàÿ íîðìà

T matrixNorm1(const vector<vector<T>>& A, int n); // ìàòðè÷íàÿ îêòàýäðè÷åñêàÿ íîðìà
T matrixNormInf(const vector<vector<T>>& A, int n); // ìàòðè÷íàÿ êóáè÷åñêàÿ íîðìà

// ×èñëî îáóñëîâëåííîñòè äëÿ ðàçëè÷íûõ ìàòðè÷íûõ íîðì
T cond1(const vector<vector<T>>& A, int n);
T condInf(const vector<vector<T>>& A, int n);

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// 
// ëð2

struct Params
{
	vector<vector<T>> C;
	vector<T> y;
	vector<T> x; // ðåøåíèå
	int iterCount; // êîëè÷åñòâî èòåðàöèé
	T normC1, normCInf;
};

void LDU(vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U, int n);

Params SimpleIterationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);

Params JacobiMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& eps, int n, int norm);

Params SeidelMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);

Params RelaxationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);
