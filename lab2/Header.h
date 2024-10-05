#pragma once

#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // ������� �������� 
#define T double                            // ���������� ��������

// ��1

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b); // ������ ���� �� �����

void printSLAE(const vector<vector<T>>& A, const vector<T>& b, int n); // ����� ���� �� �����
void printMatrix(const vector<vector<T>>& A, int n); // ����� ������� �� �����

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B, int n); // ��������� ������
vector<vector<T>> Transpose(const vector<vector<T>>& A, int n); // ����������������

T ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, int n, int norm); // ����� ������� �������

vector<vector<T>> InvLU(const vector<vector<T>>& A, int n); // ���������� �������� ������� � ������� LU-����������

T vectorNorm1(const vector<T>& b, int n); // ��������� �������������� �����
T vectorNormInf(const vector<T>& b, int n); // ��������� ���������� �����

T matrixNorm1(const vector<vector<T>>& A, int n); // ��������� �������������� �����
T matrixNormInf(const vector<vector<T>>& A, int n); // ��������� ���������� �����

// ����� ��������������� ��� ��������� ��������� ����
T cond1(const vector<vector<T>>& A, int n);
T condInf(const vector<vector<T>>& A, int n);

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// 
// ��2

struct Params
{
	vector<vector<T>> C;
	vector<T> y;
	vector<T> x; // �������
	int iterCount; // ���������� ��������
	T normC1, normCInf;
};

void LDU(vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U, int n);

Params SimpleIterationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);

Params JacobiMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& eps, int n, int norm);

Params SeidelMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);

Params RelaxationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm);