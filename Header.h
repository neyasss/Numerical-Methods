#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // ������� �������� 
#define T double                            // ���������� ��������

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b); // ������ ���� �� �����

void printSLAE(const vector<vector<T>>& A, const vector<T>& b); // ����� ���� �� �����
void printMatrix(const vector<vector<T>>& A); // ����� ������� �� �����

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B); // ��������� ������
vector<vector<T>> Transpose(const vector<vector<T>>& A); // ���������������� (��� QR-����������)

vector<T> GaussianMethod(vector<vector<T>>& A, vector<T>& b); // ����� ������ � ��������� ������� �������� �������� (�� �������)

vector<T> QRMethod(vector<vector<T>>& A, vector<T>& b); // ������� ������� QR-����������

void ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x); // ����� ������� ������� (��� �����)

vector<vector<T>> InvLU(const vector<vector<T>>& A); // ���������� �������� ������� � ������� LU-����������

T vectorNorm1(const vector<T>& b); // ��������� �������������� �����
T vectorNormInf(const vector<T>& b); // ��������� ���������� �����

T matrixNorm1(const vector<vector<T>>& A); // ��������� �������������� �����
T matrixNormInf(const vector<vector<T>>& A); // ��������� ���������� �����

// ����� ��������������� ��� ��������� ��������� ����
T cond1(const vector<vector<T>>& A);
T condInf(const vector<vector<T>>& A);

void condEstimation(vector<vector<T>>& A, vector<T>& b, const vector<T>& disturb); // ������ ���������� ��� ��������, ��������� 1 ���
void condEstimationLower(vector<vector<T>>& A, vector<T>& b, int disturbCount = 5); // ������� ��������� ��� � ������