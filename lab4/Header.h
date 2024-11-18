#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

const double pi = 3.141592653589793;

double polynomLagrange(const vector<double>& x, const vector<double>& y, int n, double X);

void cubicSplineCoeff(const vector<double>& x, const vector<double>& y, vector<double>& a, vector<double>& b,
	vector<double>& c, vector<double>& d, int n);
double cubicSpline(const vector<double>& x, const vector<double>& y, int n, double X);

double errNorm(double (*trueFunc)(double), const vector<double>& x, const vector<double>& interp, int n);

vector<double> uniformGrid(const double& A, const double& B, int n);
vector<double> chebyshevGrid(const double& A, const double& B, int n);
vector<double> uniformGrid2(const double& A, const double& B, double q, int n);

double f1(const double x);
double f2(const double x);
double f3(const double x);
double f4(const double x);
double f5(const double x);
double f6(const double x);
double f7(const double x);
double f8(const double x);

typedef double(*func) (double x);