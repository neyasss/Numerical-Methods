#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "Header.h"
using namespace std;

// Тестовые функции
double f1(const double x)
{
	return x * x;
}

double f2(const double x)
{
	return 1 / (1 + 25 * x * x);
}

double f3(const double x)
{
	return 1 / (atan(1 + 10 * x * x));
}

double f4(const double x)
{
	double func1 = pow((4 * pow(x, 3) + 2 * pow(x, 2) - 4 * x + 2), sqrt(2));
	double func2 = asin(1 / (5 + x - x * x));
	return func1 + func2 - 5;
}

double f5(const double x)
{
	return 1;
}

double f6(const double x)
{
	return x;
}

double f7(const double x)
{
	return sin(pi * x);
}

double f8(const double x)
{
	double R1 = asin((35 * pow(x, 2) - 30 * x + 9) / 20);
	double R2 = cos((10 * pow(x, 3) + 185 * pow(x, 2) + 340 * x + 103) / (50 * pow(x, 2) + 100 * x + 30));
	return R1 + R2 + 0.5;
}

// Сетки
vector<double> uniformGrid(const double& A, const double& B, int n)
{
	vector<double> grid(n);
	for (int i = 0; i < n; i++)
		grid[i] = A + (B - A) / (n-1) * i;
	return grid;
}

vector<double> chebyshevGrid(const double& A, const double& B, int n)
{
	vector<double> grid(n);
	for (int i = 0; i < n; i++)
		grid[i] = (A+B) / 2 + (B-A) / 2 * cos(pi * (2*i+1) / (2 * (n)));
	return grid;
}

vector<double> uniformGrid2(const double& A, const double& B, double q, int n)
{
	vector<double> grid(n);
	double h = (B - A) / (n - 1);
	for (int i = 0; i < n; i++)
		grid[i] = A + q * h * i;
	return grid;
}

// Полином Лагранжа
double polynomLagrange(const vector<double>& x, const vector<double>& y, int n, double X)
{
	double Ln = 0;
	for (int i = 0; i < n; i++)
	{
		double P = 1;
		for (int j = 0; j < n; j++)
		{
			if (i != j)
				P *= (X - x[j]) / (x[i] - x[j]);
		}
		Ln += y[i] * P;
	}
	return Ln;
}

// Коэффициенты кубического сплайна (с использованием метода прогонки и формул из методички)
void cubicSplineCoeff(const vector<double>& x, const vector<double>& y, vector<double>& a, vector<double>& b,
	vector<double>& c, vector<double>& d, int n)
{
	n = n - 1;
	vector<double> h(n), g(n), alpha(n), beta(n + 1), zn(n + 1);

	for (int i = 0; i < n; i++)
	{
		h[i] = x[i + 1] - x[i];
		g[i] = (y[i + 1] - y[i]) / h[i];
	}

	alpha[0] = 0;
	beta[0] = 0;
	zn[0] = 1;

	for (int i = 1; i < n; i++)
	{
		zn[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * alpha[i - 1];
		alpha[i] = h[i] / zn[i];
		beta[i] = (3 * (g[i] - g[i - 1]) - h[i - 1] * beta[i - 1]) / zn[i];
 	}

	beta[n] = 0;
	zn[n] = 1;
	c[n] = 0;

	for (int i = n - 1; i >= 0; i--)
	{
		c[i] = beta[i] - alpha[i] * c[i + 1];
		b[i] = g[i] - h[i] / 3 * (c[i + 1] + 2 * c[i]);
		d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
		a[i] = y[i];
	}
}

// Кубический сплайн
double cubicSpline(const vector<double>& x, const vector<double>& y, int n, double X)
{
	vector<double> a(n), b(n), c(n), d(n);
	cubicSplineCoeff(x, y, a, b, c, d, n);

	int i = n - 2;
	while (i >= 0 && X < x[i])
		i--;

	double dx = X - x[i];
	return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}

double errNorm(double (*trueFunc)(double), const vector<double>& x, const vector<double>& interp, int n)
{
	double maxErr = 0, err = 0;
	for (int i = 0; i < n; i++)
	{
		err = abs(trueFunc(x[i]) - interp[i]);
		if (err > maxErr)
			maxErr = err;
	}
	return maxErr;
}
