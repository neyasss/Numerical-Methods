#include "Header.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

bool dynamicStep = false;
const double memoryLimit = 0.5;

const double minTol = 0.0001 * 30;
const double maxTol = 0.001 * 30;

vector<double> test1(const double& t, const vector<double>& u)
{
	int n = u.size();
	vector<double> du(n);

	du[0] = 2 * u[0] + u[1] * u[1] - 1;
	du[1] = 6 * u[0] - u[1] * u[1] + 1;

	return du;
}

vector<double> test2(const double& t, const vector<double>& u)
{
	int n = u.size();
	vector<double> du(n);

	du[0] = 1 - u[0] * u[0] - u[1] * u[1];
	du[1] = 2 * u[0];

	return du;
}

vector<double> test3(const double& t, const vector<double>& u)
{
	int n = u.size();
	vector<double> du(n);
	double sigma = 10, r = 28, b = 8 / 3;

	du[0] = sigma * (u[1] - u[0]);
	du[1] = u[0] * (r - u[2]) - u[1];
	du[2] = u[0] * u[1] - b * u[2];

	return du;
}

vector<double> testM(const double& t, const vector<double>& u)
{
	int n = u.size();
	vector<double> du(n);
	double k = 20, m = 0.3;

	du[0] = u[1];
	du[1] = -1 * k / m * u[0];

	return du;
}

vector<double> operator*(const double& num, const vector<double>& vec)
{
	vector<double> res(vec.size());
	for (int i = 0; i < vec.size(); i++)
		res[i] = num * vec[i];
	return res;
}

vector<double> operator+(const vector<double>& vec1, const vector<double>& vec2)
{
	vector<double> res = vec1;
	for (int i = 0; i < vec1.size(); i++)
		res[i] += vec2[i];
	return res;
}

vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2)
{
	vector<double> res = vec1;
	for (int i = 0; i < vec1.size(); i++)
		res[i] -= vec2[i];
	return res;
}

vector<double> operator/(const vector<double>& vec1, const vector<double>& vec2)
{
	vector<double> res = vec1;
	for (int i = 0; i < vec1.size(); i++)
		res[i] /= vec2[i];
	return res;
}

double scmult(vector<double> v1, vector<double> v2)
{
	double result = 0;
	for (auto v = v1.begin(), p = v2.begin(); v != v1.end() && p != v2.end(); v++, p++)
	{
		result += *v * *p;
	}
	return result;
}

void EulerExplicit(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("EulerExp.txt");

	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0;
	for (int i = 1; i <= n; i++)
	{
		y_prev = y_next;
		y_next = y_prev + h * test(t0 + i * h, y_prev);

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}
	result.close();
}

vector<double> NewtonMethod1(vector<double>(*test)(const double&, const vector<double>&), const double& t, vector<double> u_old, double h)
{
	int n = u_old.size();
	double eps = h * 1e-6;
	vector<double> u_new(n);
	u_new = u_old;

	auto eq = [u_old, h, t, test](const vector<double>& u_new)
		{
			return u_new - u_old - h * test(t + h, u_new);
		};

	int iter = 0;

	do
	{
		iter++;
		vector<double> grad(n);
		for (int i = 0; i < n; i++)
		{
			vector<double> u_plus = u_new;
			vector<double> u_minus = u_new;

			u_plus[i] += h;
			u_minus[i] -= h;
			grad[i] = (eq(u_plus)[i] - eq(u_minus)[i]) / (2 * h);
		}

		u_old = u_new;
		u_new = u_new - eq(u_new) / grad;

	} while (iter < 100 && sqrt((u_new - u_old)[0] * (u_new - u_old)[0] + (u_new - u_old)[1] * (u_new - u_old)[1]) > eps);
	return u_new;
}

void EulerImplicit(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("EulerImp.txt");

	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0;
	for (int i = 1; i <= n; i++)
	{
		y_prev = y_next;
		y_next = NewtonMethod1(test, t0 + i * h, y_prev, h);

		y_next = y_prev + h * test(t0 + (i + 1) * h, y_next);

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}
	result.close();
}

vector<double> NewtonMethod2(vector<double>(*test)(const double&, const vector<double>&), const double& t, vector<double> u_old, double h)
{
	int n = u_old.size();
	double eps = h * 1e-6;
	vector<double> u_new(n);
	u_new = u_old;

	auto eq = [u_old, h, t, test](const vector<double>& u_new)
		{
			return u_new - u_old - (h / 2) * (test(t + h, u_new) + test(t, u_old));
		};

	int iter = 0;

	do
	{
		iter++;
		vector<double> grad(n);
		for (int i = 0; i < n; i++)
		{
			vector<double> u_plus = u_new;
			vector<double> u_minus = u_new;

			u_plus[i] += h;
			u_minus[i] -= h;
			grad[i] = (eq(u_plus)[i] - eq(u_minus)[i]) / (2 * h);
		}

		u_old = u_new;
		u_new = u_new - eq(u_new) / grad;

	} while (iter < 100 && sqrt((u_new - u_old)[0] + (u_new - u_old)[1]) > eps);
	return u_new;
}


void SymmetricScheme(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("SymScheme.txt");

	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0;
	for (int i = 1; i <= n; i++)
	{
		y_prev = y_next;
		y_next = NewtonMethod2(test, t0 + i * h, y_prev, h);

		y_next = y_prev + (h / 2) * (test(t0 + (i + 1) * h, y_next) + test(t0 + i * h, y_prev));

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}
	result.close();
}


void RungeKutta2(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("RungeKutta2.txt");
	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0;
	vector<double> k1(u0.size()), k2(u0.size());
	size_t maxSize = (size_t)round(0.5 * 1'073'741'824. / ((32. + 8. * u0.size())));
	
	for (int i = 0; i <= n; i++)
	{
		

		y_prev = y_next;

		k1 = test(t0 + i * h, y_prev);
		k2 = test(t0 + (i + 1) * h, y_prev + h * k1);

		y_next = y_prev + h / 2 * (k1 + k2);

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}
	result.close();
}

double dif_eval2(vector<double>(*test)(const double& t, const vector<double>&), double& tau, double t_i, double t_edge, vector <double>& k1, vector<double>& k2, vector <double> y_i, vector<double>& y_ipp_1, vector <double> & y_ipp_2)
{
	// Вычисляем компоненты k при половинном шаге 
	//первый половинный шаг
	double tau_half = tau / 2;
	double temp_t = t_i;
	vector<double> y_ipp_1_0 = y_i;
	while (t_edge - temp_t > 0)
	{
		temp_t += tau_half;
		k1 = test(temp_t - tau_half, y_ipp_1_0);
		k2 = test(temp_t + tau_half, y_ipp_1_0 + tau_half * k1);
		y_ipp_1_0 = y_ipp_1_0 + (tau_half / 2) * (k1 + k2);
	}
	//cout << t_edge << "v/s" << temp_t << endl;
	//cout << "----t_edge = " << temp_t;
	y_ipp_1 = y_ipp_1_0;
	// Вычисляем компоненты k при целом шаге
	temp_t = t_i;
	y_ipp_1_0 = y_i;
	while (t_edge - temp_t > 0)
	{
		temp_t += tau;
		k1 = test(temp_t - tau, y_ipp_1_0);
		k2 = test(temp_t + tau, y_ipp_1_0 + tau * k1);
		y_ipp_1_0 = y_ipp_1_0 + (tau / 2) * (k1 + k2);
	}
	y_ipp_2 = y_ipp_1_0;
	//проверяем апостериорную погрешность
	int p = 2;
	double denom = pow(2, p) - 1;
	double difference = sqrt(scmult(y_ipp_2 - y_ipp_1, y_ipp_2 - y_ipp_1)) / denom;
	//cout << "counted diff" << endl;
	return difference;
}

void AutoStep_RungeKutta2(vector<double>(*test)(const double& t, const vector<double>&), double t0, double T, double tau0, vector<double> u0, double eps = 1e-6) {
	ofstream result("RungeKutta2Auto.txt");
	if (result.is_open())
	{
		double t_i = t0;
		vector<double> y_i = u0;
		vector<double> y_ipp_1_0 = u0, y_ipp_1 = u0, y_ipp_2 = u0;
		vector<double> k1 = u0, k2 = u0;

		double tau = tau0;
		result << t_i << " ";
		for (int i = 0; i < y_i.size(); ++i)
			result << y_i[i] << " ";
		result << " " << endl;

		while (T - t_i > 0) {

			double edge = t_i + tau;
			double difference = dif_eval2(*test, tau, t_i, edge, k1, k2, y_i, y_ipp_1, y_ipp_2);
			while (difference >= eps)
			{
				tau /= 2;
				difference = dif_eval2(*test, tau, t_i, edge, k1, k2, y_i, y_ipp_1, y_ipp_2);
			}
			if (difference < eps)
			{
				y_i = y_ipp_2;
				t_i += tau;

				for (int i = 0; i < y_i.size(); ++i)
					result << y_i[i] << " ";
				result << " " << endl;

				if (difference <= eps * 1e-3)
				{
					tau *= 2;
				}
				continue;
			}
		}
		result.close();
	}
}

void RungeKutta4(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("RungeKutta4.txt");

	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0;
	vector<double> k1(u0.size()), k2(u0.size()), k3(u0.size()), k4(u0.size()), K(u0.size());

	for (int i = 0; i <= n; i++)
	{
		y_prev = y_next;

		k1 = test(t0 + i * h, y_prev);
		k2 = test(t0 + i * h + h / 2, y_prev + h / 2 * k1);
		k3 = test(t0 + i * h + h / 2, y_prev + h / 2 * k2);
		k4 = test(t0 + i * h + h, y_prev + h * k3);
		K = 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

		y_next = y_prev + h * K;

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}
	result.close();
}


double dif_eval4(vector<double>(*test)(const double& t, const vector<double>&), double& tau, double t_i, double t_edge, vector<double>& k1, vector<double>& k2, vector<double>& k3, vector<double>& k4, vector<double> y_i, vector<double>& y_ipp_1, vector<double>& y_ipp_2)
{
	// Вычисляем компоненты k при половинном шаге 
	//первый половинный шаг
	double tau_half = tau / 2;
	double temp_t = t_i;
	vector<double> y_ipp_1_0 = y_i;
	while (t_edge - temp_t > 0)
	{
		temp_t += tau_half;
		k1 = test(temp_t - tau_half, y_ipp_1_0);
		k2 = test(temp_t - tau_half / 2, y_ipp_1_0 + 0.5 * tau_half * k1);
		k3 = test(temp_t - tau_half / 2, y_ipp_1_0 + 0.5 * tau_half * k2);
		k4 = test(temp_t, y_ipp_1_0 + tau_half * k3);
		y_ipp_1_0 = y_ipp_1_0 + (tau_half / 6) * (k1 + 2 * k2 + 2 * k3 + k4);

	}
	//cout << t_edge << "v/s" << temp_t << endl;
	//cout << "----t_edge = " << temp_t;
	y_ipp_1 = y_ipp_1_0;
	// Вычисляем компоненты k при целом шаге
	temp_t = t_i;
	y_ipp_1_0 = y_i;
	while (t_edge - temp_t > 0)
	{
		temp_t += tau;
		k1 = test(temp_t - tau, y_ipp_1_0);
		k2 = test(temp_t - tau / 2, y_ipp_1_0 + 0.5 * tau * k1);
		k3 = test(temp_t - tau / 2, y_ipp_1_0 + 0.5 * tau * k2);
		k4 = test(temp_t, y_ipp_1_0 + tau * k3);
		y_ipp_1_0 = y_ipp_1_0 + (tau / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
	}
	y_ipp_2 = y_ipp_1_0;
	//проверяем апостериорную погрешность
	int p = 4;
	double denom = pow(2, p) - 1;
	double difference = sqrt(scmult(y_ipp_2 - y_ipp_1, y_ipp_2 - y_ipp_1)) / denom;
	//cout << "counted diff" << endl;
	return difference;
}

void AutoStep_RungeKutta4(vector<double>(*test)(const double& t, const vector<double>&), double t0, double T, double tau0, vector<double> u0, double eps = 1e-6)
{
	ofstream result("RungeKutta4Auto.txt");
	if (result.is_open())
	{
		double t_i = t0;
		vector<double> y_i = u0;
		int ind = 0;
		vector<double> y_ipp_1 = u0, y_ipp_2 = u0;
		vector<double> k1 = u0, k2 = u0, k3 = u0, k4 = u0, K = u0;
		double tau = tau0;
		result << t_i << " ";
		for (int i = 0; i < y_i.size(); ++i)
			result << y_i[i] << " ";
		result << " " << endl;
		while (T - t_i > 0) {

			double edge = t_i + tau;
			double difference = dif_eval4(*test, tau, t_i, edge, k1, k2, k3, k4, y_i, y_ipp_1, y_ipp_2);
			while (difference >= eps)
			{
				tau /= 2;
				difference = dif_eval4(*test, tau, t_i, edge, k1, k2, k3, k4, y_i, y_ipp_1, y_ipp_2);
			}
			if (difference < eps)
			{
				y_i = y_ipp_2;
				t_i += tau;
				result << t_i << " ";
				for (int i = 0; i < y_i.size(); ++i)
					result << y_i[i] << " ";
				result << " " << endl;
				if (difference <= eps * 1e-3)
				{
					tau *= 2;
				}
				continue;
			}

		}
		result.close();
	}
}

void AdamsBashforth(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("Adams.txt");
	int RANG = u0.size();
	int n = (tn - t0) / h;

	vector<double> y_next = u0, y_prev = u0, prediction(u0.size()), correction(u0.size());
	vector<vector<double>> ode(4, vector<double>(u0.size()));
	vector<double> k1(u0.size()), k2(u0.size()), k3(u0.size()), k4(u0.size()), K(u0.size());

	for (int i = 0; i < 4; i++) {

		ode[i] = test(t0 + i * h, y_prev);

		k1 = test(t0 + i * h, y_prev);
		k2 = test(t0 + i * h + h / 2, y_prev + h / 2 * k1);
		k3 = test(t0 + i * h + h / 2, y_prev + h / 2 * k2);
		k4 = test(t0 + i * h + h, y_prev + h * k3);
		;
		K = 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

		y_prev = y_next;
		y_next = y_prev + h * K;

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}

	y_prev = y_next;

	for (int i = 3; i < n; i++) {

		y_next = y_prev + (h / 24.) * (55. * ode[3] + 59. * ode[2] + 37. * ode[1] - 9. * ode[0]);
		result << i * h << " ";
		for (int elem = 0; elem < u0.size(); elem++) {
			result << y_next[elem] << " ";
		}
		result << endl;
		for (int j = 0; j < ode.size(); j++)
		{
			ode[(ode.size() + j - 1) % ode.size()] = ode[j];
		}
		ode[3] = test(t0 + (i + 1) * h, y_next);
	}
	result.close();
}


void PredictorCorrector(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn)
{
	ofstream result("PredCor.txt");

	int n = (tn - t0) / h;
	vector<double> y_next = u0, y_prev = u0, prediction(u0.size()), correction(u0.size());
	vector<vector<double>> ode(4, vector<double>(u0.size()));
	vector<double> k1(u0.size()), k2(u0.size()), k3(u0.size()), k4(u0.size()), K(u0.size());

	for (int i = 0; i < 4; i++)
	{
		ode[i] = test(t0 + i * h, y_prev);

		k1 = test(t0 + i * h, y_prev);
		k2 = test(t0 + i * h + h / 2, y_prev + h / 2 * k1);
		k3 = test(t0 + i * h + h / 2, y_prev + h / 2 * k2);
		k4 = test(t0 + i * h + h, y_prev + h * k3);
		K = 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

		y_prev = y_next;
		y_next = y_prev + h * K;

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;
	}

	for (int i = 4; i <= n; i++)
	{
		prediction = y_prev + (h / 24) * (55 * ode[3] - 59 * ode[2] + 37 * ode[1] - 9 * ode[0]);
		correction = (h / 24) * (9 * test(t0 + (i + 1) * h, prediction) + 19 * ode[3] - 5 * ode[2] + ode[1]);

		y_prev = y_next;
		y_next = y_prev + correction;

		result << t0 + i * h << " ";
		for (int j = 0; j < u0.size(); j++)
			result << y_next[j] << " ";
		result << endl;

		for (int j = 0; j < ode.size(); j++)
		{
			ode[(ode.size() + j - 1) % ode.size()] = ode[j];
		}
		ode[3] = test(t0 + (i + 1) * h, y_next);
	}
	result.close();
}
