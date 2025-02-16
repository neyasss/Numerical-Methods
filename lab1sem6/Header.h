#pragma once
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

vector<double> test1(const double& t, const vector<double>& u);
vector<double> test2(const double& t, const vector<double>& u);
vector<double> test3(const double& t, const vector<double>& u);
vector<double> testM(const double& t, const vector<double>& u);

void EulerExplicit(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void EulerImplicit(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void SymmetricScheme(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void RungeKutta2(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void RungeKutta4(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void AdamsBashforth(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void PredictorCorrector(vector<double> (*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);