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

void EulerExplicit(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void EulerImplicit(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void SymmetricScheme(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void RungeKutta2(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

double dif_eval2(vector<double>(*test)(const double& t, const vector<double>&), double& tau, double t_i, double t_edge, vector<double>& k1, vector<double>& k2, vector<double> y_i, vector<double>& y_ipp_1, vector<double>& y_ipp_2);

void AutoStep_RungeKutta2(vector<double>(*test)(const double& t, const vector<double>&), double t0, double T, double h0, vector<double> u0, double eps);

void RungeKutta4(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

double dif_eval4(vector<double>(*test)(const double& t, const vector<double>&), double& tau, double t_i, double t_edge, vector<double>& k1, vector<double>& k2, vector<double>& k3, vector<double>& k4, vector<double> y_i, vector<double>& y_ipp_1, vector<double>& y_ipp_2);

void AutoStep_RungeKutta4(vector<double>(*test)(const double& t, const vector<double>&), double t0, double T, double h0, vector<double> u0, double eps);

void AdamsBashforth(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);

void PredictorCorrector(vector<double>(*test)(const double& t, const vector<double>&), const vector<double> u0, double h, double t0, double tn);
