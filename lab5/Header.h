#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <list>
#include <cassert>

using namespace std;
#define numtype double

void errorOrder(const list<double>& xList, double x);
numtype normInf(const std::vector<numtype>&);
double sqr(vector<double> vec1, vector<double> vec2);

std::vector<numtype> operator+(const std::vector<numtype>&, const std::vector<numtype>&);
std::vector<numtype> operator-(const std::vector<numtype>&, const std::vector<numtype>&);

vector<double> localize(const function<double(double)>& f, double left, double right, size_t div);
double bisection(const function<double(double)>& f, double a, double b, double eps);
double newton(
	const std::function<double(double)>& func,
	double left,              //левая граница отрезка
	double right,             //правая граница
	double eps = 1e-6,        //точность
	bool protection = true,   //включение защит выполнения
	bool analytical = false,  //включение аналитической производной
	std::function<double(double)> fx = {}  //аналитическая производная
);
std::vector<double> newton_2(
	const std::function<double(double, double)>& func1,  //первая функция
	const std::function<double(double, double)>& func2,  //вторая функция
	double eps = 1e-6,          //точность
	double startX = 0.,         //первая координата начальной точки
	double startY = 0.          //вторая координата начальной точки
);
