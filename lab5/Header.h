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
numtype normInf(const vector<numtype>&);
double sqr(vector<double> vec1, vector<double> vec2);

vector<numtype> operator+(const vector<numtype>&, const vector<numtype>&);
vector<numtype> operator-(const vector<numtype>&, const vector<numtype>&);
ostream& operator<<(ostream&, const vector<numtype>&);

vector<double> localize(const function<double(double)>& f, double left, double right, size_t div);
double bisection(const function<double(double)>& f, double a, double b, double eps);
double newton(
	const function<double(double)>& func,
	double left,              //левая граница отрезка
	double right,             //правая граница
	double eps = 1e-6,        //точность
	bool protection = true,   //включение защит выполнения
	bool analytical = false,  //включение аналитической производной
	function<double(double)> fx = {}  //аналитическая производная
);
vector<double> newton_2(
	const function<double(double, double)>& func1,  //первая функция
	const function<double(double, double)>& func2,  //вторая функция
	double eps = 1e-6,          //точность
	double startX = 0.,         //первая координата начальной точки
	double startY = 0.          //вторая координата начальной точки
);

vector<double> newtonDiagram(
	const function<double(double, double)>& func1,  //первая функция
	const function<double(double, double)>& func2,  //вторая функция
	size_t N = 100,                                      //разрешение диаграммы
	const string& fileName = "Grid.txt"             //файл, куда диаграмма сохранится
);#pragma once
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
numtype normInf(const vector<numtype>&);
double sqr(vector<double> vec1, vector<double> vec2);

vector<numtype> operator+(const vector<numtype>&, const vector<numtype>&);
vector<numtype> operator-(const vector<numtype>&, const vector<numtype>&);
ostream& operator<<(ostream&, const vector<numtype>&);

vector<double> localize(const function<double(double)>& f, double left, double right, size_t div);
double bisection(const function<double(double)>& f, double a, double b, double eps);
double newton(
	const function<double(double)>& func,
	double left,              //левая граница отрезка
	double right,             //правая граница
	double eps = 1e-6,        //точность
	bool protection = true,   //включение защит выполнения
	bool analytical = false,  //включение аналитической производной
	function<double(double)> fx = {}  //аналитическая производная
);
vector<double> newton_2(
	const function<double(double, double)>& func1,  //первая функция
	const function<double(double, double)>& func2,  //вторая функция
	double eps = 1e-6,          //точность
	double startX = 0.,         //первая координата начальной точки
	double startY = 0.          //вторая координата начальной точки
);

vector<double> newtonDiagram(
	const function<double(double, double)>& func1,  //первая функция
	const function<double(double, double)>& func2,  //вторая функция
	size_t N = 100,                                      //разрешение диаграммы
	const string& fileName = "Grid.txt"             //файл, куда диаграмма сохранится
);
