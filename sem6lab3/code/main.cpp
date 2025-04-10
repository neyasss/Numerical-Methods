#include <iostream>
#include "data.h"
#include "scheme.h"
#include <cmath>

void test1()
{
	const double pi = 3.141'592'653'589'793'238;
	double leftX = 0., rightX = 1., density = 1., tension = 1.;
	WaveConditions wc1;
	wc1.leftX = leftX;
	wc1.rightX = rightX;
	wc1.k = tension;
	wc1.rho = density;
	wc1.u0 = [=](double x) { return sin(pi * x); };
	std::function<double(double)> fxx = [=](double x) { return -pi * pi * sin(pi * x); };
	setFxx(wc1, fxx);
	auto solution = [=](double x, double t) { return cos(pi * t) * sin(pi * x); };

	size_t segments = 10, frames = 100;
	double tau = 0.01;
	// std::cout << "courant: " << sqrt(tension / density) * tau / (rightX - leftX) * segments << "\n";
	WaveData wd1;
	
	initializeWaveData(wd1, wc1, segments, tau, frames);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "test1.txt", wd1.steps);
}

void test2()
{
	const double pi = 3.141'592'653'589'793'238;
	double leftX = 0., rightX = 1., density = 1., tension = 1.;
	WaveConditions wc1;
	wc1.leftX = leftX;
	wc1.rightX = rightX;
	wc1.k = tension;
	wc1.rho = density;
	wc1.u0 = [](double x) { return x * (1. - x); };

	size_t segments = 50, frames = 200;
	double tau = 0.02;
	std::cout << "courant: " << sqrt(tension / density) * tau / (rightX - leftX) * segments << "\n";
	WaveData wd1;

	initializeWaveData(wd1, wc1, segments, tau, frames);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "test2.txt", wd1.steps);

	
	auto cube = [](double x) { return x * x * x; };

	double maxError = 0, eps = 1e-4, h = (rightX - leftX) / segments;
	size_t k = ceil(0.5 * (sqrt(2. / (pi * pi * eps)) - 1.));
	std::cout << "k = " << k << "\n";

	for (size_t j = 1; j <= frames; ++j)
		for (size_t i = 1; i < segments; ++i) {
			double px = pi * h * i, pt = pi * tau * j;
			double u = 0.;
			for (size_t n = 0; n <= k; ++n) {
				u += sin((2 * n + 1.) * px) * cos((2 * n + 1.) * pt) / cube(2 * n + 1.);
			}
			u *= 8 / cube(pi);
			double error = fabs(wd1(j, i) - u);
			if (error > maxError)
				maxError = error;
		}
	std::cout << "eps = " << eps << "\nerr = " << maxError << "\n";
}

void dop1()
{
	double leftX = -2., rightX = 2., density = 1., tension = 1.;
	WaveConditions wc1;
	wc1.leftX = leftX;
	wc1.rightX = rightX;
	wc1.k = tension;
	wc1.rho = density;
	wc1.u0 = [](double x) {
		if (x >= -1. / 3 && x <= 1. / 3)
			return 1;
		else
			return 0;
	};

	// Числа Куранта 0.1, 0.5, 0.75, 1.0

	// численное вычисление производной
	WaveData wd1;
	
	initializeWaveData(wd1, wc1, 40, 0.01, 500);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "courant0.1_n.txt", wd1.steps);
	
	/*
	initializeWaveData(wd1, wc1, 200, 0.01, 500);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "courant0.5_n.txt", wd1.steps);

	initializeWaveData(wd1, wc1, 300, 0.01, 500);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "courant0.75_n.txt", wd1.steps);
	
	initializeWaveData(wd1, wc1, 400, 0.01, 500);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "courant1_n.txt", wd1.steps);
	*/

	// аналитическое вычисление производной

	std::function<double(double)> fxx = [](double x) { return 0.; };
	setFxx(wc1, fxx);
	initializeWaveData(wd1, wc1, 400, 0.01, 500);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "courant1_a.txt", wd1.steps);
}

void testvar9()
{
	double leftX = 0., rightX = 1., density = 1., tension = 1.;
	WaveConditions wc1;
	wc1.leftX = leftX;
	wc1.rightX = rightX;
	wc1.k = tension;
	wc1.rho = density;
	wc1.leftU = [](double t) { return t * t; };
	wc1.rightU = [](double t) { return 1.5; };
	wc1.u0 = [=](double x) { return x * (2. * x - 0.5); };
	wc1.v0 = [=](double x) { return cos(2. * x); };
	// std::function<double(double)> fxx = [=](double x) { return -pi * pi * sin(pi * x); };
	// setFxx(wc1, fxx);

	size_t segments = 20, frames = 100;
	double tau = 0.01;
	std::cout << "courant: " << sqrt(tension / density) * tau / (rightX - leftX) * segments << "\n";
	WaveData wd1;

	initializeWaveData(wd1, wc1, segments, tau, frames);
	setInitialConditions(wd1, wc1);
	Scheme(wd1, wc1, wd1.steps);
	saveToFile(wd1, "testvar9.txt", wd1.steps);
}

int main()
{
	// test1();
	// test2();
	// dop1();
	testvar9();
}
