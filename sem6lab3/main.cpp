#include <iostream>
#include "data.h"
#include "scheme.h"

void test2() {
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

int main()
{
	test2();
}
