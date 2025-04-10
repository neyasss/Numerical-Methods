#pragma once

#include <vector>
#include <functional>

struct WaveConditions
{
	double leftX = 0.;
	double rightX = 1.;
	double k = 1.;
	double rho = 1.;

	std::function<double(double)> leftU = [](double) { return 0; }; // phi(t)
	std::function<double(double)> rightU = [](double) { return 0; }; // psi(t)

	std::function<double(double)> u0 = [](double) { return 0; }; // f(x)
	std::function<double(double)> v0 = [](double) { return 0; }; // g(x)

	std::function<double(double)> fxx = [](double x) { return 0.; }; // f_xx(x)
	bool fxxSet = false;

	bool isFxxSet() const {
		return fxxSet;
	}
};

void setFxx(WaveConditions& wave, const std::function<double(double)>& f) {
	wave.fxx = f;
	wave.fxxSet = true;
}

void unsetFxx(WaveConditions& wave) {
	wave.fxx = [](double x) { return 0.; };
	wave.fxxSet = false;
}

double getFxx(const WaveConditions& wave, double x) {
	return wave.fxx(x);
}
