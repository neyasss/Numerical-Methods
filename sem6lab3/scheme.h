#pragma once

#include <fstream>
#include "data.h"

struct WaveData
{
	std::vector<std::vector<double>> y;
	double tau = 0.;
	double h = 0.;
	size_t steps = 0;
	size_t N = 0;
	double a2 = 0.;

    double operator()(size_t frame, size_t node) const
    {
        return y[frame][node];
    }
};

void initializeWaveData(WaveData& data, const WaveConditions& wc, size_t stringSegments, double timeStep, size_t steps)
{
	data.tau = timeStep;
	data.N = stringSegments;
    data.steps = steps;
	data.y.resize(steps + 1, std::vector<double>(stringSegments + 1));
	data.h = (wc.rightX - wc.leftX) / stringSegments;
	data.a2 = wc.k / wc.rho;
}

void setInitialConditions(WaveData& data, const WaveConditions& wc)
{
	data.y[0][0] = wc.leftU(0.);
	data.y[0][data.N] = wc.rightU(0.);

	data.y[1][0] = wc.leftU(data.tau);
	data.y[1][data.N] = wc.rightU(data.tau);

	for (size_t i = 1; i < data.N; ++i)
	{
		data.y[0][i] = wc.u0(wc.leftX + i * data.h);
		data.y[1][i] = data.y[0][i] + data.tau * wc.v0(wc.leftX + i * data.h);
	}
}

void Scheme(WaveData& data, const WaveConditions& wc, size_t steps)
{
    std::vector<double> fxx(data.N);

    // Вычисление второй производной
    if (wc.isFxxSet())
    {
        for (size_t i = 1; i < data.N; ++i)
        {
            fxx[i] = getFxx(wc, wc.leftX + i * data.h);
        }
    }
    else
    {
        for (size_t i = 1; i < data.N; ++i)
        {
            fxx[i] = (data.y[0][i - 1] - 2.0 * data.y[0][i] + data.y[0][i + 1]) / (data.h * data.h);
        }
    }

    for (size_t i = 1; i < data.N; ++i)
    {
        data.y[1][i] += data.a2 * data.tau * data.tau * 0.5 * fxx[i];
    }

    // Основной цикл процесса
    for (size_t j = 1; j < steps; ++j)
    {
        data.y[j][0] = wc.leftU(data.tau * j);
        data.y[j][data.N] = wc.rightU(data.tau * j);

        for (size_t i = 1; i < data.N; ++i)
        {
            data.y[j + 1][i] =
                (data.a2 / (data.h * data.h)) *
                (data.y[j][i + 1] - 2.0 * data.y[j][i] + data.y[j][i - 1]) *
                (data.tau * data.tau) +
                2.0 * data.y[j][i] - data.y[j - 1][i];
        }
    }
}

void saveToFile(const WaveData& data, const std::string& fileName, size_t steps)
{
    std::ofstream file(fileName, std::ios_base::out);
    for (size_t i = 0; i <= steps; ++i)
    {
        file << i * data.tau;
        for (size_t j = 0; j <= data.N; ++j)
        {
            file << " " << data.y[i][j];
        }
        file << "\n";
    }
    file.flush();
    file.close();
}

double absError(const WaveData& data, std::function<double(double x, double t)> u, size_t steps)
{
    double maxError = 0.0;
    for (size_t i = 1; i <= steps; ++i)
    {
        for (size_t j = 0; j <= data.N; ++j)
        {
            double error = std::fabs(u(data.h * j, data.tau * i) - data.y[i][j]);
            if (error > maxError)
                maxError = error;
        }
    }
    return maxError;
}

std::vector<double> getFrame(const WaveData& data, size_t i)
{
    return data.y[i];
}