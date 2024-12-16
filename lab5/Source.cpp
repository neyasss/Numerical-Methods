#include "Header.h"

//Порядок ошибки
void errorOrder(const list<double>& xList, double x) {
	if (xList.size() < 3)
		return;
	auto x2 = xList.begin(); //x_{k+2}
	auto x0 = x2++;          //x_k
	auto x1 = x2++;          //x_{k+1}
	for (; x2 != --xList.end(); ++x0, ++x1, ++x2) {
		cout << "Порядок ошибки: " << log(abs((*x2 - x) / (*x1 - x))) / log(abs((*x1 - x) / (*x0 - x))) << "\n";
	}
	cout << endl;
}

numtype normInf(const vector<numtype>& vec) {
	numtype max = 0;
	for (size_t i = 0; i < vec.size(); ++i)
		if (fabs(vec[i]) > max)
			max = fabs(vec[i]);
	return max;
}

/* Функция, вычисляющая норму разности векторов */
double sqr(vector<double> vec1, vector<double> vec2) {
	int m = vec1.size();
	double sum = 0;
	for (int i = 0; i < m; i++) {
		sum = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sum;
}

vector<numtype> operator+(const vector<numtype>& vec1, const vector<numtype>& vec2) {
	assert(vec1.size() == vec2.size() && "Vector size mismatch!");
	vector<numtype> vec3(vec1);
	for (size_t i = 0; i < vec3.size(); ++i) {
		vec3[i] += vec2[i];
	}
	return vec3;
}

vector<numtype> operator-(const vector<numtype>& vec1, const vector<numtype>& vec2) {
	assert(vec1.size() == vec2.size() && "Vector size mismatch!");
	vector<numtype> vec3(vec1);
	for (size_t i = 0; i < vec3.size(); ++i) {
		vec3[i] -= vec2[i];
	}
	return vec3;
}

ostream& operator<<(ostream& out, const vector<numtype>& vec) {
	if (vec.size() > 0) {
		out << vec[0];
		for (size_t i = 1; i < vec.size(); ++i)
			out << " " << vec[i];
	}
	else out << "null";
	return out;
}


vector<double> localize(const function<double(double)>&f, double left, double right, size_t div) {
		cout << "Локализация корней:" << endl << "новый промежуток: ";
		list<double> sections;  //список начал и концов отрезка
		double prev, next;
		double step = (right - left) / (double)div;
		prev = left;
		for (size_t i = 0; i < div; i++) {
			next = prev + step;
			if (f(prev) * f(next) < 0) {  //проверка на разные знаки
				sections.push_back(prev);
				sections.push_back(next);
			}
			prev = next;
		}
		if (sections.size() == 0)
			sections = { left, right };

		auto sec = sections.begin();      //вывод найденных отрезков
		for (int i = 0; i < sections.size() / 2; i++) {
			cout << *(sec++) << " -- ";
			cout << *(sec++) << "\n";
		}
		cout << endl;
		vector<double> result(sections.size());
		copy(sections.begin(), sections.end(), result.begin());
		return result;
	};

	double bisection(const function<double(double)>&f, double a, double b, double eps) {
		cout << "Бисекция: \n";
		double fa = f(a), fb = f(b);
		double x = (a + b) * 0.5;  //середина отрезка
		list<double> xList({ x });
		size_t iter = 0;

		//пока половина отрезка неопределённости больше точности
		while ((b - a) * 0.5 > eps) {
			/*cout << "Итерации: " << iter \
				<< "\n a = " << a << ",      f(a) = " << fa \
				<< "\n b = " << b << ",      f(b) = " << fb << "\n"; */
			++iter;
			double fx = f(x);
			if (fa * fx < 0) {
				b = x;
				fb = fx;
			}
			else {
				a = x;
				fa = fx;
			}
			x = (a + b) * 0.5;
			xList.push_back(x);
		}
		cout << "\nВсего итераций: " << iter << "\nx = " << x << "\n\n";
		//errorOrder(xList, x);
		return x;
	};


double newton(const function<double(double)>& f, double a, double b, double eps, bool protection, bool analytical, function<double(double)> fx) {
	cout << "Метод Ньютона: \n";
	double loopProtection = false, rangeProtection = false;
	if (protection) {
		loopProtection = true;
		rangeProtection = true;
	}
	double fa = f(a), fb = f(b);
	double xNext, xPrev;
	if (a == b) {   //чтобы вручную указать начальную точку,
		xNext = a;  //можно указать одинаковые концы
		rangeProtection = false;
	}
	else
		xNext = (fa * b - fb * a) / (fa - fb);  //метод хорд
	//xNext = 8.;  //или можно здесь, чтобы работал внутри отрезка

	double h = 1e-6; //шаг для вычиcления разностной производной
	list<double> xList({ xNext });

	if (!analytical) {
		fx = [&](double x) { return (f(x + h) - f(x - h)) / (2. * h); }; //центральная разностная производная
		//fx = [&](double x) { return (f(x + h) - f(x)) / h; }; //правая
		//fx = [&](double x) { return (f(x) - f(x - h)) / h; }; //левая
	}
	size_t iter = 0;

	function<void(double)> protectRange; //защита от выхода за диапазон
	if (rangeProtection)
		protectRange = [&](double frac) {
		for (double alpha = 0.5; xNext < a || xNext > b; alpha *= 0.5) {
			cout << "Защита от выхода за диапазон:  x = " << xNext << ",  alpha = " << alpha;
			xNext = xPrev - alpha * frac;
			cout << "\n Новый x = " << xNext << "\n";
		}
		};
	else protectRange = [](double) {};

	function<void()> protectLoop; //защита от зацикливания
	if (loopProtection)
		protectLoop = [&]() {
		for (auto it = xList.begin(); it != xList.end(); ++it)
			if (fabs(xNext - *it) <= (fabs(xNext) + fabs(*it)) * 1e-14) {
				cout << "Защита от зацикливания\n Новый x = " << xNext;
				bool minus = rangeProtection && fabs(b - xNext) < eps;
				xNext += (minus ? -1. : 1.) * eps;   //провека, чтобы не добавить возмущение за границу
				cout << (minus ? " - " : " + ") << eps << "\n";
				xList = { xNext };
				break;
			}
		};
	else protectLoop = []() {};

	//основной цикл
	do {
		//cout << "Iteration " << iter << ",  x = " << xNext << "\n";
		++iter;
		xPrev = xNext;
		double frac = f(xPrev) / fx(xPrev);
		xNext = xPrev - frac;

		protectRange(frac);
		protectLoop();

		xList.push_back(xNext);
	} while (abs(xNext - xPrev) > eps);

	cout << "\nВсего итераций: " << iter << "\nx = " << xNext << "\n\n";
	//errorOrder(xList, xNext);
	return xNext;
}

vector<double> newton_2(const function<double(double, double)>& func1,const function<double(double, double)>& func2, double eps, double startX, double startY) {
	cout << "Метод Ньютона для системы из 2-х уравнений: " << endl << endl;
	vector<double> xNext({startX, startY}), xPrev(2);
	const double h = 1e-6;
	double F11, F12, F21, F22;
	do {
		//print(xNext);
		xPrev = xNext;
		//разностные производные
		F11 = (func1(xPrev[0] + h, xPrev[1]) - func1(xPrev[0] - h, xPrev[1])) / (2 * h);
		F12 = (func1(xPrev[0], xPrev[1] + h) - func1(xPrev[0], xPrev[1] - h)) / (2 * h);
		F21 = (func2(xPrev[0] + h, xPrev[1]) - func2(xPrev[0] - h, xPrev[1])) / (2 * h);
		F22 = (func2(xPrev[0], xPrev[1] + h) - func2(xPrev[0], xPrev[1] - h)) / (2 * h);

		//определитель матрицы Якоби
		double detF = F11 * F22 - F12 * F21;

		//решаем по иттерационной формуле 5.7 из методички
		//обратная матрица:
		//  F22 -F12
		// -F21  F11
		//и делить на detF
		xNext[0] = xPrev[0] - (F22 * func1(xPrev[0], xPrev[1]) - F12 * func2(xPrev[0], xPrev[1])) / detF;
		xNext[1] = xPrev[1] - (-F21 * func1(xPrev[0], xPrev[1]) + F11 * func2(xPrev[0], xPrev[1])) / detF;
	} while (normInf(xNext - xPrev) > eps);
	cout << "Решения системы в заданной области: " << xNext << endl;
	return xNext;
	
}

vector<double> newtonDiagram(const function<double(double, double)>& func1, const function<double(double, double)>& func2, size_t N, const string& fileName) {
	cout <<endl << "Диаграмма сходимости: " << endl;
	const double eps = 1e-6;

	bool analytical = false;
	//analytical = true;

	vector<double> xGrid, yGrid;
	double xR = 10., yR = 10.;
	double hx = 2 * xR / (double)N, hy = 2 * yR / (double)N;

	vector<vector<unsigned>> square;
	square.resize(N);
	for (size_t i = 0; i < N; ++i)
		square[i].resize(N);

	vector<double> xNext(2, 0), xPrev(2, 0);

	unsigned iter = 0;
	const double h = 1e-6;

	double F11, F12, F21, F22;
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j) {
			xNext[0] = -yR + hy * j;
			xNext[1] = -xR + hx * i;
			iter = 0;
			do {
				iter++;
				if (iter > 30) {
					iter = 30;
					break;
				}
				xPrev = xNext;
				//кринге, я согласен
				if (analytical) {   //но по-другому не придумали
					//test4
					/*F11 = 2 * xPrev[0];
					F12 = -2 * xPrev[1];
					F21 = xPrev[1];
					F22 = xPrev[0];*/

					//test5
					F11 = 2 * xPrev[0] + 1;
					F12 = 2 * xPrev[1] + 1;
					F21 = 2 * xPrev[0] + xPrev[1];
					F22 = 2 * xPrev[1] + xPrev[0];
				}
				else {
					F11 = (func1(xPrev[0] + h, xPrev[1]) - func1(xPrev[0] - h, xPrev[1])) / (2 * h);
					F12 = (func1(xPrev[0], xPrev[1] + h) - func1(xPrev[0], xPrev[1] - h)) / (2 * h);
					F21 = (func2(xPrev[0] + h, xPrev[1]) - func2(xPrev[0] - h, xPrev[1])) / (2 * h);
					F22 = (func2(xPrev[0], xPrev[1] + h) - func2(xPrev[0], xPrev[1] - h)) / (2 * h);
				}
				//определитель матрицы Якоби
				double detF = F11 * F22 - F12 * F21;
				if (abs(detF) < 1e-12) {
					iter = 0;
					break;
				}
				xNext[0] = xPrev[0] - (F22 * func1(xPrev[0], xPrev[1]) - F12 * func2(xPrev[0], xPrev[1])) / detF;
				xNext[1] = xPrev[1] - (-F21 * func1(xPrev[0], xPrev[1]) + F11 * func2(xPrev[0], xPrev[1])) / detF;
			} while ((xPrev[0] - xNext[0])* (xPrev[0] - xNext[0]) + (xPrev[1] - xNext[1])*(xPrev[1] - xNext[1]) > eps);
			square[i][j] = iter;
			//if (iter < 3) \
				cout << F11 << " " << F12 << " " << F21 << " " << F22 << "\n";
		}
	ofstream file("Grid.txt", ios_base::out);
	file << xR << " " << yR << " " << N << endl;
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			file << square[i][j] << " ";
		}
		file << endl;
	}
	file.close();
	size_t pvSize = 39;
	size_t cellSize = N / pvSize;
	setlocale(0, "C");
	for (long i = pvSize - 1; i >= 0; --i) {
		for (size_t j = 0; j < pvSize; ++j) {
			size_t i1 = cellSize * i;
			size_t i2 = i1 + cellSize;
			size_t j1 = cellSize * j;
			size_t j2 = j1 + cellSize;
			int value = -1;
			for (size_t ii = i1; ii < i2; ++ii)
				for (size_t jj = j1; jj < j2; ++jj)
					if (square[ii][jj] == 0) {
						value = 0;
						break;
					}
					else if ((int)square[ii][jj] > value)
						value = square[ii][jj];
			if (value == 0)
				cout << (unsigned char)219 << (unsigned char)219;
			else if (value < 4)
				cout << "  ";
			else if (value < 7)
				cout << (unsigned char)176 << (unsigned char)176;
			else if (value < 10)
				cout << (unsigned char)177 << (unsigned char)177;
			else if (value < 20)
				cout << (unsigned char)178 << (unsigned char)178;
			else if (value < 40)
				cout << (unsigned char)219 << (unsigned char)219;
		}
		cout << "\n";
	};
	return xNext;
}
