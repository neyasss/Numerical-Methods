#include<fstream>
#include<istream>
#include<vector>
#include<string>
#include<sstream>
#include<iostream>
#include <iomanip>
using namespace std;

//TODO: вынести шаблоны функций в .cpp файл
//запись вектора в файл
template <typename DT>
void writeVectorToFile(ofstream& file, vector<DT> v) {
    for (int i = 0; i < v.size(); ++i)
        file << setprecision(16) << v[i] << " ";
    file << " " << endl;
}

template <typename DT>
void writeVectorToFile(fstream& file, vector<DT> v) {
    for (int i = 0; i < v.size(); ++i)
        file << setprecision(16) << v[i] << " ";
    file << " " << endl;
}
template <typename DT>
void writeVectorToFile(ofstream& file, DT v_0, vector<DT> v)
{
    file << v_0 << " ";
    for (int i = 0; i < v.size(); ++i)
        file << setprecision(16) << v[i] << " ";
    file << " " << endl;
}
template <typename DT>
void writeVectorToFile(fstream& file, DT v_0, vector<DT> v)
{
    file << v_0 << " ";
    for (int i = 0; i < v.size(); ++i)
        file << setprecision(16) << v[i] << " ";
    file << " " << endl;
}

template<typename DT>
void write_data_to_file(string filepath, vector<vector<DT>> data)
{
    ofstream output_data;
    output_data.open(filepath);
    int n = data.size();
    int m = data[0].size();
    cout << m << endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            output_data << data[i][j] << "  ";
        }
        output_data << endl;
    }
    output_data.close();
}



// ВСЕ ЗНАЧЕНИЯ ДАНЫ В СИ
// [rho] = [кг/м^3] плотность
// [c] = [Дж/(кг*К)] теплоёмкость
// [K] = [Вт/(м*К)] теплопроводность

//COPPER: (ГОСТ 859-78)
double COPPER_RHO = 8500;
double COPPER_C = 4200;
double COPPER_K = 407;

//ALUMINUM:  (ГОСТ 22233-83)
double ALUMINUM_RHO = 2600;
double ALUMINUM_C = 840;
double ALUMINUM_K = 221;

//STEEL: (Сталь стержневая арматурная (ГОСТ 10884-81))
double STEEL_RHO = 7850;
double STEEL_C = 482;
double STEEL_K = 58;

//
// Реализация функций и переопределений для std::vector
// для возможности абстракции в математические векторы, матрицы и т.д.
//
//
// Объявление функций и переопределений для std::vector
// для возможности абстракции в математические векторы, матрицы и другой доп. функционал
//

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>


/* *** Начальные функции для испорта/экспорта данных *** */



/* Функция импорта матрицы из текстового файла*/
template <typename T>
std::vector<std::vector<T>> importSLAU(const std::string& filename);


/* Функция вывода матрицы на экран */
template <typename T>
void print(const std::vector<std::vector<T>>& matrix);


/* Функция вывода вектора на экран */
template <typename T>
void print(const std::vector<T>& vec);


/* Функция вывода обрезанного вектора на экран */
template <typename T>
void print_short(const std::vector<T>& vec, const int& n);


/* Функция, которая красиво выводит вектор*/
template<typename T>
void print_vec(const std::vector<T>& vec);


/* Функция вывода разделительной линии на экран */
void printline(const int& n);


/* Функция для получения матрицы из СЛАУ */
template <typename T>
std::vector<std::vector<T>> SLAU_to_matrix(const std::vector<std::vector<T>>& SLAU);


/* Функция для получения векторая из СЛАУ */
template <typename T>
std::vector<T> SLAU_to_vec(const std::vector<std::vector<T>>& SLAU);



/* *** Функции математики векторов *** */



/* Операция cложения векторов */
template <typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const  std::vector<T>& vec2);


/* Операция вычитания векторов */
template <typename T>
std::vector<T> operator-(const std::vector<T>& vec1, const std::vector<T>& vec2);


/* Операция почленного умножения векторов */
template <typename T>
std::vector<T> operator*(const std::vector<T>& vec1, const std::vector<T>& vec2);


/* Операция умножения вектора на число */
template <typename T>
std::vector<T> operator*(const T& c, const std::vector<T>& vec2);


template <typename T>
std::vector<T> operator*(const std::vector<T>& vec2, const T& c);

/* Операция деления вектора на число */
template<typename T>
std::vector<T> operator/(const std::vector<T>& vec, const T& c);


/* Операция почленного деления векторов */
template <typename T>
std::vector<T> operator/(const std::vector<T>& vec1, const std::vector<T>& vec2);


// Определение оператора отрицания для матрицы
template <typename T>
std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>>& matrix);


/* Функция для скалярного умножения векторов */
template <typename T>
T dot(const std::vector<T>& vec1, const std::vector<T>& vec2);


/* Функция для нормы вектора */
template <typename T>
T norm(const std::vector<T>& vec, const int& p = 2);


/* Функция, которая возращает матрицу комбинаций элементов вектора */
template<typename T>
std::vector<std::vector<T>> generateCombinations(const std::vector<T>& vec);


/* Функция, возвращает вектор модулей */
template<typename T>
std::vector<T> vec_abs(const std::vector<T>& vec);


/* Функция, возращающая сумму элементов вектора */
template<typename T>
T sum(const std::vector<T>& vec);




/* *** Функции математики матриц *** */




/* Матричное умножение */
template <typename T>
std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


/* Функция поэлементного сложения матриц */
template <typename T>
std::vector<std::vector<T>> operator+(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


/* Функция поэлементного вычитания матриц */
template <typename T>
std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


/* Функция для умножения матрицы на вектор */
template <typename T>
std::vector<T> operator*(const std::vector<std::vector<T>>& matrix, const std::vector<T>& vec);


/* Функция для транспонирования матрицы */
template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& A);


/* Функция для создания единичной матрицы размера n x n */
template <typename T>
std::vector<std::vector<T>> create_identity_matrix(const int& n);


/* Функция для поэлементного умножения матриц */
template <typename T>
std::vector<std::vector<T>> Multyply(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


/* Функция округления чисел в матрицах */
template <typename T>
std::vector<std::vector<T>> Matrix_round(const std::vector<std::vector<T>>& A, const double& eps);


/* Функция для вычисления нормы матрицы */
template <typename T>
T norm(const std::vector<std::vector<T>>& matrix, const int& p = 2);


/* Функция для вычисления числа обусловленности матрицы c нормой 1*/
template <typename T>
T cond_1(const std::vector<std::vector<T>>& matrix);


/* Функция для вычисления числа обусловленности матрицы c нормой 2*/
template <typename T>
T cond_2(const std::vector<std::vector<T>>& matrix);


/* Функция для вычисления числа обусловленности матрицы c нормой oo*/
template <typename T>
T cond_oo(const std::vector<std::vector<T>>& matrix);


/* Функция поворота матрицы вправо */
template <typename T>
std::vector<std::vector<T>> RotateRight(const std::vector<std::vector<T>>& A);


/* Функция поворота матрицы влево */
template <typename T>
std::vector<std::vector<T>> RotateLeft(const std::vector<std::vector<T>>& A);


// Функция для обратной матрицы с проверкой на вырожденность c определенной точностью
template <typename T>
std::vector<std::vector<T>> inverseMatrix(const std::vector<std::vector<T>>& A, const T& eps);


// Функция для обратной матрицы с проверкой на вырожденность c определенной точностью
template <typename T>
std::vector<std::vector<T>> inverseMatrix(const std::vector<std::vector<T>>& A);


// Функция обрезки матрицы снизу и справа
template <typename T>
std::vector<std::vector<T>> crop_matrix(const std::vector<std::vector<T>>& A, const int& k);


/* Функция, вычисляющая определитель матрицы 4х4 */
template <typename T>
double det(const std::vector<std::vector<T>>& matrix);


/* Функция, сортирующая вектор */
template< typename T>
std::vector<T> sorted(const std::vector<T>& vec_not_sort);


/* Функция, возращающая максимальный элемент вектора */
template<typename T>
T vec_max(const std::vector<T>& vec);


/* Функция, вычисляющая норму разности векторов */
double sqr(std::vector<double> vec1, std::vector<double> vec2);


/* Функция, численно вычисляющая произвоную в точке point по i переменной */
double Differential(std::vector<double>(*F)(const std::vector<double>&), const std::vector<double>& point, const int& i, const double& eps);


/* Функция, вычисляющая градиент функции в точке point */
std::vector<double> Gradient(std::vector<double>(*F)(const std::vector<double>&), const std::vector<double>& point, const double& eps);


/* Функция для сдвига вектора на n элементов */
template<typename T>
std::vector<T> shift(const std::vector<T>& vec, int n);

/* *** Начальные функции для испорта/экспорта данных *** */



/* Функция импорта матрицы из текстового файла*/
template <typename T>
std::vector<std::vector<T>> importSLAU(const std::string& filename) {
    std::vector<std::vector<T>> matrix;
    std::vector<T> vec;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cout << "Error: not open file \n" << std::endl;
        exit(1);
    }

    int size;
    file >> size;

    matrix.resize(size, std::vector<T>(size + 1));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size + 1; ++j) {
            T value;
            if (file >> value) {
                matrix[i][j] = value;
            }
        }
    }

    file.close();
    return matrix;
};


/* Функция вывода матрицы на экран */
template <typename T>
void print(const std::vector<std::vector<T>>& matrix) {
    for (std::vector<T> row : matrix) {
        for (T value : row) {
            std::cout << value << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}


/* Функция вывода вектора на экран */
template <typename T>
void print(const std::vector<T>& vec) {
    for (T value : vec) {
        std::cout << value << ' ';
    }
    std::cout << std::endl;
}


/* Функция вывода обрезанного вектора на экран */
template <typename T>
void print_short(const std::vector<T>& vec, const int& n) {

    for (int i = 0; i < n; ++i) {
        std::cout << vec[i] << ' ';
    }
    std::cout << "..." << std::endl;
}


/* Функция, которая красиво выводит вектор*/
template<typename T>
void print_vec(const std::vector<T>& vec) {
    std::cout << "(" << vec[0];
    for (int i = 1; i < vec.size(); i++) {
        std::cout << ", " << vec[i];
    }
    std::cout << ")" << std::endl;
}


/* Функция вывода разделительной линии на экран */
void printline(const int& n) {
    for (int i = 0; i < n; i++) {
        std::cout << "-";
    }
    std::cout << std::endl;
}


/* Функция для получения матрицы из СЛАУ */
template <typename T>
std::vector<std::vector<T>> SLAU_to_matrix(const std::vector<std::vector<T>>& SLAU) {
    std::vector<std::vector<T>> matrix;
    matrix.resize(SLAU.size(), std::vector<T>(SLAU.size()));

    for (int i = 0; i < SLAU.size(); i++) {
        for (int j = 0; j < SLAU.size(); j++) {
            matrix[i][j] = SLAU[i][j];
        }
    }
    return matrix;
}


/* Функция для получения вектора из СЛАУ */
template <typename T>
std::vector<T> SLAU_to_vec(const std::vector<std::vector<T>>& SLAU) {
    int s = SLAU.size();
    std::vector<T> vec(s);

    for (int i = 0; i < SLAU.size(); i++) {
        vec[i] = SLAU[i][s];
    }
    return vec;
}



/* *** Функции математики векторов *** */



/* Функция для сложения векторов */
template <typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> pert_vec = vec1;
    for (int i = 0; i < vec1.size(); i++) {
        pert_vec[i] += vec2[i];
    }
    return pert_vec;
}


/* Функция вычитания векторов */
template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
    // Проверка на возможность умножения
    if (a.size() != b.size()) {
        std::cout << "Error: size a != size b in substraction vectors." << std::endl;
        exit(1);
    }
    // Создание результирующего вектора
    std::vector<T> result(a.size(), 0);

    // Умножение матрицы на вектор
    for (int i = 0; i < a.size(); ++i) {
        result[i] += a[i] - b[i];
    }
    return result;

}


/* Операция почленного умножения векторов */
template <typename T>
std::vector<T> operator*(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: vector1 size != vector2 size in operator*." << std::endl;
        exit(1);
    }
    std::vector<T> result(vec1.size(), 0);
    for (int i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}


/* Операция умножения вектора на число */
template <typename T>
std::vector<T> operator*(const T& c, const std::vector<T>& vec) {
    std::vector<T> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * c;
    }
    return result;
}


template <typename T>
std::vector<T> operator*(const std::vector<T>& vec, const T& c) {
    std::vector<T> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * c;
    }
    return result;
}

/* Операция деления вектора на число */
template<typename T>
std::vector<T> operator/(const std::vector<T>& vec, const T& c) {
    std::vector<T> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        result[i] = vec[i] / c;
    }
    return result;
}


/* Операция почленного деления векторов */
template <typename T>
std::vector<T> operator/(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: vector1 size != vector2 size in operator*." << std::endl;
        exit(1);
    }
    std::vector<T> result(vec1.size(), 0);
    for (int i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] / vec2[i];
    }
    return result;
}


/* Функция для скалярного умножения векторов */
template <typename T>
T dot(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: vector1 size != vector2 size in operator*." << std::endl;
        exit(1);
    }
    T result;
    for (int i = 0; i < vec1.size(); i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}


/* Функция для нормы вектора */
template <typename T>
T norm(const std::vector<T>& vec, const int& p) {
    if (vec.empty()) {
        std::cerr << "Error: Empty vector in norm() \n";
        exit(1);
    }

    T result = 0.0;

    // Вычисление нормы
    if (p == 0) {
        // Норма oo
        for (const auto& element : vec) {
            T absElement = abs(element);
            if (absElement > result) {
                result = absElement;
            }
        }
    }
    else {
        // Общий случай для норм L1, L2 и т.д.
        for (const auto& element : vec) {
            result += pow(abs(element), p);
        }

        result = pow(result, 1.0 / p);
    }

    return result;
}


/* Функция, которая возращает матрицу комбинаций элементов вектора */
template<typename T>
std::vector<std::vector<T>> generateCombinations(const std::vector<T>& vec) {
    int n = vec.size();

    // Вектор для хранения всех комбинаций
    std::vector<std::vector<T>> combinations;

    // Внешний цикл по всем возможным комбинациям
    for (int i = 0; i < (1 << n); ++i) {
        std::vector<T> current(n);

        // Внутренний цикл для каждой позиции вектора
        for (int j = 0; j < n; ++j) {
            current[j] = (i & (1 << j)) ? vec[j] : -vec[j];
        }

        // Добавить текущую комбинацию в вектор
        combinations.push_back(current);
    }

    return combinations;
}


/* Функция, возвращает вектор модулей */
template<typename T>
std::vector<T> vec_abs(const std::vector<T>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = fabs(vec[i]);
    }
    return vec;
}


/* Функция, возращающая сумму элементов вектора */
template<typename T>
T sum(const std::vector<T>& vec) {
    T sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}



/* *** Функции математики матриц *** */



/* Операция для умножения матрицы на число */
//template <typename T>
//vector<vector<T>> operator*(const vector<vector<T>>& A, const T& scalar){
//    // Создание результирующей матрицы с теми же размерами
//    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));
//
//    // Умножение каждого элемента матрицы на число
//    for (size_t i = 0; i < A.size(); ++i) {
//        for (size_t j = 0; j < A[0].size(); ++j) {
//            result[i][j] = A[i][j] * scalar;
//        }
//    }
//
//    return result;
//}


/* Операция для умножения  числа на матрицу */
//template <typename T>
//vector<vector<T>> operator*(const T& scalar, const vector<vector<T>>& A){
//    // Создание результирующей матрицы с теми же размерами
//    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));
//
//    // Умножение каждого элемента матрицы на число
//    for (size_t i = 0; i < A.size(); ++i) {
//        for (size_t j = 0; j < A[0].size(); ++j) {
//            result[i][j] = A[i][j] * scalar;
//        }
//    }
//
//    return result;
//}


/* Операция поэлементного сложения матриц */
template <typename T>
std::vector<std::vector<T>> operator+(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    // Проверка на совпадение размеров матриц
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        std::cout << "Error: size A != size B in addition matrix." << std::endl;
        exit(1);
    }

    // Создание результирующей матрицы с теми же размерами
    std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size(), 0));

    // Поэлементное сложение
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}


/* Операция поэлементного вычитания матриц */
template <typename T>
std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    // Проверка на совпадение размеров матриц
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        std::cout << "Error: size A != size B in substraction matrix." << std::endl;
        exit(1);
    }

    // Создание результирующей матрицы с теми же размерами
    std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size(), 0));

    // Поэлементное сложение
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}


/* Операция умножения матрицы на вектор */
template <typename T>
std::vector<T> operator*(const std::vector<std::vector<T>>& matrix, const std::vector<T>& vec) {
    // Проверка на возможность умножения
    if (matrix[0].size() != vec.size()) {
        std::cout << "Error: size A != size b in multiply Matrix By Vector." << std::endl;
        exit(1);
    }
    // Создание результирующего вектора
    std::vector<T> result(matrix.size(), 0);

    // Умножение матрицы на вектор
    for (int i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}


/* Матричное умножение */
template <typename T>
std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    int m = A.size();    // Количество строк в матрице A
    int n = A[0].size(); // Количество столбцов в матрице A
    int p = B[0].size(); // Количество столбцов в матрице B

    if (n != B.size()) {
        std::cout << "Error: impossible multiply matrix" << std::endl;
        exit(1);
    }

    std::vector<std::vector<T>> result(m, std::vector<T>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}


// Определение оператора отрицания для матрицы
template <typename T>
std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, 0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = -matrix[i][j];
        }
    }
    return result;
}


/* Функция для поэлементного умножения матриц */
template <typename T>
std::vector<std::vector<T>> Multyply(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    int m = A.size();    // Количество строк в матрице A
    int n = A[0].size(); // Количество столбцов в матрице A
    int p = B[0].size(); // Количество столбцов в матрице B

    if (n != B.size()) {
        printf("Error: impossible multiply matrix");
        exit(1);
    }

    std::vector<std::vector<T>> result(m, std::vector<T>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
    return result;
}


/* Функция округления чисел в матрицах */
template <typename T>
std::vector<std::vector<T>> Matrix_round(const std::vector<std::vector<T>>& A, const double& eps) {
    std::vector<std::vector<T>> roundA = A;
    int size = A.size();

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            roundA[i][j] = (round(A[i][j]) >= 0) ? round(abs(A[i][j]) * (1 / eps)) / (1 / eps) : -1 * round(abs(A[i][j]) * (1 / eps)) / (1 / eps);
        }
    }
    return roundA;
}


/* Функция для вычисления нормы матрицы */
template <typename T>
T norm(const std::vector<std::vector<T>>& matrix, const int& p) {
    // Проверка на пустую матрицу
    if (matrix.empty() || matrix[0].empty()) {
        std::cout << "Error: Empty matrix in norm()\n" << std::endl;
        exit(1);
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    T result = 0.0;

    // Вычисление нормы матрицы
    if (p == 0) {
        // Норма матрицы Чебышева (максимальное значение по модулю в строке)
        for (int i = 0; i < rows; ++i) {
            T rowSum = 0.0;
            for (int j = 0; j < cols; ++j) {
                rowSum += abs(matrix[i][j]);
            }
            if (rowSum > result) {
                result = rowSum;
            }
        }
    }
    else {
        // Общий случай для норм матрицы (Фробениуса и др.)
        for (int j = 0; j < cols; ++j) {
            T colSum = 0.0;
            for (T i = 0; i < rows; ++i) {
                colSum += pow(abs(matrix[i][j]), p);
            }
            result += pow(colSum, 1.0 / p);
        }

        result = pow(result, 1.0 / p);
    }
    return result;
}


/* Функция поворота матрицы вправо */
template <typename T>
std::vector<std::vector<T>> RotateRight(const std::vector<std::vector<T>>& A) {

    std::vector<std::vector<T>> A_rotate(A.size(), std::vector<T>(A.size(), 0));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            A_rotate[A.size() - 1 - j][i] = A[i][j];
        }
    }

    return A_rotate;

}


/* Функция поворота матрицы влево */
template <typename T>
std::vector<std::vector<T>> RotateLeft(const std::vector<std::vector<T>>& A) {

    std::vector<std::vector<T>> A_rotate(A.size(), std::vector<T>(A.size(), 0));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            A_rotate[j][A.size() - 1 - i] = A[i][j];
        }
    }

    return A_rotate;
}


// Функция для создания единичной матрицы размера n x n
template <typename T>
std::vector<std::vector<T>> create_identity_matrix(const int& n) {
    std::vector<std::vector<T>> identity(n, std::vector<T>(n, 0));
    for (int i = 0; i < n; i++) {
        identity[i][i] = 1;
    }
    return identity;
}


// Функция для обратной матрицы с проверкой на вырожденность
template <typename T>
std::vector<std::vector<T>> inverseMatrix(const std::vector<std::vector<T>>& A, const T& eps) {
    std::vector<std::vector<T>> E = create_identity_matrix<T>(A.size());
    std::vector<std::vector<T>> E_rotate = RotateLeft(E);
    std::vector<T> e(A.size());
    std::vector<std::vector<T>> X(A.size(), std::vector<T>(A.size(), 0));


    for (int i = 0; i < A.size(); i++) {
        e = E_rotate[i];
        X[i] = method_Gaussa(A, e, eps);

    }
    std::vector<std::vector<T>> A_inv = RotateLeft(X);
    return A_inv;
}


// Функция для обратной матрицы с проверкой на вырожденность с максимальной точностью
template <typename T>
std::vector<std::vector<T>> inverseMatrix(const std::vector<std::vector<T>>& A) {
    T eps = std::numeric_limits<T>::epsilon();
    return inverseMatrix(A, eps);
}


/* Функция для вычисления числа обусловленности матрицы c нормой 1*/
template <typename T>
T cond_1(const std::vector<std::vector<T>>& matrix) {
    T n_1 = norm_1(matrix);
    if (n_1 == 0) {
        printf("Error: Det(A) = 0  =>  cond_1(A) = oo");
        return std::numeric_limits<T>::infinity();
    }
    std::vector<std::vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_1(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}


/* Функция для вычисления числа обусловленности матрицы c нормой 2*/
template <typename T>
T cond_2(const std::vector<std::vector<T>>& matrix) {
    T n_1 = norm_2(matrix);
    if (n_1 == 0) {
        std::cout << "Error: Det(A) = 0  =>  cond_2(A) = oo" << std::endl;
        return std::numeric_limits<T>::infinity();
    }
    std::vector<std::vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_2(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}


/* Функция для вычисления числа обусловленности матрицы с нормой oo*/
template <typename T>
T cond_oo(const std::vector<std::vector<T>>& matrix) {
    T n_1 = norm_oo(matrix);
    if (n_1 == 0) {
        printf("Error: Det(A) = 0  =>  cond_oo(A) = oo");
        return std::numeric_limits<T>::infinity();
    }
    std::vector<std::vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_oo(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}

/* Функция транспонирования матрицы */
template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}


// Функция обрезки матрицы снизу и справа
template <typename T>
std::vector<std::vector<T>> crop_matrix(const std::vector<std::vector<T>>& A, const int& k) {

    int n = A.size();
    std::vector<std::vector<T>> A_crop(n - k, std::vector<T>(n - k, 0));
    for (int i = 0; i < (n - k); i++) {
        for (int j = 0; j < (n - k); j++) {
            A_crop[i][j] = A[i][j];
        }
    }

    return A_crop;
}

/* Функция, вычисляющая определитель матрицы 4х4 */
template <typename T>
double det(const std::vector<std::vector<T>>& matrix) {
    return
        matrix[0][0] * (
            matrix[1][1] * (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
            matrix[1][2] * (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) +
            matrix[1][3] * (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1])
            ) -
        matrix[0][1] * (
            matrix[1][0] * (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
            matrix[1][2] * (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
            matrix[1][3] * (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0])
            ) +
        matrix[0][2] * (
            matrix[1][0] * (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) -
            matrix[1][1] * (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
            matrix[1][3] * (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0])
            ) -
        matrix[0][3] * (
            matrix[1][0] * (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1]) -
            matrix[1][1] * (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0]) +
            matrix[1][2] * (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0]));
}


/* Функция, сортирующая вектор */
template< typename T>
std::vector<T> sorted(const std::vector<T>& vec_not_sort) {
    std::vector<T> vec(vec_not_sort);
    int n = vec.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (vec[j] > vec[j + 1]) {
                // Обмен элементов, если они не упорядочены
                T temp = vec[j];
                vec[j] = vec[j + 1];
                vec[j + 1] = temp;
            }
        }
    }
    return vec;
}


/* Функция, возращающая максимальный по модулю элемент вектора */
template<typename T>
T vec_max(const std::vector<T>& vec) {
    int n = vec.size();
    T max = 0;
    for (int i = 0; i < n; i++) {
        if (abs(vec[i]) > max)
            max = abs(vec[i]);
    }
    return max;
}


/* ### Переопределение потока вывода для vector ### */

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (int i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


/* Функция, вычисляющая норму разности векторов */
double sqr(std::vector<double> vec1, std::vector<double> vec2) {
    int m = vec1.size();
    double sum;
    for (int i = 0; i < m; i++) {
        sum = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sum;
}


/* Функция, численно вычисляющая произвоную в точке point по i переменной */
double Differential(std::vector<double>(*F)(const std::vector<double>&), const std::vector<double>& point, const int& i, const double& eps) {

    std::vector<double> left_point(point);
    left_point[i] -= eps;
    std::vector<double> right_point(point);
    right_point[i] += eps;

    return (F(right_point)[i] - F(left_point)[i]) / (2 * eps);
}

/* Функция, вычисляющая градиент функции в точке point */
std::vector<double> Gradient(std::vector<double>(*F)(const std::vector<double>&), const std::vector<double>& point, const double& eps) {

    int N = point.size();
    std::vector<double> grad(N, 0);

    for (int i = 0; i < N; i++) {
        grad[i] = Differential(F, point, i, eps);
    }
    return grad;
}


/* Функция для сдвига вектора на n элементов */
template<typename T>
std::vector<T> shift(const std::vector<T>& vec, int n) {
    std::vector<T> shiftedVec(vec.size()); // Создаем вектор той же длины
    int size = vec.size();

    // Если сдвиг больше длины вектора, находим остаток от деления
    n = n % size;

    // Перемещаем элементы вправо
    if (n >= 0) {
        for (int i = 0; i < size; ++i) {
            shiftedVec[(i + n) % size] = vec[i];
        }
    }
    // Перемещаем элементы влево
    else {
        for (int i = 0; i < size; ++i) {
            shiftedVec[(size + i + n) % size] = vec[i];
        }
    }

    return shiftedVec;
}

/* ### Тесты ### */



#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <functional>


/* Класс условий задачи */
class PDE_data {

public:

    double tau;        // Шаг по времени
    double h;          // Шаг по пространству
    double c = 0;      // Удельная теплоемкость    (мб в точке)
    double rho = 0;    // Линейная плотность массы (мб в точке)
    double L = 0;      // Длина стержня
    double t0 = 0;     // Начальная температура
    double T = 0;      // Конец временного интервала
    double u0 = 0;     // Начальная температура
    bool G_left_type = false;  // Тип граничных условий слева (0 - первого рода, 1 - второго рода)
    bool G_right_type = false; // Тип граничных условий справа (0 - первого рода, 1 - второго рода)
    bool K_type = false;       // Коэффициент теплопроводности зависит от температуры (0 - K = const, 1 - K = K(T))

    /* Получение функции теплопроводности K(x) */

    // Функция, для присваивания лямбда-функции к функции double K(double)
    void set_K(std::function<double(double, double)> func) {
        myFunctionK = func;
        K_is_set = true;
    }

    // Функция - коэффициента теплопроводности
    double K(double x, double u) {
        if (myFunctionK) {
            return myFunctionK(x, u);
        }
        else {
            return 0;
        }
    }

    std::function<double(double, double)> K_ptr = [&](double x, double u) {return K(x, u);};

    /* Получение Граничных условий G_left, G_right*/

    // Функция, для задания функции левой границы
    void set_G_left(std::function<double(double)> func) {
        myFunction_G_left = func;
        G_left_is_set = true;
    }

    // Функция, для задания фунции правой границы
    void set_G_right(std::function<double(double)> func) {
        myFunction_G_right = func;
        G_right_is_set = true;
    }

    // Функция для задания начального состояния системы
    void set_init_func(std::function<double(double)> func) {
        initFunction = func;
        init_is_set = true;
    }

    // Функция - Левое граничное условие
    double G_left(double t) {
        if (myFunction_G_left) {
            return myFunction_G_left(t);
        }
        else {
            return 0;
        }
    }

    // Функция - Правое граничное условие
    double G_right(double t) {
        if (myFunction_G_right) {
            return myFunction_G_right(t);
        }
        else {
            return 0;
        }
    }



    // Вывод информации об объекте
    void show() {
        std::cout << "PDE_data object info:" << std::endl;
        std::cout << "c   = " << c << std::endl;
        std::cout << "rho = " << rho << std::endl;
        std::cout << "L   = " << L << std::endl;
        std::cout << "T   = " << T << std::endl;
        std::cout << "K       is " << ((K_is_set) ? "set" : "NOT set") << std::endl;
        std::cout << "G_left  is " << ((G_left_is_set) ? "set" : "NOT set") << std::endl;
        std::cout << "G_right is " << ((G_right_is_set) ? "set" : "NOT set") << std::endl;
    }

    // Вывод вектора значений класса: заданы ли функции
    std::vector<bool> info() {
        return { G_left_type, G_left_is_set, G_right_type, G_right_is_set, K_is_set };
    }


public:
    std::function<double(double)> initFunction;
private:
    // Хранение лямбда-функции как std::function
    std::function<double(double, double)> myFunctionK;
    std::function<double(double)> myFunction_G_left;
    std::function<double(double)> myFunction_G_right;

    // Состояние заданности условий
    bool K_is_set = false;
    bool G_left_is_set = false;
    bool G_right_is_set = false;
    bool init_is_set = false;
};

//
// Реализация функций для решения Уравнений в частных производных(УЧП)
//


//
//печать вектора
template<typename LT>
void out(vector<LT> vec)
{
    int n = vec.size();
    for (int i = 0; i < n; ++i)
    {
        cout << fixed << setprecision(2) << setw(8) << setfill(' ') << vec[i] << "  ";
    }
    cout << endl;

}
//TODO: вынести в отдельный файл
//решение СЛАУ методом правой прогонки
//A x_{i-1} - B x_i + C x_{i+1} = -D  (***)
// Если векторы диагоналей исходной системы заданы как A,B,C,D (A - диагональ опд главной, B - главная диагональ, C - диагональ над главной, D- правая часть)
// То для правильного расчёта необходимо передавать A, (-1.)*B, C, (-1.)*D
// Так как прогонка актуальная для системы (***)
template<typename DT>
std::vector<DT> TridiagonalMatrixAlgorithm(
    std::vector<DT> a,
    std::vector<DT> b,
    std::vector<DT> c,
    std::vector<DT> d
) {
    int n = b.size();
    std::vector<DT> alphas({ c[0] / b[0] });
    std::vector<DT> betas({ d[0] / b[0] });
    for (int i = 1; i < n - 1; ++i)
    {
        DT denom = b[i] - a[i] * alphas[i - 1];
        alphas.push_back(c[i] / denom);
        betas.push_back((d[i] + a[i] * betas[i - 1]) / denom);
    }
    betas.push_back((d[n - 1] + a[n - 1] * betas[n - 2]) / (b[n - 1] - a[n - 1] * alphas[n - 2]));
    std::vector<DT> SolutionX({ betas[n - 1] });
    for (int i = n - 2; i >= 0; --i)
    {
        SolutionX.push_back(alphas[i] * SolutionX[n - i - 2] + betas[i]);
    }
    reverse(SolutionX.begin(), SolutionX.end());
    return SolutionX;
}

/*Инициализация начального состояния во всех точках стержня
 * arguments: n  - amount of points
 *            u0 - initial temperature
 * return:    initial state of the system
 * */

std::vector<double> init_state(int n, double u0)
{
    std::vector<double> result(n, u0);
    return result;
}

std::vector<double> init_state(int n, double h, PDE_data& test)
{
    std::vector<double> result(n, 0);
    double x_i = 0;
    result[0] = test.initFunction(x_i);
    if (!test.G_left_type)
        result[0] = test.G_left(x_i);
    for (int i = 1; i < n - 1; ++i)
    {
        x_i += h;
        result[i] = test.initFunction(x_i);
    }
    x_i += h;
    result[n - 1] = test.initFunction(x_i);
    if (!test.G_right_type)
        result[n - 1] = test.G_right(x_i);
    return result;
}

// Если температура не передана (т.е. K не зависит т температуры u),
// то u присваивается фиктивное значение 0 (какая разница, чему равно u, если в формуле для K оно не используется в return)(аргумент есть, но он не участвует в вычислении - сделано для универсальности)
template<typename F>
double a(F K, double x_i, double x_im, double u_i = 0, double u_im = 0) {
    //return 0.5 * (K(x_i, u_i)+K(x_im, u_im));
    //return K(x_i - 0.5*(x_i-x_im));
    //return sqrt(K(x_i)*K(x_im));

    // Предотвращаем деление на ноль
    if (K(x_i, u_i) + K(x_im, u_im) != 0)
        return 2 * K(x_i, u_i) * K(x_im, u_im) / (K(x_i, u_i) + K(x_im, u_im));
    else
        return sqrt(K(x_i, u_i) * K(x_im, u_im));
}

double w(double a, double u_i, double u_im, double h) {
    return a * (u_i - u_im) / h;
}


// Случай 1 (линейное ур-е)
bool FiniteScheme(double tau, double h, double sigma, PDE_data test, std::string filename = "ExpScheme.txt") {

    // Физические параметры
    double c = test.c;
    double rho = test.rho;
    double t_0 = 0;
    double T = test.T;
    double x_0 = 0;
    double X = test.L;

    // Шаги по времени и пространству
    int num_time_steps = static_cast<int>((T - t_0) / tau);
    int num_space_steps = static_cast<int>((X - x_0) / h);

    // Инициализация начального состояния
    //std::vector<double> state_0 = init_state(num_space_steps, u_0); //TODO: расширить init_state
    std::vector<double> state_0 = init_state(num_space_steps + 1, h, test);
    std::vector<double> As(num_space_steps + 1, 0);
    std::vector<double> Cs(num_space_steps + 1, 0);
    std::vector<double> Bs(num_space_steps + 1, 0);
    std::vector<double> Fs(num_space_steps + 1, 0);

    // Создание файла
    std::string path = filename;
    std::ofstream fpoints(path);
    std::ofstream convergence("convergence.txt");
    std::cout << "log[INFO]: Starting ExplicitScheme" << std::endl;
    std::cout << "log[INFO]: Opening a file \"" << filename << "\" to write..." << std::endl;
    if (fpoints.is_open())
    {   
        
        double t_i = t_0;
        std::vector<double> state_i = state_0;
        vector<double> errors;
        vector<double> delta;
        delta[0] = 0;
        vector<double> logdelta;
        logdelta[0] = 0;
        writeVectorToFile(fpoints, t_i, state_i);
        double x_i = x_0;

        // Эволюция системы во времени
        for (int j = 0; j <= num_time_steps; ++j) {
            t_i += tau;

            // Граничные условия слева

            // 1-го рода
            if (!test.G_left_type) {
                Cs[0] = -1.;
                Bs[0] = 0.;
                As[0] = 0.;
                Fs[0] = -state_0[0];
            }

            // 2-го рода
            else {
                double a1 = a(test.K_ptr, x_0 + h, x_0);
                double w1 = w(a1, state_i[1], state_i[0], h);
                double kappa = sigma * a1 / h / (c * rho * h / (2 * tau) + sigma * a1 / h);
                double mu = (c * rho * state_i[0] * h / (2 * tau) + sigma * test.G_left(t_i) + (1 - sigma) * (test.G_left(t_i - tau) + w1)) / (c * rho * h / (2 * tau) + sigma * a1 / h);
                Cs[0] = -1.;
                Bs[0] = -kappa;
                As[0] = 0;
                Fs[0] = -mu;
            }

            // Граничные условия справа
            // 1-го рода
            if (!test.G_right_type) {
                Bs[num_space_steps] = 0.;
                As[num_space_steps] = 0.;
                Cs[num_space_steps] = -1.;
                Fs[num_space_steps] = -state_0[num_space_steps];
            }

            // 2-го рода
            else {
                double am = a(test.K_ptr, X, X - h);
                double wn = w(am, state_i[num_space_steps], state_i[num_space_steps - 1], h);
                double denom = c * rho * h / (2 * tau) + sigma * am / h;
                double kappa = sigma * am / h / denom;
                double mu = (c * rho * state_i[num_space_steps] * h / (2 * tau) + sigma * test.G_right(t_i) + (1 - sigma) * (test.G_right(t_i - tau) - wn)) / denom;
                Cs[num_space_steps] = -1.;
                Bs[num_space_steps] = 0.;
                As[num_space_steps] = -kappa;
                Fs[num_space_steps] = -mu;
            }

            // Обход пространства
            for (int i = 1; i < num_space_steps; ++i) {
                x_i += h;
                double a_i = a(test.K_ptr, x_i, x_i - h);
                double a_ip = a(test.K_ptr, x_i + h, x_i);
                As[i] = sigma / h * a_i;
                Bs[i] = sigma / h * a_ip;
                Cs[i] = As[i] + Bs[i] + c * rho * h / tau;
                Fs[i] = c * rho * h / tau * state_i[i] +
                    (1 - sigma) * (w(a_ip, state_i[i + 1], state_i[i], h) - w(a_i, state_i[i], state_i[i - 1], h));
            }
            double absError = 0.;
            for (size_t i = 1; i <= size(state_i); ++i) {
                    double error = fabs(test.K_ptr(h * j, tau * i) - state_i[j]);
                    if (error > absError)
                        absError = error;
                }
            errors[j] = absError;
            convergence <<"errors:" << absError << " ";
            // Получение нового состояния системы
            // A - C + B = - F (не домножаем векторы на -1, так как уже считали домноженные)
            state_i = TridiagonalMatrixAlgorithm(As, Cs, Bs, Fs);

            // Запись в файл
            writeVectorToFile(fpoints, t_i, state_i);
        }
        convergence << endl;
     
        for (int k = 1; k <= size(errors); k++) {

            delta[k] = errors[k] - errors[k - 1];
            convergence << "delta:" << delta[k] << " ";
        }
        convergence << endl;
        for (int k = 1; k <= size(delta); k++) {
            logdelta[k] = log(delta[k]) / log(0.5);
            convergence << "logdelta:" << logdelta[k] << " ";
        }
        convergence.close();
        fpoints.close();
        return true;

    }
    else {
        std::cout << "log[ERROR]: Couldn't open or create a file" << std::endl;
        return false;
    }
};

// Итерационный метод решения СЛАУ с трёхдиагональной матрицей
// A x_{i-1} + B x_i + C x_{i+1} = D
// Если исходная система задаётся диагоналями,
// То передавать векторы как они есть (не домножать на -1)
vector<double> TripleBigRelaxSolve(const vector<double>& a, const vector<double>& b,
    const vector<double>& c, const vector<double>& d,
    const vector<double>& x_0, double EPS = 1e-6)
{
    int n = x_0.size();
    int max_iter = 10000;
    int iter = 0;
    double w = 1;
    //LT w = 1.1;
    vector<double> x_now(x_0);
    vector<double> x_prev;

    do {
        x_prev = x_now;
        x_now[0] = (d[0] - c[0] * x_prev[1]);
        x_now[0] *= w;
        x_now[0] /= b[0];
        x_now[0] += (1 - w) * x_prev[0];
        for (int i = 1; i < n - 1; ++i)
        {
            x_now[i] = d[i];
            x_now[i] -= a[i] * x_now[i - 1];
            x_now[i] -= c[i] * x_prev[i + 1];
            x_now[i] *= w;
            x_now[i] /= b[i];
            x_now[i] += (1 - w) * x_prev[i];
        }
        x_now[n - 1] = d[n - 1] - a[n - 1] * x_now[n - 2];
        x_now[n - 1] *= w;
        x_now[n - 1] /= b[n - 1];
        x_now[n - 1] += (1 - w) * x_prev[n - 1];
        ++iter;
    } while (norm(x_now - x_prev) > EPS && iter <= max_iter);

    vector<double> an_sol(n, 2);
    for (int i = 0; i < n; ++i)
        an_sol[i] -= (i + 1) % 2;
    //for (int i = 0; i < n; ++i)
    //cout << x_now[i] << endl;

    //Relax_log_info.C_norm = C_norm;
    //Relax_log_info.aprior = aprior_iters;
    //cout << "Число итераций = " << iter << endl;
    //cout << "Достигнутая точность " << norm(x_now - an_sol) << endl;
    //Relax_log_info.error_vector = error_vector();
    //Relax_log_info.error = vec_norm(Relax_log_info.error_vector);

    return x_now;
}

// Случай 2 (квазилинейное уравнение)
bool IterationScheme(double tau, double h, double sigma, PDE_data test, std::string filename = "ImpScheme.txt") {

    // Физические параметры
    double c = test.c;
    double rho = test.rho;
    double t_0 = 0;
    double T = test.T;
    double x_0 = 0;
    double X = test.L;

    // Шаги по времени и пространству
    int num_time_steps = static_cast<int>((T - t_0) / tau);
    int num_space_steps = static_cast<int>((X - x_0) / h);

    // Инициализация начального состояния
    //std::vector<double> state_0 = init_state(num_space_steps, u_0); //TODO: расширить init_state
    std::vector<double> state_0 = init_state(num_space_steps + 1, h, test);
    std::vector<double> As(num_space_steps + 1, 0);
    std::vector<double> Cs(num_space_steps + 1, 0);
    std::vector<double> Bs(num_space_steps + 1, 0);
    std::vector<double> Fs(num_space_steps + 1, 0);

    // Создание файла
    std::string path = filename;
    std::ofstream fpoints(path);
    std::ofstream convergence("convergence.txt");
    std::cout << "log[INFO]: Starting ExplicitScheme" << std::endl;
    std::cout << "log[INFO]: Opening a file \"" << filename << "\" to write..." << std::endl;
    if (fpoints.is_open()) {
        vector<double> errors;
        vector<double> delta;
        delta[0] = 0;
        vector<double> logdelta;
        double t_i = t_0;
        std::vector<double> state_i = state_0;
        writeVectorToFile(fpoints, t_i, state_i);
        double x_i = x_0;

        // Эволюция системы во времени
        for (int j = 0; j <= num_time_steps; ++j) {
            t_i += tau;
            for (int s = 0; s < 3; ++s) {
                // Граничные условия слева
                // 1-го рода
                if (!test.G_left_type) {
                    Cs[0] = -1.;
                    Bs[0] = 0.;
                    As[0] = 0.;
                    Fs[0] = -state_0[0];
                }
                // 2-го рода
                else {
                    double a0 = a(test.K_ptr, x_0 + h, x_0, state_i[1], state_i[0]);
                    double w0 = w(a0, state_i[1], state_i[0], h);
                    double kappa = sigma * a0 / h / (c * rho * h / (2 * tau) + sigma * a0 / h);
                    double mu = (c * rho * state_i[0] * h / (2 * tau) + sigma * test.G_left(t_i) +
                        (1 - sigma) * (test.G_left(t_i - tau) + w0)) /
                        (c * rho * h / (2 * tau) + sigma * a0 / h);
                    Cs[0] = -1.;
                    Bs[0] = -kappa;
                    As[0] = 0;
                    Fs[0] = -mu;
                }

                // Граничные условия справа
                // 1-го рода
                if (!test.G_right_type) {
                    Bs[num_space_steps] = 0.;
                    As[num_space_steps] = 0.;
                    Cs[num_space_steps] = -1.;
                    Fs[num_space_steps] = -state_0[num_space_steps];
                }
                // 2-го рода
                else {
                    double am = a(test.K_ptr, X, X - h, state_i[num_space_steps], state_i[num_space_steps - 1]);
                    double wn = w(am, state_i[num_space_steps], state_i[num_space_steps - 1], h);
                    double denom = c * rho * h / (2 * tau) + sigma * am / h;
                    double kappa = sigma * am / h / denom;
                    double mu = (c * rho * state_i[num_space_steps] * h / (2 * tau) + sigma * test.G_right(t_i) +
                        (1 - sigma) * (test.G_right(t_i - tau) - wn)) / denom;
                    Cs[num_space_steps] = -1.;
                    Bs[num_space_steps] = 0.;
                    As[num_space_steps] = -kappa;
                    Fs[num_space_steps] = -mu;
                }

                // Обход пространства
                for (int i = 1; i < num_space_steps; ++i) {
                    x_i += h;
                    double a_i = a(test.K_ptr, x_i, x_i - h, state_i[i], state_i[i - 1]);
                    double a_ip = a(test.K_ptr, x_i + h, x_i, state_i[i + 1], state_i[i]);
                    As[i] = sigma / h * a_i;
                    Bs[i] = sigma / h * a_ip;
                    Cs[i] = (As[i] + Bs[i] + c * rho * h / tau);
                    Fs[i] = (c * rho * h / tau * state_i[i] +
                        (1 - sigma) *
                        (w(a_ip, state_i[i + 1], state_i[i], h) - w(a_i, state_i[i], state_i[i - 1], h)));

                }
                double absError = 0.;
                for (size_t i = 1; i <= size(state_i); ++i) {
                    double error = fabs(test.K_ptr(h * j, tau * i) - state_i[j]);
                    if (error > absError)
                        absError = error;
                }
                errors[j] = absError;
                convergence << "errors:" << absError << " ";
                // Получение нового состояния системы
                //state_i = TripleBigRelaxSolve(As, Cs, Bs, Fs, state_i);
                state_i = TridiagonalMatrixAlgorithm(As, Cs, Bs, Fs);
                // Запись в файл
            }

            writeVectorToFile(fpoints, t_i, state_i);
        }
        convergence << endl;

        for (int k = 1; k <= size(errors); k++) {

            delta[k] = errors[k] - errors[k - 1];
            convergence << "delta:" << delta[k] << " ";
        }
        convergence << endl;
        for (int k = 1; k <= size(delta); k++) {
            logdelta[k] = log(delta[k]) / log(0.5);
            convergence << "logdelta:" << logdelta[k] << " ";
        }
        convergence.close();
        fpoints.close();
        return true;
    }
    else {
        std::cout << "log[ERROR]: Couldn't open or create a file" << std::endl;
        return false;
    }

};


//Для определения числа итераций "до сходимости"
bool infoIterationScheme(double tau, double h, double sigma, PDE_data test, std::string filename = "ImpScheme.txt", double EPS = 1e-12) {

    // Физические параметры
    double c = test.c;
    double rho = test.rho;
    double t_0 = 0;
    double T = test.T;
    double x_0 = 0;
    double X = test.L;

    // Шаги по времени и пространству
    int num_time_steps = static_cast<int>((T - t_0) / tau);
    int num_space_steps = static_cast<int>((X - x_0) / h);

    // Инициализация начального состояния
    //std::vector<double> state_0 = init_state(num_space_steps, u_0); //TODO: расширить init_state
    std::vector<double> state_0 = init_state(num_space_steps + 1, h, test);
    std::vector<double> As(num_space_steps + 1, 0);
    std::vector<double> Cs(num_space_steps + 1, 0);
    std::vector<double> Bs(num_space_steps + 1, 0);
    std::vector<double> Fs(num_space_steps + 1, 0);

    // Создание файла
    std::string path = filename;
    std::ofstream fpoints(path);
    std::cout << "log[INFO]: Starting ExplicitScheme" << std::endl;
    std::cout << "log[INFO]: Opening a file \"" << filename << "\" to write..." << std::endl;
    if (fpoints.is_open()) {

        double t_i = t_0;
        std::vector<double> state_i = state_0;
        std::vector<double> state_ipp = state_i;
        writeVectorToFile(fpoints, t_i, state_i);
        double x_i = x_0;
        int iter_counter = 0;
        // Эволюция системы во времени
        for (int j = 0; j <= num_time_steps; ++j) {
            t_i += tau;
            iter_counter = 0;
            do {
                state_i = state_ipp;
                // Граничные условия слева
                // 1-го рода
                if (!test.G_left_type) {
                    Cs[0] = -1.;
                    Bs[0] = 0.;
                    As[0] = 0.;
                    Fs[0] = -state_0[0];
                }
                // 2-го рода
                else {
                    double a0 = a(test.K_ptr, x_0 + h, x_0, state_i[1], state_i[0]);
                    double w0 = w(a0, state_i[1], state_i[0], h);
                    double kappa = sigma * a0 / h / (c * rho * h / (2 * tau) + sigma * a0 / h);
                    double mu = (c * rho * state_i[0] * h / (2 * tau) + sigma * test.G_left(t_i) +
                        (1 - sigma) * (test.G_left(t_i - tau) + w0)) /
                        (c * rho * h / (2 * tau) + sigma * a0 / h);
                    Cs[0] = -1.;
                    Bs[0] = -kappa;
                    As[0] = 0;
                    Fs[0] = -mu;
                }
                // Граничные условия справа
                // 1-го рода
                if (!test.G_right_type) {
                    Bs[num_space_steps] = 0.;
                    As[num_space_steps] = 0.;
                    Cs[num_space_steps] = -1.;
                    Fs[num_space_steps] = -state_0[num_space_steps];
                }
                // 2-го рода
                else {
                    double am = a(test.K_ptr, X, X - h, state_i[num_space_steps], state_i[num_space_steps - 1]);
                    double wn = w(am, state_i[num_space_steps], state_i[num_space_steps - 1], h);
                    double denom = c * rho * h / (2 * tau) + sigma * am / h;
                    double kappa = sigma * am / h / denom;
                    double mu = (c * rho * state_i[num_space_steps] * h / (2 * tau) + sigma * test.G_right(t_i) +
                        (1 - sigma) * (test.G_right(t_i - tau) - wn)) / denom;
                    Cs[num_space_steps] = -1.;
                    Bs[num_space_steps] = 0.;
                    As[num_space_steps] = -kappa;
                    Fs[num_space_steps] = -mu;
                }

                // Обход пространства
                for (int i = 1; i < num_space_steps; ++i) {
                    x_i += h;
                    double a_i = a(test.K_ptr, x_i, x_i - h, state_i[i], state_i[i - 1]);
                    double a_ip = a(test.K_ptr, x_i + h, x_i, state_i[i + 1], state_i[i]);
                    As[i] = sigma / h * a_i;
                    Bs[i] = sigma / h * a_ip;
                    Cs[i] = (As[i] + Bs[i] + c * rho * h / tau);
                    Fs[i] = (c * rho * h / tau * state_i[i] +
                        (1 - sigma) *
                        (w(a_ip, state_i[i + 1], state_i[i], h) - w(a_i, state_i[i], state_i[i - 1], h)));

                }
                // Получение нового состояния системы
                //state_ipp = TripleBigRelaxSolve(As, (-1.)*Cs, Bs, (-1.)*Fs, state_i);
                state_ipp = TridiagonalMatrixAlgorithm(As, Cs, Bs, Fs);
                ++iter_counter;
            } while (norm(state_ipp + (-1.) * state_i) >= EPS);
            std::cout << "Iterations on time-step" << t_i << " is " << iter_counter << std::endl;
            // Запись в файл
            writeVectorToFile(fpoints, t_i, state_i);
        }
        fpoints.close();
        return true;
    }
    else {
        std::cout << "log[ERROR]: Couldn't open or create a file" << std::endl;
        return false;
    }
};


//
//std::vector<double> progonka(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d) {
//
//    int n = b.size();
//    std::vector<double> alph(n, 0);
//    std::vector<double> beth(n, 0);
//    std::vector<double> solve(n, 0);
//    double tmp;
//
//    //vectinput();
//    alph[0] = 0;
//    beth[0] = 0;
//    tmp = b[0];
//    alph[1] = c[0] / tmp;
//    beth[1] = d[0] / tmp;
//
//    //std::cout << alph[1] << " " << beth[1] << "\n";
//    for (int i = 1; i < n - 1; i++) {
//        tmp = b[i] - a[i] * alph[i];
//        alph[i + 1] = c[i] / tmp;//(b[i] - a[i] * alph[i]);
//        beth[i + 1] = (d[i] + a[i] * beth[i]) / tmp; // (b[i] - a[i] * alph[i]);
//    }
//
//    solve[n - 1] = (d[n - 1] + a[n - 1] * beth[n - 1]) / (b[n - 1] - a[n - 1] * alph[n - 1]);
//
//    for (int i = n - 2; i >= 0; i--) {
//        solve[i] = alph[i + 1] * solve[i + 1] + beth[i + 1];
//    }
//
//    return solve;
//}
//
///* Функция для решения PDE в случае 1 (по методичке) */
//bool SolvePDE_1(PDE_data test, double tau, double h, double sigma, std::string filename){
//
//
//    double t_0 = 0; // Начальное время
//    double x_0 = 0; // Начальное условие?
//
//    // Количество шагов по времени и пространству
//    int num_time_steps = static_cast<int>((test.T-t_0) / tau);
//    int num_space_steps = static_cast<int>((test.L - x_0) / h);
//
//    // TODO: брать граничное условие из теста (здесь начальная температура)
//    double u_0 = 10;
//
//    // Инициализация начального состояния
//    std::vector<double> state_0 = init_state(num_space_steps, u_0); //TODO: расширить init_state
//    std::vector<double> As(num_space_steps, 0);
//    std::vector<double> Cs(num_space_steps, 0);
//    std::vector<double> Bs(num_space_steps, 0);
//    std::vector<double> Fs(num_space_steps, 0);
//
//    std::string path = "./OutputData/" + filename + ".txt";
//    std::ofstream fpoints(path);
//    std::cout << "log[INFO]: Starting ExplicitScheme" << std::endl;
//    std::cout << "log[INFO]: Opening a file \"" << filename << "\" to write..." << std::endl;
//
//    if (fpoints.is_open()) {
//
//        double t_i = t_0;
//        std::vector<double> state_i = state_0;
//        int ind = 0;
//        writeVectorToFile(fpoints, t_i, state_i);
//        double x_i = x_0;
//        Cs[0] = 1;
//        Bs[0] = 0;
//        As[0] = 0;
//        Fs[0] = u_0;
//        Bs[num_space_steps-1] = 0;
//        As[num_space_steps-1] = 0;
//        Cs[num_space_steps-1] = 1;
//        Fs[num_space_steps-1] = u_0;
//
//        for (int j = 0; j < num_time_steps; ++j) {
//            t_i += tau;
//
//            for (int i = 1; i < num_space_steps - 1; ++i) {
//                x_i += h;
//
//                //double a_i = a(test.K_ptr, x_i, x_i - h);
//                double a_i = 0.5 * (test.K(x_i) + test.K(x_i - h));
//
//                //double a_ip = a(test.K_ptr, x_i + h, x_i);
//                double a_ip = 0.5 * (test.K(x_i + h) + test.K(x_i - h));
//
//                As[i] = sigma / h * a_i;
//                Bs[i] = sigma / h * a_ip;
//                Cs[i] = As[i] + Bs[i] + test.c * test.rho * h / tau;
//                Fs[i] = test.c * test.rho * h / tau * state_i[i]
//                        + (1 - sigma) * (w(a_ip, state_i[i + 1],
//                                           state_i[i], h) - w(a_i, state_i[i], state_i[i - 1], h));
//            }
//
//            Cs = (-1.) * Cs;
//            Fs = (-1.) * Fs;
//            state_i = progonka(As, Cs, Bs, Fs);
//            writeVectorToFile(fpoints, t_i, state_i);
//        }
//
//        fpoints.close();
//        return true;
//
//    } else {
//        std::cout << "log[ERROR]: Couldn't open or create a file" << std::endl;
//        return false;
//    }
//}





int main() {

    /* Тест 1: Алюминий, фиксированная температура на концах */
    PDE_data test1;
    test1.c = 1;
    test1.rho = 1;
    test1.h = 0.05;
    test1.L = 1.;
    test1.tau = 0.009;
    test1.tau = 1.;
    test1.T = 100.;
    test1.u0 = 800.;
    test1.set_G_left([&](double x) { return test1.u0; });
    test1.G_left_type = false;
    test1.set_G_right([&](double x) { return test1.u0; });
    test1.G_right_type = false;
    test1.set_init_func([&](double x) { return test1.u0 - 500 - x * (test1.L - x); });
    test1.set_K([&](double x, double u) {return 237 * (1 + 0.0034 * (u - 293));});
    test1.K_type = true; //Решаем итерационным методом
    if (!test1.K_type) {
        FiniteScheme(test1.tau, test1.h, 1., test1, "test1.txt");
    }
    else {
        IterationScheme(test1.tau, test1.h, 0., test1, "test1_iterational.txt");
    }
    test1.set_K([&](double x, double u) { return ALUMINUM_K; });
    test1.K_type = false;
    if (!test1.K_type) {
        FiniteScheme(test1.tau, test1.h, 1., test1, "test1.txt");
    }
    else {
        IterationScheme(test1.tau, test1.h, 0., test1, "test1_iterational.txt");
    }
    test1.show(); // Вывод информации о тесте



    /* Тест 2: Алюминий, постоянная температура на левом конце + нулевой поток на правом (теплоизоляция) */
    PDE_data test2;
    test2.c = ALUMINUM_C;
    test2.rho = ALUMINUM_RHO;
    test2.L = 1.;
    test2.T = 100.;
    test2.h = 0.005;
    test2.tau = 0.1;
    test2.u0 = 800.;
    test2.set_G_left([&](double x) { return test2.u0; });
    test2.G_left_type = false;
    test2.set_G_right([&](double x) { return 0; });
    test2.G_right_type = true;
    test2.set_init_func([&](double x) { return test2.u0 - 500 - x * (test2.L - x); });
    test2.set_K([&](double x, double u) { return ALUMINUM_K; });
    test2.K_type = false;
    if (!test2.K_type) {
        FiniteScheme(test2.tau, test2.h, 1., test2, "test2.txt");
    }
    else {
        IterationScheme(test2.tau, test2.h, 0., test2, "test2_iterational.txt");
    }



    /* Тест: Вариант 2 */
    PDE_data testv;
    testv.c = 1;
    testv.rho = 1;
    testv.L = 1;
    testv.t0 = 0.5;
    testv.T = 10;
    testv.u0 = 0.1;
    testv.h = 0.5;
    testv.tau = 0.5;

    testv.set_K([](double x, double u) {
        double x1 = 1. / 3, x2 = 2. / 3.;
        double k1 = 1., k2 = 0.1;
        double L = 1;
        double alpha = 2;
        double beta = 0.5;
        double gamma = 3;

        // return alpha + beta * pow(u, gamma);

        if (x <= x1) {
            return k1;
        }
        else if (x < x2) {
            return (k1 * ((x - x2) / (x1 - x2)) + k2 * ((x - x1) / (x2 - x1)));
        }
        else if (x <= L) {
            return k2;
        }
        else {
            return 0.;
        }
        });
    testv.set_G_left([&](double t) { if (t < 0.5) return 20 * t; else return 0.; });
    testv.G_left_type = false;
    testv.set_G_right([](double x) { return 0.1; });
    testv.G_right_type = true;
    testv.set_init_func([](double x) { return 0.1; });  // Начальная температура по всему стержню
    testv.K_type = true;
    if (!testv.K_type) {
        FiniteScheme(testv.tau, testv.h, 1., testv, "testv.txt");
    }
    else {
        IterationScheme(testv.tau, testv.h, 0., testv, "testv_iterational.txt");
    }

    /* Случай test 5. */
    //ExplicitScheme(0.01, 0.1, 1., test5);
    //SolvePDE_1(test5, 0.01, 0.1, 0.5, "ExpScheme_test5");



    std::cout << std::endl << "Complete!" << std::endl;
}
