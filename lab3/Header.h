#pragma once

/* algebra.h и algebra.cpp - основательные файлы,
 * в которых соответственно объявляются и реализовываются функции
 * для работы с матрицами, векторами и т.д.
 * */


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>

using namespace std;

/* *** Начальные функции для испорта/экспорта данных *** */

/* Функция импорта матрицы из текстового файла*/
template <typename T>
vector<vector<T>> importSLAU(const string& filename);

/* Функция импорта матрицы из текстового файла*/
template <typename T>
vector<vector<T>> importMatrix(const string& filename);

/* Функция вывода матрицы на экран */
template <typename T>
void print(const vector<vector<T>>& matrix);


/* Функция вывода вектора на экран */
template <typename T>
void print(const vector<T>& vec);


/* Функция вывода обрезанного вектора на экран */
template <typename T>
void print_short(const vector<T>& vec, const int& n);

/* Функция, которая красиво выводит вектор*/
template<typename T>
void print_vec(const vector<T>& vec);

/* Функция для получения матрицы из СЛАУ */
template <typename T>
vector<vector<T>> SLAU_to_matrix(const vector<vector<T>>& SLAU);


/* Функция для получения векторая из СЛАУ */
template <typename T>
vector<T> SLAU_to_vec(const vector<vector<T>>& SLAU);


/* *** Функции математики векторов *** */

/* Операция cложения векторов */
template <typename T>
vector<T> operator+(const vector<T>& vec1, const  vector<T>& vec2);

/* Операция вычитания векторов */
template <typename T>
vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2);

/* Операция почленного умножения векторов */
template <typename T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2);

/* Операция умножения вектора на число */
template <typename T>
vector<T> operator*(const T& c, const vector<T>& vec2);

template <typename T>
vector<T> operator*(const vector<T>& vec2, const T& c);



/* Операция почленного деления векторов */
template <typename T>
vector<T> operator/(const vector<T>& vec1, const vector<T>& vec2);

// Определение оператора отрицания для матрицы
template <typename T>
vector<vector<T>> operator-(const vector<vector<T>>& matrix);

/* Функция для скалярного умножения векторов */
template <typename T>
T dot(const vector<T>& vec1, const vector<T>& vec2);


/* Функция для нормы вектора */
template <typename T>
T norm(const vector<T>& vec, const int& p = 2);

/* Функция, возвращает вектор модулей */
template<typename T>
vector<T> vec_abs(const vector<T>& vec);



/* Функция, возращающая сумму элементов вектора */
template<typename T>
T sum(const vector<T>& vec);




/* *** Функции математики матриц *** */

/* Матричное умножение */
template <typename T>
vector<vector<T>> operator*(const vector<vector<T>>& A, const vector<vector<T>>& B);

/* Операция для умножения матрицы на число */
template <typename T>
vector<vector<T>> operator*(const vector<vector<T>>& A, const T& scalar);

template <typename T>
vector<vector<T>> operator*(const T& scalar, const vector<vector<T>>& A);

template <typename T>
vector<vector<T>> operator*(const vector<vector<T>>& A, const T& scalar);

/* Функция поэлементного сложения матриц */
template <typename T>
vector<vector<T>> operator+(const vector<vector<T>>& A, const vector<vector<T>>& B);


/* Функция поэлементного вычитания матриц */
template <typename T>
vector<vector<T>> operator-(const vector<vector<T>>& A, const vector<vector<T>>& B);


/* Функция для умножения матрицы на вектор */
template <typename T>
vector<T> operator*(const vector<vector<T>>& matrix, const vector<T>& vec);

/* Функция для транспонирования матрицы */
template <typename T>
vector<vector<T>> transpose(const vector<vector<T>>& A);


/* Функция для создания единичной матрицы размера n x n */
template <typename T>
vector<vector<T>> create_identity_matrix(const int& n);

template <typename T>
vector<vector<T>> E(const int& n);

/* Функция для поэлементного умножения матриц */
template <typename T>
vector<vector<T>> Multyply(const vector<vector<T>>& A, const vector<vector<T>>& B);


/* Функция округления чисел в матрицах */
template <typename T>
vector<vector<T>> Matrix_round(const vector<vector<T>>& A, const double& eps);


/* Функция для вычисления нормы матрицы */
template <typename T>
T norm(const vector<vector<T>>& matrix, const int& p = 2);


/* Функция для вычисления числа обусловленности матрицы c нормой 1*/
template <typename T>
T cond_1(const vector<vector<T>>& matrix);


/* Функция для вычисления числа обусловленности матрицы c нормой 2*/
template <typename T>
T cond_2(const vector<vector<T>>& matrix);


/* Функция для вычисления числа обусловленности матрицы c нормой oo*/
template <typename T>
T cond_oo(const vector<vector<T>>& matrix);


/* Функция поворота матрицы вправо */
template <typename T>
vector<vector<T>> RotateRight(const vector<vector<T>>& A);


/* Функция поворота матрицы влево */
template <typename T>
vector<vector<T>> RotateLeft(const vector<vector<T>>& A);


// Функция для обратной матрицы с проверкой на вырожденность c определенной точностью
template <typename T>
vector<vector<T>> inverseMatrix(const vector<vector<T>>& A, const T& eps);

// Функция для обратной матрицы с проверкой на вырожденность c определенной точностью
template <typename T>
vector<vector<T>> inverseMatrix(const vector<vector<T>>& A);

// Функция обрезки матрицы снизу и справа
template <typename T>
vector<vector<T>> crop_matrix(const vector<vector<T>>& A, const int& k);

/* Функция, вычисляющая определитель матрицы 4х4 */
template <typename T>
double det(const vector<vector<T>>& matrix);


/* Функция, сортирующая вектор */
template< typename T>
vector<T> sorted(const vector<T>& vec_not_sort);

/* Функция, возращающая максимальный элемент вектора */
template<typename T>
T vec_max(const vector<T>& vec);

/* Функция, возращающая det(A - lambda * E) */
template<typename T>
T test_eigen(const vector<vector<T>>& matrix, const vector<T>& lambda);

/* Функция, возращающая норму разницы решений */
template<typename T>
T testeps(const vector<T>& x, const vector<T>& true_x, const int p);

template<typename T>
T test_eigen_vec(const vector<vector<T>>& matrix, vector<vector<T>> eigen_vec, const vector<T>& lambda);

template<typename T>
T test_eigen_vec2(const vector<vector<T>>& matrix, vector<T> eigen_vec, const T lambda);

template<typename T>
T cos_vec(const vector<T> x1, const vector<T> x2);

template<typename T>
vector<vector<T>> uncrop(vector<vector<T>> A, vector<vector<T>>A1, int crop);


/* methods.h и methods.cpp - файлы,
 * в которых соответственно объявляются и реализовываются
 * функции методов необходимых в лабораторной работе.
 * */


 /* ### Функции лабы 1 ### */

 /* Функция для LU-разложения с частичным выбором */
template <typename T>
void lu_decomposition(const vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& U);


/* Функция для вычисления нормы вектора невязки */
template <typename T>
T norm_vector_nevazki(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, const int& n);


/* Функция для решения СЛАУ прямым методом Гаусса */
template <typename T>
vector<T> method_Gaussa(const vector<vector<T>>& matrix, const vector<T>& vec, const T& eps);
template <typename T>
vector<T> method_Gaussa2(const vector<vector<T>>& matrix, const vector<T>& vec, const T& eps);

/* Функция QR-разложения матрицы методом вращений */
template<typename T>
void QR_decomposition(const vector<vector<T>>& matrix, vector<vector<T>>& Q, vector<vector<T>>& R, const T& eps);


/* Функция для решения СЛАУ методом QR-разложения */
template <typename T>
vector<T> method_QR(const vector<vector<T>>& A, const vector<T>& b, const T& eps);


/* Функция для оценки изменения числа обуcловленности от возмущения вектора правой части */
template <typename T>
void min_change_cond(const vector<vector<T>>& matrix, const vector<T>& vec, const vector<T>& mod);


/* ### Функций лабы 2 ### */

/* Структура, с помощью которой будет выводится результат метода лаб 3 */
template<typename T>
struct MyResult2 {
    vector<T> solve;
    int iterations;
    vector<vector<T>> C;
    vector<T> y;
    T batch;

};


/* Функция преобразования матрицы в сумму из Нижнетреугольной, Диагональной, Верхнетреугольной */
template<typename T>
void LDU_decomposotion(const vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U);


/* Функция решения СЛАУ методом Простой Итерации */
template<typename T>
MyResult2<T> method_SimpleIteration(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& tau, const T& eps, const int& p, const int& MaxIter);


/* Функция решения СЛАУ методом Якоби */
template<typename T>
MyResult2<T> method_Yacobi(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& eps, const int& p, const int& MaxIter);


/* Функция решения СЛАУ методом Релаксации */
template<typename T>
MyResult2<T> method_Relax(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& w, const T& eps, const int& p, const int& MaxIter);


/* Функция решения СЛАУ методом Зейделя */
template<typename T>
MyResult2<T> method_Zeidel(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& eps, const int& p, const int& MaxIter);


/* Функция решения трехдиагональной СЛАУ большой размерности методом Зейделя */
template <typename T>
vector<T> method_Zeidel_diag(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& x0, const T& eps, const T& maxIterations);


/* Функция решения трехдиагональной СЛАУ большой размерности методом Релаксации */
template <typename T>
vector<T> method_Relax_diag(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& x0, const T& w, const T& eps, const T& MaxIter);


/* Функция для вычисления нормы вектора невязки трехдиагональной СЛАУ */
template <typename T>
T norm_vector_nevazki(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& solution, const int& n);


/* Функция исследования итерационного параметра tau для метода простых итераций (Метод Золотого сечения)*/
template<typename T>
T SimpleIterations_method_matrix_norm_C(const vector<vector<T>>& A, const T& tau, const int& p);


// Метод золотого сечения для поиска минимума функции на заданном интервале [a, b]
template<typename T>
T golden_section_search_tau(const vector<vector<T>>& A, T a, T b, const int& p, const T& epsilon);


/* Функция исследования итерационного параметра W для метода Релаксации (Метод Золотого сечения)*/
template<typename T>
T golden_section_search_W(const vector<vector<T>>& A, T a, T b, const int& p, const T& eps);


/* Функция от которой ищется минимум в золотом сечении для релаксации */
template <typename T>
T C_matrix_for_relax(const vector<vector<T>>& A, const T& w, const int& p);


/* Функция исследования итерационного параметра W для метода Релаксации для трехдиагональной матрицы (Метод Золотого сечения)*/
template<typename T>
T golden_section_search_W(vector<T> A, vector<T> B, vector<T> C, vector<T> vec, vector<T> x, T EPS, int MaxIteration, T a, T b);


/* Функция априорной оценки */
template <typename T>
void aprior_eps(const vector<vector<T>>& C, const vector<T>& y, const vector<T>& x0, const int& p);


/* Функция апостериорной оценки */
template <typename T>
void aposter_eps(const vector<vector<T>>& C, T norm_delta, const int& p);



/* ### Функций лабы 4 ### */

/* Структура для вывода данных 4 лабы*/
template <typename T>
struct MyResult4 {
    vector<T>  eigen;
    int iterations = 0;
    vector<vector<T>> eigens_vec;
    vector<vector<T>> R;
    vector<vector<vector<T>>> A_iter;
};

/* Функция приведения матрицы к форме Хессенберга методом вращений */
template <typename T>
vector<vector<T>> Hessenberg_decomposition(const vector<vector<T>>& matrix);


/* Функция нахождения собственных значений матрицы методом QR-разложения за одну итерацию */
template <typename T>
MyResult4<T> Eigen_method_QR(const vector<vector<T>>& A, const T& eps, const int& maxIterations);

/* Функция нахождения собственных значений матрицы методом QR-разложения */
template <typename T>
MyResult4<T> Eigen_method_QR(const vector<vector<T>>& matrix, const T& sigma, const T& eps, const int& maxIterations);

template <typename T>
MyResult4<T> Eigen_method_QR2(const vector<vector<T>>& matrix, const T& eps, const int& maxIterations);

template <typename T>
MyResult4<T> Eigen_method_QR3(const vector<vector<T>>& matrix, const T& eps, const int& maxIterations);
template <typename T>
MyResult4<T> Eigen_method_QR3(const vector<vector<T>>& matrix, const T& sigma, const T& eps, const int& maxIterations);

/* Функция нахождения собственных векторов матрицы методом Обратных Итераций */
template<typename T>
MyResult4<T> reverse_iteration(const vector<vector<T>>& matrix, const vector<T>& lambda, const T& eps, const int& maxIteration);

/* Функция нахождения собственных значений и собственных векторов методом Обратных Итераций
 * с использованием отношения Рэлея (Модификация метода Обратных Итераций) */

 //template <typename T>
 //MyResult4<T> reverse_iterator_with_reley(const vector<vector<T>>& matrix, const vector<vector<T>>& X0, const T eps, const int& maxIteration);

template <typename T>
MyResult4<T> reverse_iterator_with_reley(const vector<vector<T>>& matrix, const vector<vector<T>>& X0, const T eps, const int& maxIteration);
