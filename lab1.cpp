#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

#define T double

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b)
{
    ifstream fin(file);
    if (fin.is_open())
    {
        int n;
        fin >> n;
        A.resize(n, vector<T>(n));
        b.resize(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fin >> A[i][j];
            }
            fin >> b[i];
        }
    }
    fin.close();
}

void printSLAE(const vector<vector<T>>& A, const vector<T>& b)
{
    int n = A.size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i][j] << "x" << j + 1;
            if (j < n - 1)
                cout << " + ";
        }
        cout << " = " << b[i] << endl;
    }
}

//vector<T> GaussianMethod(vector<vector<T>>& A, vector<T>& b)
//{
    
//}

//vector<vector<T>> InvLU(vector<vector<T>>& A)
//{

//}

int main()
{
    vector<vector<T>> A;
    vector<T> b;
    readSLAE("test.txt", A, b);
    printSLAE(A, b);
}
