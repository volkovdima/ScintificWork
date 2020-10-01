#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

// "d" case
int const N = 4;
int const k = 25;
float const Ba = 0.85;
float const Bb = 0.1;
float const Bc = 0.1;
int const alpha = 216;
int const eta = 2;
float const n = 2.6;
int const ks0 = 1;
float const ks1 = 0.01;
float const Q = 0.4;

typedef vec::fixed<N> defSize;
// typedef 
defSize a;
defSize b;
defSize c;
defSize A;
defSize B;
defSize C;
defSize S;


inline defSize Funct_a(defSize a, defSize C, defSize emptyVar)
{
    return -a + (alpha/(1 + pow(C, n)));
}

inline defSize Funct_b(defSize b, defSize A, defSize emptyVar)
{
    return -b + (alpha/(1 + pow(A, n)));
}

inline defSize Funct_c(defSize c, defSize S, defSize B)
{
    return -c + (alpha/(1 + pow(B, n)) + k*S/(1 + S));
}

inline defSize Funct_A(defSize A, defSize a, defSize emptyVar)
{
    return Ba*(a - A);
}

inline defSize Funct_B(defSize B, defSize b, defSize emptyVar)
{
    return Bb*(b - B);
}

inline defSize Funct_C(defSize C, defSize c, defSize emptyVar)
{
    return Bc*(c - C);
}

inline defSize Funct_S(defSize S, defSize A, defSize emptyVar)
{
    double S_mean = mean(S);
    double Se = Q*S_mean;

    return ks0*S + ks1*A - eta*(S - Se);
}

int const iter = 10;

defSize Calc(defSize (*func)(defSize, defSize, defSize))
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_for_betta (0, 1);

    double h = 0.05;

    defSize k1;
    defSize k2;
    defSize k3;
    defSize k4;
    mat::fixed<N, iter> need_mat;

    for (int i = 0; i < N - 1; i++)
    {
        need_mat(i, 0) = distribution_for_betta(generator); 
    }

    for (long int i = 0; i < iter - 1; i++) 
    {
        k1 = (*func)(need_mat.col(i), need_mat.col(i), need_mat.col(i));
        k2 = (*func)(need_mat.col(i) + h/2*k1, need_mat.col(i) + h/2*k1, need_mat.col(i) + h/2*k1);
        k3 = (*func)(need_mat.col(i) + h/2*k2, need_mat.col(i) + h/2*k2, need_mat.col(i) + h/2*k2);
        k4 = (*func)(need_mat.col(i) + h*k3, need_mat.col(i) + h*k3, need_mat.col(i) + h*k3);

        need_mat.col(i + 1) = need_mat.col(i) + h/6*(k1 + 2*k2 + 2*k3 + k4);

    }

    // cout << need_mat << endl;

    return need_mat;
}

int main() 
{

    mat::fixed<N, iter> xmat_a;
    mat::fixed<N, iter> xmat_b;
    mat::fixed<N, iter> xmat_c;
    mat::fixed<N, iter> xmat_A;
    mat::fixed<N, iter> xmat_B;
    mat::fixed<N, iter> xmat_C;
    mat::fixed<N, iter> xmat_S;


    xmat_a = Calc(Funct_a);
    xmat_b = Calc(Funct_b);
    xmat_c = Calc(Funct_c);
    xmat_A = Calc(Funct_A);
    xmat_B = Calc(Funct_B);
    xmat_C = Calc(Funct_C);
    xmat_S = Calc(Funct_S);

    return 0;
}