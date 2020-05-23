#include <iostream>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main() {
    // ex1
    cout << "EX 1: " << endl;
    MatrixXd A(100,100);
    VectorXd b(100);
    VectorXd x(100);

    A = MatrixXd::Random(100, 100);
    b = VectorXd::Random(100);

    A = A * A.transpose();

    // inverse
    clock_t time_stt = clock();
    x = A.inverse() * b;
    cout << "Direct Inverse Time Cost: " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;

    // QR Decomposition decomposition
    time_stt = clock();
    x = A.colPivHouseholderQr().solve(b);
    cout << "QR Decomposition Time Cost: " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;

    // Cholesky Decomposition decomposition
    time_stt = clock();
    x = A.ldlt().solve(b);
    cout << "Cholesky Decomposition Time Cost: " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << endl;

    // ex2
    cout << "EX 2: " << endl;
    Quaterniond q1(0.55, 0.3,  0.2, 0.2);
    Quaterniond q2(-0.1, 0.3, -0.7, 0.2);
    q1.normalize();
    q2.normalize();

    Vector3d t1( 0.7, 1.1, 0.2);
    Vector3d t2(-0.1, 0.4, 0.8);

    Isometry3d T1w(q1);
    Isometry3d T2w(q2);

    T1w.pretranslate(t1);
    T2w.pretranslate(t2);

    Vector3d p1( 0.5, -0.1, 0.2);

    Vector3d p2 = T2w * T1w.inverse() * p1;

    cout << "The coordinate in frame 2 is: " << p2.transpose() << endl;

    return 0;
}