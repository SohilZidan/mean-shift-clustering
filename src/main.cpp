#include <iostream>
#include <Eigen/Dense>
#include "distance.hpp"
#include "kernel.hpp"
#include "meanshift.hpp"

using namespace std;
using namespace ModelFitting;

int main(int argc, char* argv[])
{

    cout<< "Welcome Mean Shift"<<endl;
    Vec3d p_a, p_b;
    // p_a << 2,3,4;
    // p_b << 0,0,2;
    // cout << euclidean_distance(p_a, p_b) << endl;
    // cout << jaccard_distance(p_a, p_b) << endl;

    // p_a << 2,3,6;
    // p_b << 0,0,4;
    // cout << euclidean_distance(p_a, p_b) << endl;
    // cout << jaccard_distance(p_a, p_b) << endl;

    // p_a << 2,3,16;
    // p_b << 0,0,14;
    // cout << euclidean_distance(p_a, p_b) << endl;
    // cout << jaccard_distance(p_a, p_b) << endl;

    MatXd x(3,1), y;
    MatXd t;
    x << 1, 2, 3;
    y = x;
    y(1) = 18;
    // cout << t.rows() << endl;
    // t.conservativeResize(1,1);
    // cout << t.rows() << endl;
    // cout << t.cols() << endl;
    // t.conservativeResize(2,Eigen::NoChange);
    // cout << t.rows() << endl;
    // cout << t.cols() << endl;
    cout << Eigen::nbThreads( ) << endl;
    // cout << y << endl;
    // cout << x << endl;
    // cout << gaussian_kernel(0, 2) << endl;
    // cout << gaussian_kernel(1, 2) << endl;
    // cout << gaussian_kernel(2, 2) << endl;
    // cout << gaussian_kernel(0, 1) << endl;
    // cout << gaussian_kernel(1, 1) << endl;
    return 0;
}