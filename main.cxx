#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>
// #include <Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <opencv2/matlab/bridge.hpp>
// #include <opencv2/matlab/bridge.hpp>
// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
#include "distance.h"
#include "kernel.h"
#include "meanshift.h"
// #include "kernel.hpp"
// #include "meanshift.hpp"

using namespace std;
using namespace ModelFitting;
using namespace cv;


// int main1(int argc, char* argv[])
// {

//     cout<< "Welcome Mean Shift"<<endl;
//     dist_calc = &euclidean_distance<double>;
//     testClass<double> t;
//     t.dist_calc = &euclidean_distance;
//     // Vec3d p_a, p_b;
//     // p_a << 2,3,4;
//     // p_b << 0,0,2;
//     // cout << euclidean_distance(p_a, p_b) << endl;
//     // cout << jaccard_distance(p_a, p_b) << endl;

//     // p_a << 2,3,6;
//     // p_b << 0,0,4;
//     // cout << euclidean_distance(p_a, p_b) << endl;
//     // cout << jaccard_distance(p_a, p_b) << endl;

//     // p_a << 2,3,16;
//     // p_b << 0,0,14;
//     // cout << euclidean_distance(p_a, p_b) << endl;
//     // cout << jaccard_distance(p_a, p_b) << endl;

//     // MatXd x(3,1), y;
//     // MatXd t;
//     // x << 1, 2, 3;
//     // y = x;
//     // y(1) = 18;
//     // cout << t.rows() << endl;
//     // t.conservativeResize(1,1);
//     // cout << t.rows() << endl;
//     // cout << t.cols() << endl;
//     // t.conservativeResize(2,Eigen::NoChange);
//     // cout << t.rows() << endl;
//     // cout << t.cols() << endl;
//     cout << "number of threads: "<< Eigen::nbThreads( ) << endl;

//     cv::Mat A = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0);

//     // cv::Mat_<double> const C = A;
//     // cout << &C << endl;
//     cv::Mat D = (cv::Mat_<double>(3,1) << 0.0, 1.0, 0.0);
//     cout << euclidean_distance<double>(A,D,3) << endl;
//     D.row(2) = 2;
//     cout << t.dist_calc(A,D,3) << endl;

//     cv::Mat E = (cv::Mat_<double>(3,2) << 0.0, 1.0, 0.0,
//                                           1.0, 0.0 ,1.0);

//     cout << "E: " << E << endl;
//     cout << "&E: " << &E << endl;
    
//     cv::Mat E0;
//     cout << "&E0: " << &E0 << endl;
//     E0 = E.row(0);
//     cout << "E.row(0): " << E0 << endl;
//     cout << "&E.row(0): " << &E0 << endl;
//     const double * Ep = reinterpret_cast<double *>(E.data);
//     cout << "Ep: " << Ep << endl;
//     cout << "*Ep: " << *Ep << endl;
//     cout << "Ep+1: " << (Ep+1) << endl;
//     cout << "*Ep+1: " << *(++Ep) << endl;

//     ///
//     ///
//     ///
//     std::vector<int> v = {0,0,0,1,0,1,0, 1,0,1};
//     // v.erase(std::remove(v.begin(), v.end(), 0), v.end());
//     for(auto ele:v)
//         cout << ele << ", ";
//     cout << endl;

//     std::vector<int> init_pts_inds(10); // initial points indices
//     std::iota(init_pts_inds.begin(), init_pts_inds.end(), 0);
//     for(auto ele:init_pts_inds)
//         cout << ele << ", ";
//     cout << endl;
//     init_pts_inds.erase(
//         std::remove_if(
//         init_pts_inds.begin(),
//         init_pts_inds.end(),
//         [v](int x){return v.at(x) == 1;}
//     ),
//     init_pts_inds.end());
//     // init_pts_inds.erase()

//     for(auto ele:init_pts_inds)
//         cout << ele << ", ";
//     cout << endl;

//     // std::vector<std::vector<int>> a;
//     // test_array(a);
    
//     // init_pts_inds.data();
//     cv::Mat t1 = (cv::Mat_<double>(1,3) << 1.0, 2.0, 3.0);
//     cv::Mat t2 = (cv::Mat_<double>(1,3) << 0.0, 1.0, 0.0);
//     std::vector<cv::Mat> v1 = {t1,t2};

//     cv::Mat m = cv::Mat_<double>(v1.size(),3);

//     std::memcpy(m.data, v1.data(), v1.size() * sizeof(v1.at(0)));
//     cout << m << endl;
    
//     // C = &D;//(cv::Mat_<double>(3,1) << 0.0, 1.0, 1.0);
//     // cout << &C << endl;
//     // C(0,0) = 5;
//     // int const dim = 3;
//     // dim = 9;
//     // cv::Mat E = C - D;
//     // cout << static_cast<double>(cv::Mat_<double>((D - C).t() * (D - C))) << endl;
//     // cout << (E.t() * E).type() << endl;
    
//     test_kernels();
//     return 0;
// }


// Testing meanshift
int main(int argc, char* argv[])
{
    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);

    int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
    int i, sampleCount = rng.uniform(1, 1001);
    Mat points(sampleCount, 1, CV_32FC2), labels;
    clusterCount = MIN(clusterCount, sampleCount);
    std::vector<Point2f> centers;
    /* generate random sample from multigaussian distribution */
    for( k = 0; k < clusterCount; k++ )
    {
        Point center;
        center.x = rng.uniform(0, img.cols);
        center.y = rng.uniform(0, img.rows);
        Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                            k == clusterCount - 1 ? sampleCount :
                                            (k+1)*sampleCount/clusterCount);
        rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
    }
    randShuffle(points, 1, &rng);

    
    Mat_<double>
        in_ms = Mat_<double>(points.rows, 2);
    Mat cluster_centers;
    std::vector<std::vector<int>> cluster_points;
    double bandwidth = 40;

    MeanShift alg(jaccard_distance<double>, uniform_kernel);
    
    for (size_t i = 0; i < points.rows; i++)
    {
        Point2d ipt = static_cast<Point2d>(points.at<Point2f>(i));
        cout << ipt << endl;
        in_ms(i,0) = static_cast<double>(ipt.x);
        in_ms(i,1) = static_cast<double>(ipt.y);
        cout << in_ms.row(i) << endl;
        circle( img, ipt, 2, colorTab[0], FILLED, LINE_AA );
    }
    
    alg.cluster(
        in_ms,
        cluster_centers,
        cluster_points,
        bandwidth);
    cout << cluster_centers.size() << endl;
    // cout << cluster_centers << endl;
    for (size_t i = 0; i < cluster_centers.rows; i++)
    {
        Point ipt(cluster_centers.at<double>(i, 0), cluster_centers.at<double>(i, 1));
        // in_ms.row(i) = (Mat_<double>(1,2) << ipt.x, ipt.y);
        
        circle( img, ipt, 20, colorTab[1], 1, LINE_AA );
    }
    imshow("clusters", img);

    

    waitKey(0);
    
}
// {
//     cv::Mat img(100, 100, CV_8UC3);
//     std::random_device rd{};
//     std::mt19937 gen{rd()};

//     cv::randn(img, 25, 20);
//     cv::randn(img, 75, 20);


//     cv::namedWindow("random colors image", cv::WINDOW_FREERATIO );
//     cv::imshow("random colors image", img);
//     // cv::namedWindow("Disparity - Naive", cv::WINDOW_AUTOSIZE);
//     // cv::imshow("Disparity - Naive", naive_disparities);

//     cv::waitKey(0);
//     cout << "welcome" << endl;
//     return 0;
// }