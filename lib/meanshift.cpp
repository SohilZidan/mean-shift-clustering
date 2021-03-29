
// #include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <vector>
#include "kernel.hpp"
#include "distance.hpp"
#include "meanshift.hpp"


using namespace ModelFitting;

// #define EPSILON 0.00000001
// #define CLUSTER_EPSILON 0.5

MeanShift::MeanShift(){}

MeanShift::MeanShift(
    double (*_distance_metirc)(cv::Mat &, cv::Mat &, const int &) ,
    double (*_kernel)(const double &, const double &))
{
    this->calculate_distance = _distance_metirc;
    this->calculate_kernel = _kernel;
}

MeanShift::~MeanShift()
{
}

void MeanShift::cluster(
    cv::InputArray _data_pts, 
    cv::OutputArray _clusters_centers, 
    cv::OutputArray _cluster_pts,
    double kernel_bandwidth)
{
    // initialization
    const cv::Mat data_pts = _data_pts.getMat();
    const int dim_num = data_pts.cols;
    const int pts_num = data_pts.rows;
    int cluster_num = 0;
    const double bandwidth_sq = kernel_bandwidth * kernel_bandwidth;

    std::vector<int> init_pts_inds(pts_num); // initial points indices
    std::iota(init_pts_inds.begin(), init_pts_inds.end(), 1);

    double stop_threshold = 1e-3 * kernel_bandwidth; // need casting

    std::vector<int> been_visited(pts_num,0);
    std::vector<int> cluster_votes;//(numPts,0);
    std::vector<int> current_cluster_votes;
    std::vector<cv::Mat> clusters_centers;

    cv::RNG gen;
    int tmp_idx, pt_idx;
    cv::Mat current_mean;
    std::vector<int> current_cluster_members(0);

    // Iterate over all data points
    while (init_pts_inds.size())// && num pf iterations)
    {
        tmp_idx = gen.uniform(0,init_pts_inds.size());
        pt_idx = init_pts_inds[tmp_idx];
        //
        current_mean = data_pts.row(pt_idx);
        current_cluster_members.resize(0);
        current_cluster_votes.resize(pts_num, 0);
        //

        while(true)
        {
            // mean shift procedure
            // 1. Compute the mean shift vector m
            // 2. translate the old mean
            // 1.1 iterate over all points and find indices of points 
            //     that its distances from the current mean is less than the bandwidth
            // 1.2 vote for those points as clusters
            // 1.3 pass those points ||(current_mean - pts_i)/bandwidth|| to the kernel
            // 1.4 sum the return values of the kernel
            // meanshift(
            //     const cv::Mat &data_pts,
            //     const int pts_num,
            //     const int dim_num,
            //     cv::Mat &old_mean, 
            //     cv::Mat &new_mean,
            //     std::vector<int> &current_cluster_votes,
            //     std::vector<int> &current_culster_members,
            //     std::vector<int> &been_visited,
            //     const double kernel_bandwidth,
            //     int &window_members_num);

        }
        break;
    }
    
    

    return;
}


void MeanShift::meanshift(
    const cv::Mat &_data_pts,
    const int _pts_num,
    const int _dim_num,
    cv::Mat &_old_mean, 
    cv::Mat &_new_mean,
    std::vector<int> &_current_cluster_votes,
    std::vector<int> &_current_culster_members,
    std::vector<int> &_been_visited,
    const double _kernel_bandwidth)
    // int &_window_members_num)
{
    // mean shift procedure
    // 1. Compute the mean shift vector m
    // 2. translate the old mean
    // 1.1 iterate over all points and find indices of points 
    //     that its distances from the current mean is less than the bandwidth
    double distance_val = 0;
    double kernel_val = 0;
    double normalizer = 0;
    cv::Mat current_pt;
    for (size_t idx = 0; idx < _pts_num; idx++)
    {
        // calculate the distance
        current_pt = _data_pts.row(idx);
        distance_val = calculate_distance(_old_mean, current_pt, _dim_num);
        
        if(distance_val <= _kernel_bandwidth)
        {
        // 1.2 vote for those points as clusters
            ++_current_cluster_votes.at(idx);
        // 1.3 pass those points ||(current_mean - pts_i)/bandwidth|| to the kernel
            kernel_val = calculate_kernel(distance_val, _kernel_bandwidth);
            //_window_members_num++;
        // 1.4 sum the return values of the kernel
            _new_mean += _data_pts.row(idx) * kernel_val;
            normalizer += kernel_val;
            _been_visited.at(idx) = 1;
            _current_culster_members.push_back(idx);

        }
        idx++;
        
    }
    

    return;
}


// void MeanShift::set_kernel( double (*_kernel_func)(double,double) ) {
//     if(!_kernel_func){
//         kernel_func = ModelFitting::gaussian_kernel;
//     } else {
//         kernel_func = _kernel_func;    
//     }
// }

// void MeanShift::set_distancefunc(double (*_dist_func)(const Vecd&, const Vecd&))
// {
//     if(!_dist_func){
//         distance_func = ModelFitting::euclidean_distance;
//     } else {
//         distance_func = _dist_func;    
//     }
// }


// Vecd MeanShift::shift(
//     const Vecd& point, 
//     const MatXd& points, 
//     double kernel_bandwidth)
// {
//     Vecd shifted_point = Vecd::Zero(point.rows());
//     double total_weight = 0.0;
//     //
//     for(int i=0; i<points.rows(); i++){
//         Vecd temp_point = points.row(i);
//         double distance = euclidean_distance(point, temp_point);
//         double weight = kernel_func(distance, kernel_bandwidth);
//         //
//         shifted_point += temp_point * weight;
//         total_weight += weight;
//     }
//     //
//     shifted_point /= total_weight;
//     return point;
// }


// MatXd MeanShift::meanshift(
//     const MatXd& points,
    
//     double kernel_bandwidth)
// {
//     VecXb stop_moving = VecXb::Constant(points.rows(), false);

//     MatXd shifted_points = points;

//     double max_shift_distance;
//     do
//     {
//         max_shift_distance = 0;
//         for (size_t i = 0; i < shifted_points.rows(); i++)
//         {
//             if (!stop_moving(i))
//             {
//                 Vecd point_new = shift(shifted_points.row(i), points, kernel_bandwidth);
//                 double shift_distance = distance_func(point_new, shifted_points.row(i));
//                 if(shift_distance > max_shift_distance){
//                     max_shift_distance = shift_distance;
//                 }
//                 if(shift_distance <= EPSILON) {
//                     stop_moving(i) = true;
//                 }
//                 shifted_points.row(i) = point_new;
//                 /* code */
//             }
            
//             /* code */
//         }
        
//         /* code */
//     } while (max_shift_distance);
    
    
//     // stop_moving = false;
//     return shifted_points;
// }


// std::vector<Cluster> MeanShift::labelling(
//     const MatXd& points, 
//     const MatXd& shifted_points)
// {
//     std::vector<Cluster> modes;
//     // MatXd modes(0,points.cols());

//     for (int i = 0; i < shifted_points.rows(); i++) {

//         int c = 0;
//         for (; c < modes.size(); c++) {
//             if (distance_func(shifted_points.row(i), modes[c].mode) <= CLUSTER_EPSILON) {
//                 break;
//             }
//         }

//         if (c == modes.size()) {
//             // modes.conservativeResize(modes.rows()+1,Eigen::NoChange);
//             // modes.row(modes.rows()-1) = shifted_points.row(i)
//             Cluster m;
//             m.mode = shifted_points.row(i);
//             modes.push_back(m);
//         }

//         modes[c].data_points.conservativeResize(modes[c].data_points.rows()+1,Eigen::NoChange);
//         modes[c].data_points.row(modes[c].data_points.rows()-1) = points.row(i);
//         // push_back();
//         // clusters[c].shifted_points.push_back(shifted_points[i]);
//     }

//     return modes;
// }

