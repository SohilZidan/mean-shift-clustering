
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
    cv::Mat current_mean, old_mean;
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
            old_mean = current_mean.clone();
            current_mean = cv::Mat::zeros(1, data_pts.cols, data_pts.type());

            meanshift(
                data_pts,
                pts_num,
                dim_num,
                old_mean, 
                current_mean,
                current_cluster_votes,
                current_cluster_members,
                been_visited,
                kernel_bandwidth);

            
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
{
    // mean shift procedure
    // 1. Compute the mean shift vector m
    // 2. translate the old mean
    // 1.1 iterate over all points and find indices of points 
    //     that its distances from the current mean is less than the bandwidth
    double distance_val = 0;
    double kernel_val = 0;
    double normalizer = 0;
    int window_members_num = 0;
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
            ++window_members_num;
        // 1.4 sum the return values of the kernel
            _new_mean += _data_pts.row(idx) * kernel_val;
            normalizer += kernel_val;
            _been_visited.at(idx) = 1;
            _current_culster_members.push_back(idx);
            
        }
        idx++;
        
    }
    if(window_members_num == 0)
    {
        _new_mean = _old_mean;
        return;
    }
    _new_mean /= normalizer;
}