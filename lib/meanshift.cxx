
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

MeanShift::MeanShift():
        max_iterations(100000),
		max_inner_iterations(10000)
{}

MeanShift::MeanShift(
    double (*_distance_metirc)(cv::Mat &, cv::Mat &, const int &) ,
    double (*_kernel)(const double &, const double &)): 
    MeanShift()
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
    std::vector<std::vector<int>> &_cluster_pts,
    double kernel_bandwidth)
{
    // initialization
    const cv::Mat data_pts = _data_pts.getMat();
    const int dim_num = data_pts.cols;
    const int pts_num = data_pts.rows;
    int cluster_num = 0;
    const double bandwidth_sq = kernel_bandwidth * kernel_bandwidth;
    const double hald_bandwith = static_cast<double>(kernel_bandwidth/2);
    const double stop_threshold = static_cast<double>(1e-3*kernel_bandwidth); 

    std::vector<int> init_pts_inds(pts_num); // initial points indices
    std::iota(init_pts_inds.begin(), init_pts_inds.end(), 0);

    // double stop_threshold = 1e-3 * kernel_bandwidth; // need casting

    std::vector<int> been_visited(pts_num,0);
    std::vector<cv::Mat> cluster_votes;//(pts_num,std::vector<int>(pts_num, 0));
    cv::Mat current_cluster_votes;
    std::vector<cv::Mat> clusters_centers;

    cv::RNG gen;
    int tmp_idx, pt_idx;
    cv::Mat current_mean, old_mean;
    std::vector<int> current_cluster_members(0);

    // Iterate over all data points
    int iters = 0;
    while (init_pts_inds.size() && iters++ < max_iterations)// && num pf iterations)
    {
        tmp_idx = gen.uniform(0,init_pts_inds.size());
        pt_idx = init_pts_inds[tmp_idx];
        //
        current_mean = data_pts.row(pt_idx);
        current_cluster_members.resize(0);
        current_cluster_votes = cv::Mat_<int>(1,pts_num, 0);//.resize(pts_num, 0);
        //
        int inner_iters = 0;
        while(inner_iters++ < max_inner_iterations)
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

            // checking stopping condition for current cluster
            if(this->calculate_distance(current_mean, old_mean, dim_num) < stop_threshold)
            {
                // check for merge posibilities
                int merge_with = -1;
                for(size_t c = 0; c < cluster_num; c++)
                {
                    // distance from posible new cluster max to old cluster max
                    double dist_to_other = this->calculate_distance(current_mean, clusters_centers.at(c), dim_num);
                    if(dist_to_other < hald_bandwith)
                    {
                        merge_with = c;
                        break;
                    }
                }

                if (merge_with > -1)
                {
                    clusters_centers.at(merge_with) = 0.5 * (current_mean + clusters_centers.at(merge_with));
                    cluster_votes.at(merge_with) += current_cluster_votes;
                }
                else
                {
                    // increment cluster numbers
                    ++cluster_num;
                    // record the current mean
                    clusters_centers.push_back(current_mean);
                    cluster_votes.push_back(current_cluster_votes);
                }
                break;
            }
        }

        // remove visited points
        init_pts_inds.erase(
            std::remove_if(
                init_pts_inds.begin(),
                init_pts_inds.end(),
                [been_visited](int pts_idx){ return been_visited.at(pts_idx) == 1;}
            ),
            init_pts_inds.end()
        );
    }
    
    // assigning points to clusters
    std::vector<int> pts_cluster_votes(pts_num, 0);
    std::vector<int> pts_cluster_idx(pts_num, -1);
    
    for (size_t pt_idx = 0; pt_idx < pts_num; pt_idx++)
    {
        for (size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++)
        {
            if(pts_cluster_votes.at(pt_idx) < cluster_votes.at(cluster_idx).at<int>(pt_idx))
            {
                pts_cluster_votes.at(pt_idx) = cluster_votes.at(cluster_idx).at<int>(pt_idx);
                pts_cluster_idx.at(pt_idx) = cluster_idx;
            }
        }
        
    }
    
    
    // refactor centers as Mat(center_num, dim_num)
    _clusters_centers.create(static_cast<int>(cluster_votes.size()), dim_num, data_pts.type());
	cv::Mat const &clusters_ref = _clusters_centers.getMatRef();
	for (auto i = 0; i < clusters_centers.size(); ++i)
		clusters_centers[i].copyTo(clusters_ref.row(i));
    

    // refactor cluster points as a vector of vectors of points indices
    _cluster_pts.resize(cluster_votes.size());
	for (size_t i = 0; i < pts_cluster_idx.size(); ++i)
	{
		if (pts_cluster_idx[i] == -1)
			continue;
		_cluster_pts[pts_cluster_idx[i]].push_back(i);
	}
    

    return;
}


void MeanShift::meanshift(
    const cv::Mat &_data_pts,
    const int _pts_num,
    const int _dim_num,
    cv::Mat &_old_mean, 
    cv::Mat &_new_mean,
    cv::Mat &_current_cluster_votes,
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
        distance_val = this->calculate_distance(_old_mean, current_pt, _dim_num);
        
        if(distance_val <= _kernel_bandwidth)
        {
        // 1.2 vote for those points as clusters
            ++_current_cluster_votes.at<int>(idx);
        // 1.3 pass those points ||(current_mean - pts_i)/bandwidth|| to the kernel
            kernel_val = this->calculate_kernel(distance_val, _kernel_bandwidth);
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