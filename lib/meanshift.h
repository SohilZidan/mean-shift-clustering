#pragma once

// Refs:
// https://www.mathworks.com/matlabcentral/fileexchange/10161-mean-shift-clustering
// https://github.com/danini/multi-x/blob/master/include/mean_shift.h

namespace ModelFitting
{
    class MeanShift
    {
    private:
        /* data */
        // double (*kernel_func)(double, double);
        // double (*distance_func)(const Vecd&, const Vecd&);
        // distance function pointer
        double (*calculate_distance)(cv::Mat &, cv::Mat &, const int &);
        // kernel pointer
        double (*calculate_kernel)(const double &, const double &);
        //
        int max_iterations;
		int max_inner_iterations;
    public:
        MeanShift(/* args */);
        MeanShift(
            double (*_distance_metirc)(cv::Mat &, cv::Mat &, const int &) ,
            double (*_kernel)(const double &, const double &));
        ~MeanShift();

        // void set_kernel(double (*_kernel_func)(double,double));
        // void set_distancefunc(double (*_dist_func)(const Vecd&, const Vecd&));
        // MatXd meanshift(const MatXd& points, double kernel_bandwidth);
        
        // Vecd shift(const Vecd&, const MatXd&, double);
        // std::vector<Cluster> labelling(const MatXd&, const MatXd&);
        void cluster(
            cv::InputArray _data_pts, 
            cv::OutputArray _clusters_centers, 
            std::vector<std::vector<int>> &_cluster_pts,
            double kernel_bandwidth);

        void meanshift(
            const cv::Mat &data_pts,
            const int pts_num,
            const int dim_num,
            cv::Mat &old_mean, 
            cv::Mat &new_mean,
            cv::Mat &current_cluster_votes,
            std::vector<int> &current_culster_members,
            std::vector<int> &been_visited,
            const double kernel_bandwidth);//,
            // int &window_members_num);


    };


    
} // namespace ModelFitting
