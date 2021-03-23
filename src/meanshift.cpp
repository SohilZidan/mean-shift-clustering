
#include <Eigen/Dense>

#include "kernel.hpp"
#include "distance.hpp"
#include "meanshift.hpp"


using namespace ModelFitting;

#define EPSILON 0.00000001
#define CLUSTER_EPSILON 0.5

MeanShift::MeanShift(/* args */)
{
}

MeanShift::~MeanShift()
{
}


void MeanShift::set_kernel( double (*_kernel_func)(double,double) ) {
    if(!_kernel_func){
        kernel_func = ModelFitting::gaussian_kernel;
    } else {
        kernel_func = _kernel_func;    
    }
}

void MeanShift::set_distancefunc(double (*_dist_func)(const Vecd&, const Vecd&))
{
    if(!_dist_func){
        distance_func = ModelFitting::euclidean_distance;
    } else {
        distance_func = _dist_func;    
    }
}


Vecd MeanShift::shift(
    const Vecd& point, 
    const MatXd& points, 
    double kernel_bandwidth)
{
    Vecd shifted_point = Vecd::Zero(point.rows());
    double total_weight = 0.0;
    //
    for(int i=0; i<points.rows(); i++){
        Vecd temp_point = points.row(i);
        double distance = euclidean_distance(point, temp_point);
        double weight = kernel_func(distance, kernel_bandwidth);
        //
        shifted_point += temp_point * weight;
        total_weight += weight;
    }
    //
    shifted_point /= total_weight;
    return point;
}


MatXd MeanShift::meanshift(
    const MatXd& points,
    
    double kernel_bandwidth)
{
    VecXb stop_moving = VecXb::Constant(points.rows(), false);

    MatXd shifted_points = points;

    double max_shift_distance;
    do
    {
        max_shift_distance = 0;
        for (size_t i = 0; i < shifted_points.rows(); i++)
        {
            if (!stop_moving(i))
            {
                Vecd point_new = shift(shifted_points.row(i), points, kernel_bandwidth);
                double shift_distance = distance_func(point_new, shifted_points.row(i));
                if(shift_distance > max_shift_distance){
                    max_shift_distance = shift_distance;
                }
                if(shift_distance <= EPSILON) {
                    stop_moving(i) = true;
                }
                shifted_points.row(i) = point_new;
                /* code */
            }
            
            /* code */
        }
        
        /* code */
    } while (max_shift_distance);
    
    
    // stop_moving = false;
    return shifted_points;
}


std::vector<Cluster> MeanShift::labelling(
    const MatXd& points, 
    const MatXd& shifted_points)
{
    std::vector<Cluster> modes;
    // MatXd modes(0,points.cols());

    for (int i = 0; i < shifted_points.rows(); i++) {

        int c = 0;
        for (; c < modes.size(); c++) {
            if (distance_func(shifted_points.row(i), modes[c].mode) <= CLUSTER_EPSILON) {
                break;
            }
        }

        if (c == modes.size()) {
            // modes.conservativeResize(modes.rows()+1,Eigen::NoChange);
            // modes.row(modes.rows()-1) = shifted_points.row(i)
            Cluster m;
            m.mode = shifted_points.row(i);
            modes.push_back(m);
        }

        modes[c].data_points.conservativeResize(modes[c].data_points.rows()+1,Eigen::NoChange);
        modes[c].data_points.row(modes[c].data_points.rows()-1) = points.row(i);
        // push_back();
        // clusters[c].shifted_points.push_back(shifted_points[i]);
    }

    return modes;
}