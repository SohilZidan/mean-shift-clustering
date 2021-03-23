#include <math.h>
#include <Eigen/Dense>
#include "distance.hpp"


double ModelFitting::euclidean_distance(const Vecd &point_a, const Vecd &point_b){
    
    Vec3d tmp = (point_a - point_b);
    double total = tmp.transpose() * tmp;
    return sqrt(total);
}

double ModelFitting::jaccard_distance(const Vecd &point_a, const Vecd &point_b){
    double minsum = 0, maxsum = 0;
    for(size_t i = 0; i < point_a.rows(); i++){
        minsum += std::min({point_a(i), point_b(i)});
        maxsum += std::max({point_a(i), point_b(i)});
    }
    double dist = 1 - (minsum/maxsum);
    return dist;
}
