#include <math.h>
#include "kernel.hpp"

double ModelFitting::gaussian_kernel(double distance, double kernel_bandwidth){
    // TODO: check what is the output according to the distance
    double temp = 0;
    if(distance <= kernel_bandwidth)
        temp =  std::exp((distance*distance) / (kernel_bandwidth*kernel_bandwidth));
    return temp;
}

double ModelFitting::uniform_kernel(double distance, double kernel_bandwidth){
    double temp = 0;
    if(distance <= kernel_bandwidth)
        temp =  static_cast<double>(1/(2*kernel_bandwidth));
    return temp;
}