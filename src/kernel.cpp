#include <math.h>
#include "kernel.hpp"

double ModelFitting::gaussian_kernel(double distance, double kernel_bandwidth){
    double temp =  std::exp(-1.0/2.0 * (distance*distance) / (kernel_bandwidth*kernel_bandwidth));
    return temp;
}