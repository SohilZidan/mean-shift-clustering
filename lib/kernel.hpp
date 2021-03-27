#pragma once

namespace ModelFitting
{
    double gaussian_kernel(const double &distance, double &kernel_bandwidth);
    double uniform_kernel(const double &distance, double &kernel_bandwidth);
    
}