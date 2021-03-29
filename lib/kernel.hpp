#pragma once

namespace ModelFitting
{
    double gaussian_kernel(const double &distance, const double &kernel_bandwidth);
    double uniform_kernel(const double &distance, const double &kernel_bandwidth);
    
}