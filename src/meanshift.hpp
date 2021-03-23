#pragma once
#include <vector>

namespace ModelFitting
{
    struct Cluster {
        Vecd mode;
        MatXd data_points;
    };
    
    class MeanShift
    {
    private:
        /* data */
        double (*kernel_func)(double, double);
        double (*distance_func)(const Vecd&, const Vecd&);
    public:
        MeanShift(/* args */);
        ~MeanShift();

        void set_kernel(double (*_kernel_func)(double,double));
        void set_distancefunc(double (*_dist_func)(const Vecd&, const Vecd&));
        MatXd meanshift(const MatXd& points, double kernel_bandwidth);
        Vecd shift(const Vecd&, const MatXd&, double);
        std::vector<Cluster> labelling(const MatXd&, const MatXd&);

    };


    
} // namespace ModelFitting
