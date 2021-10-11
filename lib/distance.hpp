#pragma once

namespace ModelFitting
{
    template<typename T>
    T euclidean_distance(cv::Mat_<T>&, cv::Mat_<T>&, const int&);
    template<typename T>
    T jaccard_distance(cv::Mat_<T>&, cv::Mat_<T>&, const int&);

    template<typename T>
    T euclidean_distance(cv::Mat_<T> &point_a, cv::Mat_<T> &point_b, const int &_dim_num)
    {
        const T * point_a_ptr = reinterpret_cast<T *>(point_a.data);
        const T * point_b_ptr = reinterpret_cast<T *>(point_b.data);

        T difference, distance = 0;
        for (size_t dim = 0; dim < _dim_num; ++dim)
        {
            difference = *(point_a_ptr++) - *(point_b_ptr++);
            distance += difference * difference;
        }
        distance = sqrt(distance);

        return distance;
    }

    template<typename T>
    T jaccard_distance(cv::Mat_<T> &point_a, cv::Mat_<T> &point_b, const int &_dim_num)
    {
        const T * point_a_ptr = reinterpret_cast<T *>(point_a.data);
        const T * point_b_ptr = reinterpret_cast<T *>(point_b.data);

        T minsum = 0, maxsum = 0;
        for (size_t dim = 0; dim < _dim_num; ++dim)
        {
            minsum += std::min({*(point_a_ptr++), *(point_b_ptr++)});
            maxsum += std::max({*(point_a_ptr++), *(point_b_ptr++)});
        }
        T dist = 1 - (minsum/maxsum);
        return dist;
    }
} // namespace ModelFitting