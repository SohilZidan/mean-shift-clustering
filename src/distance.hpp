#pragma once

namespace ModelFitting
{
    using Vecd = Eigen::VectorXd;
    using Vec3d = Eigen::Vector3d;
    using MatXd = Eigen::MatrixXd;
    using VecXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    double euclidean_distance(const Vecd&, const Vecd&);
    double jaccard_distance(const Vecd&, const Vecd&);
} // namespace ModelFitting