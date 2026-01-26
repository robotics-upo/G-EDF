#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include "ceres/ceres.h"

#define PARAMS_PER_GAUSSIAN 7

/// Sample point with 3D position and SDF value
struct GMMData {
    double x, y, z;
    double d;
};

/// Dynamic cost function for GMM SDF fitting (supports variable gaussian count)
class DynamicGMMCostFunction : public ceres::CostFunction {
public:
    DynamicGMMCostFunction(double x, double y, double z, double d, double weight_scale, int num_gaussians, bool use_importance)
        : x_(x), y_(y), z_(z), d_(d), w_scale_(weight_scale), num_gaussians_(num_gaussians), use_importance_(use_importance) {
        // Set residual and parameter block sizes dynamically
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(num_gaussians * PARAMS_PER_GAUSSIAN);
    }

    virtual ~DynamicGMMCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        const double* params = parameters[0];
        const int n_params = num_gaussians_ * PARAMS_PER_GAUSSIAN;
        double sdf_pred = 0.0;
        constexpr double eps = 1e-9;

        double* jacobian = (jacobians != nullptr && jacobians[0] != nullptr) ? jacobians[0] : nullptr;
        if (jacobian)
            std::fill(jacobian, jacobian + n_params, 0.0);

        for (int i = 0; i < num_gaussians_; ++i) {
            int j = i * PARAMS_PER_GAUSSIAN;

            double mx = params[j + 0];
            double my = params[j + 1];
            double mz = params[j + 2];

            double p_l00 = params[j + 3];
            double inv_l00 = 1.0 / (p_l00 * p_l00 + eps);

            double p_l11 = params[j + 4];
            double inv_l11 = 1.0 / (p_l11 * p_l11 + eps);

            double p_l22 = params[j + 5];
            double inv_l22 = 1.0 / (p_l22 * p_l22 + eps);

            double w = params[j + 6];

            double vx = x_ - mx;
            double vy = y_ - my;
            double vz = z_ - mz;

            double z0 = vx * inv_l00;
            double z1 = vy * inv_l11;
            double z2 = vz * inv_l22;

            double dist_sq = z0 * z0 + z1 * z1 + z2 * z2;
            double exp_val = std::exp(-0.5 * dist_sq);
            double term = w * exp_val;

            sdf_pred += term;

            // Jacobians
            if (jacobian) {
                double alpha = term * (-0.5);
                double pre_geom = -2.0 * alpha * w_scale_;

                jacobian[j + 6] = exp_val * w_scale_;

                double a0 = z0 * inv_l00;
                double a1 = z1 * inv_l11;
                double a2 = z2 * inv_l22;

                jacobian[j + 0] = pre_geom * a0;
                jacobian[j + 1] = pre_geom * a1;
                jacobian[j + 2] = pre_geom * a2;

                jacobian[j + 3] = (pre_geom * a0 * z0) * (2.0 * p_l00);
                jacobian[j + 4] = (pre_geom * a1 * z1) * (2.0 * p_l11);
                jacobian[j + 5] = (pre_geom * a2 * z2) * (2.0 * p_l22);
            }
        }

        double importance = 1.0;
        if (use_importance_) {
            importance = std::exp(-5.0 * std::abs(d_));
        }
        residuals[0] = (sdf_pred - d_) * w_scale_ * importance;

        if (jacobian) {
            for (int k = 0; k < n_params; ++k)
                jacobian[k] *= importance;
        }

        return true;
    }

private:
    const double x_, y_, z_, d_, w_scale_;
    const int num_gaussians_;
    const bool use_importance_;
};

/// Solver configuration
struct SolverConfig {
    int max_iterations = 150;
    double max_time_seconds = 0.0;      // 0 = unlimited
    double function_tolerance = 1e-4;
    double gradient_tolerance = 1e-4;
    double parameter_tolerance = 1e-4;
    int num_threads = 1;
    bool verbose = false;
    bool use_importance_weighting = false; // Default to true (original behavior)
};

/// GMM solver using Ceres optimizer (supports variable gaussian count)
class GMMSolver {
public:
    GMMSolver() = default;
    explicit GMMSolver(const SolverConfig& cfg) : config_(cfg) {}

    void setConfig(const SolverConfig& cfg) { config_ = cfg; }
    SolverConfig& config() { return config_; }

    bool solve(std::vector<GMMData>& data, std::vector<double>& params) {
        if (data.empty() || params.empty()) return false;

        int num_gaussians = params.size() / PARAMS_PER_GAUSSIAN;
        if (num_gaussians <= 0) return false;

        double* x = params.data();
        ceres::Problem problem;

        for (const auto& d : data) {
            auto* cost = new DynamicGMMCostFunction(d.x, d.y, d.z, d.d, 1.0, num_gaussians, config_.use_importance_weighting);
            problem.AddResidualBlock(cost, nullptr, x);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = config_.verbose;
        options.max_num_iterations = config_.max_iterations;
        options.num_threads = config_.num_threads;
        options.function_tolerance = config_.function_tolerance;
        options.gradient_tolerance = config_.gradient_tolerance;
        options.parameter_tolerance = config_.parameter_tolerance;

        if (config_.max_time_seconds > 0.0)
            options.max_solver_time_in_seconds = config_.max_time_seconds;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        return summary.IsSolutionUsable();
    }

private:
    SolverConfig config_;
};

#endif // SOLVER_HPP