#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include "ceres/ceres.h"

// Definiciones globales
#ifndef NUM_GAUSSIANS
#define NUM_GAUSSIANS 10
#endif

#define PARAMS_PER_GAUSSIAN 7
#define N_PARAMS (NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN)

struct GMMData
{
    double x, y, z;
    double d;
};

class AnalyticGMMCostFunction : public ceres::SizedCostFunction<1, N_PARAMS>
{
public:
    AnalyticGMMCostFunction(double x, double y, double z, double d, double weight_scale)
        : x_(x), y_(y), z_(z), d_(d), w_scale_(weight_scale) {}

    virtual ~AnalyticGMMCostFunction() {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override
    {
        const double *params = parameters[0];
        double sdf_pred = 0.0;
        double eps = 1e-9;

        double *jacobian = (jacobians != NULL) ? jacobians[0] : NULL;
        if (jacobian)
            std::fill(jacobian, jacobian + N_PARAMS, 0.0);

        for (int i = 0; i < NUM_GAUSSIANS; ++i)
        {
            int j = i * PARAMS_PER_GAUSSIAN;

            // --- LEER PARÁMETROS (Solo Diagonales) ---
            double mx = params[j + 0];
            double my = params[j + 1];
            double mz = params[j + 2];

            // Diagonal X
            double p_l00 = params[j + 3];
            double inv_l00 = 1.0 / (p_l00 * p_l00 + eps);

            // Diagonal Y (Antes j+5, ahora j+4)
            double p_l11 = params[j + 4];
            double inv_l11 = 1.0 / (p_l11 * p_l11 + eps);

            // Diagonal Z (Antes j+8, ahora j+5)
            double p_l22 = params[j + 5];
            double inv_l22 = 1.0 / (p_l22 * p_l22 + eps);

            // Peso (Antes j+9, ahora j+6)
            double w = params[j + 6];

            // --- FORWARD (Simplificado, sin términos cruzados) ---
            double vx = x_ - mx;
            double vy = y_ - my;
            double vz = z_ - mz;

            double z0 = vx * inv_l00;
            double z1 = vy * inv_l11; // Ya no depende de z0 ni l10
            double z2 = vz * inv_l22; // Ya no depende de z0, z1, l20, l21

            double dist_sq = z0 * z0 + z1 * z1 + z2 * z2;
            double exp_val = std::exp(-0.5 * dist_sq);
            double term = w * exp_val;

            sdf_pred += term;

            // --- BACKWARD (Jacobianos) ---
            if (jacobian)
            {
                double alpha = term * (-0.5); 
                double pre_geom = -2.0 * alpha * w_scale_;

                // 1. Derivada del Peso (Índice j+6)
                jacobian[j + 6] = exp_val * w_scale_;

                // Vectores auxiliares simplificados
                double a0 = z0 * inv_l00;
                double a1 = z1 * inv_l11;
                double a2 = z2 * inv_l22;

                // 2. Derivadas Centros (mx, my, mz)
                jacobian[j + 0] = pre_geom * a0;
                jacobian[j + 1] = pre_geom * a1;
                jacobian[j + 2] = pre_geom * a2;

                // 3. Derivadas L (Solo diagonales: l00, l11, l22)
                // d(inv)/dp = -2 * p / (p^2)^2 -> simplificado en la lógica actual:
                // La regla de la cadena aplicada a la inversa cuadrada ya está implícita en tu lógica original
                // d(dist_sq)/d(p_l00) = ...
                
                jacobian[j + 3] = (pre_geom * a0 * z0) * (2.0 * p_l00);
                jacobian[j + 4] = (pre_geom * a1 * z1) * (2.0 * p_l11);
                jacobian[j + 5] = (pre_geom * a2 * z2) * (2.0 * p_l22);
            }
        }

        // Residual Ponderado
        double importance = std::exp(-5.0 * std::abs(d_));
        residuals[0] = (sdf_pred - d_) * w_scale_ * importance;

        if (jacobian)
        {
            for (int k = 0; k < N_PARAMS; ++k)
                jacobian[k] *= importance;
        }

        return true;
    }

private:
    const double x_, y_, z_, d_, w_scale_;
};

class GMMSolver
{
private:
    int _max_num_iterations;
    int _max_num_threads;

public:
    GMMSolver() : _max_num_iterations(150), _max_num_threads(1) {}

    void setMaxNumIterations(int n) { _max_num_iterations = n; }

    bool solve(std::vector<GMMData> &data, std::vector<double> &initial_params)
    {
        double *x = initial_params.data();
        ceres::Problem problem;

        // OPCIONAL: Si los datos son limpios, cambiar esto a NULL 
        // ahorra el cálculo de sqrt(1+x^2) en cada punto.
        // ceres::LossFunction *loss_function = new ceres::SoftLOneLoss(1.0);
        ceres::LossFunction *loss_function = nullptr;

        for (const auto &d : data)
        {
            // Usamos la clase optimizada
            ceres::CostFunction *cost = new AnalyticGMMCostFunction(d.x, d.y, d.z, d.d, 1.0);
            problem.AddResidualBlock(cost, loss_function, x);
            // problem.AddResidualBlock(cost, NULL, x);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.max_solver_time_in_seconds = 0.8;
        options.max_num_iterations = _max_num_iterations;
        options.num_threads = _max_num_threads;
        
        options.check_gradients = false; 

        options.function_tolerance = 1e-4;
        options.gradient_tolerance = 1e-4;
        options.parameter_tolerance = 1e-4;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        return true;
    }
};

#endif