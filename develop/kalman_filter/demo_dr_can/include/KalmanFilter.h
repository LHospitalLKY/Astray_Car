#ifndef KF_H
#define KF_H

#define DEBUG

#include <iostream>
#include <thread>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector> 
#include <glog/logging.h>

#define SxSMatrix Eigen::Matrix<double, states_num, states_num>
#define OxOMatrix Eigen::Matrix<double, obvs_num, obvs_num>

#define States_StdVector(states_num) \
std::vector<Eigen::Matrix<double, states_num, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, states_num, 1>>>
#define Obvs_StdVector(obvs_num) \
std::vector<Eigen::Matrix<double, obvs_num, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, obvs_num, 1>>>

template <int states_num, int obvs_num>
class KalmanFilter
{
private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /* data */
    // states_num & observation num
    int _states_num;
    int _obvs_num; 
    
    // Kalman推导所需要的矩阵
    // state matrix
    Eigen::Matrix<double, states_num, states_num> state_M_;
    // observe matrix
    Eigen::Matrix<double, obvs_num, states_num> observe_M_;
    // covariance-random error of states
    Eigen::Matrix<double, states_num, states_num> Q_;
    // covariance-random error of observation
    Eigen::Matrix<double, obvs_num, obvs_num> R_;

    // Kalman中要更新的举证和状态
    // prior states estimate
    Eigen::Matrix<double, states_num, 1> x_prior_;
    // states estimate
    Eigen::Matrix<double, states_num, 1> x_;
    // Kalman Gain
    Eigen::Matrix<double, states_num, obvs_num> kalman_G_;
    // covariance matrix
    Eigen::Matrix<double, states_num, states_num> P_;

    // 初始化flag
    bool _init_FLAG_ = false;


public:
    KalmanFilter();
    KalmanFilter(
        SxSMatrix& states_Matrix, 
        OxOMatrix& obvs_Matrix,
        SxSMatrix& states_cov_Matrix,
        OxOMatrix& obvs_cov_Matrix
    );
    ~KalmanFilter();

    bool initial(
        SxSMatrix& states_Matrix, 
        OxOMatrix& obvs_Matrix,
        SxSMatrix& states_cov_Matrix,
        OxOMatrix& obvs_cov_Matrix
    );

    // 根据当前的观测，通过kalman filter计算得到当前的状态
    static bool estimate_CallBack(
        Eigen::Matrix<double, states_num, 1>& state,           // 状态
        const Eigen::Matrix<double, obvs_num, 1>& observe,     // 观测
        void* context
    );

    // ! 这个函数用来测试 kalman filter是否能不能用
    bool estimate_test(
        const Obvs_StdVector(obvs_num)& observes, 
        const Eigen::Matrix<double, states_num, 1>& start_states,
        States_StdVector(states_num)& states 
    );

private:
    /*初始化functions*/
    bool _initial_(
        SxSMatrix& states_Matrix, 
        OxOMatrix& obvs_Matrix,
        SxSMatrix& states_cov_Matrix,
        OxOMatrix& obvs_cov_Matrix
    );
    bool _init_State_M(const Eigen::Matrix<double, states_num, states_num>& states_matrix);
    bool _init_Observe_M(const Eigen::Matrix<double, obvs_num, obvs_num>& observe_matrix);
    bool _init_Q(const Eigen::Matrix<double, states_num, states_num>& state_cov_Matrix);
    bool _init_R(const Eigen::Matrix<double, obvs_num, obvs_num>& obvs_cov_Matrix);
};


#endif // KF_H