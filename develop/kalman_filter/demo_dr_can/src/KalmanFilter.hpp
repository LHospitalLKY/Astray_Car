#include <../include/KalmanFilter.h>

void create_Kalman(){}


template<int states_num, int obvs_num>
KalmanFilter<states_num, obvs_num>::KalmanFilter() {
    _states_num = states_num;
    _obvs_num = obvs_num;

    // initial matrices in states & observation Function
    state_M_ = Eigen::Matrix<double, states_num, states_num>::Identity();
    observe_M_ = Eigen::Matrix<double, obvs_num, obvs_num>::Identity();
    Q_ = Eigen::Matrix<double, states_num, states_num>::Identity();
    R_ = Eigen::Matrix<double, obvs_num, obvs_num>::Identity();

    // initial matrices/vectors in kalman process
    kalman_G_ = Eigen::Matrix<double, states_num, obvs_num>::Identity();
    P_ = Eigen::Matrix<double, states_num, states_num>::Identity();

#ifdef DEBUG
    LOG(INFO) << "\nStates Matrix:\n" << state_M_;
    LOG(INFO) << "\nObservation Matrix:\n" << observe_M_;
    LOG(INFO) << "\nConvariance Matrix Q:\n" << Q_;
    LOG(INFO) << "\nConvariance Matrix R:\n" << R_;
    LOG(INFO) << "\nKalman Gain:\n" << kalman_G_;
    LOG(INFO) << "\nP:\n" << P_;
#endif // DEBUG

    _init_FLAG_ = false;
}

template<int states_num, int obvs_num>
KalmanFilter<states_num, obvs_num>::KalmanFilter(
        SxSMatrix& states_Matrix, 
        OxOMatrix& obvs_Matrix,
        SxSMatrix& states_cov_Matrix,
        OxOMatrix& obvs_cov_Matrix
) {
    _states_num = states_num;
    _obvs_num = obvs_num;

    bool succ =  _initial_(states_Matrix, obvs_Matrix, states_cov_Matrix, obvs_cov_Matrix);
    bool succ_1;
    succ_1 = succ;
    _init_FLAG_ = succ;
    std::cout << "Initial Flags" << succ << std::endl;
    std::cout << "Initial Flags" << _init_FLAG_ << std::endl;
    int a = 0, b = 3;
    int c;
    c = a + b;

#ifdef DEBUG
    LOG(INFO) << "\nStates Matrix:\n" << state_M_;
    LOG(INFO) << "\nObservation Matrix:\n" << observe_M_;
    LOG(INFO) << "\nConvariance Matrix Q:\n" << Q_;
    LOG(INFO) << "\nConvariance Matrix R:\n" << R_;
    LOG(INFO) << "\nKalman Gain:\n" << kalman_G_;
    LOG(INFO) << "\nP:\n" << P_;
#endif // DEBUG

    LOG_IF(ERROR, _init_FLAG_ == false) << "Kalman Filter cannot initialize ";
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::initial(
        SxSMatrix& states_Matrix, 
        OxOMatrix& obvs_Matrix,
        SxSMatrix& states_cov_Matrix,
        OxOMatrix& obvs_cov_Matrix
) {
    bool succ = _initial_(states_Matrix, obvs_Matrix, states_cov_Matrix, obvs_cov_Matrix);
    return succ;
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::_initial_(
    SxSMatrix& states_Matrix, 
    OxOMatrix& obvs_Matrix,
    SxSMatrix& states_cov_Matrix,
    OxOMatrix& obvs_cov_Matrix
) {
    // initial matrices/vectors in kalman process
    kalman_G_ = Eigen::Matrix<double, states_num, obvs_num>::Identity();
    P_ = Eigen::Matrix<double, states_num, states_num>::Identity();

    // initial kalman filter start states
    bool succ_states_M, succ_obvs_M, succ_Q, succ_R;
    succ_states_M = _init_State_M(states_Matrix);
    succ_obvs_M = _init_Observe_M(obvs_Matrix);
    succ_Q = _init_Q(states_cov_Matrix);
    succ_R = _init_R(obvs_cov_Matrix);

#ifdef DEBUG
    LOG_IF(ERROR, succ_states_M == false) << "states transfer matrix cannot initial correctly";
    LOG_IF(ERROR, succ_obvs_M == false) << " observation matrix cannot initial correctly";
    LOG_IF(ERROR, succ_Q == false) << "convariance matrix of states random error cannot initial correctly";
    LOG_IF(ERROR, succ_R == false) << "convariance matrix of observation random error cannot initial correctly";
#endif // DEBUG

    if(succ_obvs_M && succ_obvs_M && succ_Q && succ_R) {
        return true;
    } else {
        return false;
    }
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::_init_State_M(const Eigen::Matrix<double, states_num, states_num>& states_matrix) {
    // TODO: 通过写catch来返回true或false
    state_M_ = states_matrix;
    return true;
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::_init_Observe_M(const Eigen::Matrix<double, obvs_num, obvs_num>& observe_matrix) {
    // TODO: 通过写catch来返回true或false
    observe_M_ = observe_matrix;
    return true;
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::_init_Q(const Eigen::Matrix<double, states_num, states_num>& states_cov_matrix) {
    // TODO: 通过写catch来返回true或false
    Q_ = states_cov_matrix;
    return true;
}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::_init_R(const Eigen::Matrix<double, obvs_num, obvs_num>& obvs_cov_matrix) {
    // TODO: 通过写catch来返回true或false
    R_ = obvs_cov_matrix;
    return true;
}

// FIXME: 补充完成回调
template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::estimate_CallBack(
    Eigen::Matrix<double, states_num, 1>& state, 
    const Eigen::Matrix<double, obvs_num, 1>& observe, 
    void* context
) {
#ifdef DEBUG
    LOG(INFO) << "Start Kalman Filter Callback";
#endif // DEBUG
    KalmanFilter *thisKF = (KalmanFilter*) context;
    if(thisKF) {
        return true;
    }
    
    return false;

}

template<int states_num, int obvs_num>
bool KalmanFilter<states_num, obvs_num>::estimate_test(
    const Obvs_StdVector(obvs_num)& observes, 
    const Eigen::Matrix<double, states_num, 1>& start_states,
    States_StdVector(states_num)& states
) {
    if(_init_FLAG_ == false) {
        LOG(ERROR) << "Kalman Filter has not been initialized!";
        return false;
    }
    // 将状态估计设置为初始状态
    Eigen::Matrix<double, states_num, 1> state = start_states;

    // 中间值
    Eigen::Matrix<double, states_num, 1> state_per;
    Eigen::Matrix<double, states_num, states_num> P_per;
    Eigen::Matrix<double, states_num, 1> state_esti;
    Eigen::Matrix<double, states_num, states_num> P_esti;
    Eigen::Matrix<double, states_num, states_num> I = Eigen::Matrix<double, states_num, states_num>::Identity();
    
    int size = observes.size();
    P_esti = P_;
    for(size_t i = 0; i < size; i++) {
        // 1. 计算先验估计
        state_per = state_M_ * state;

        // 2. 计算先验误差协方差矩阵
        P_per = state_M_ * P_esti * state_M_.transpose() + Q_;
        std::cout << state_M_ << "\n " << std::endl;
        std::cout << state_M_.transpose() << "\n " << std::endl;
        std::cout << state_M_ * P_esti * (state_M_.transpose()) << std::endl;
        
        // 3. 计算卡尔曼增益
        // TODO: 考虑这里要不要写成广义逆
        kalman_G_ = P_per * observe_M_.transpose() * (observe_M_ * P_per * observe_M_.transpose() + R_).inverse();

        // 4. 计算后验估计
        Eigen::Matrix<double, obvs_num, 1> observation(observes[i]);
        // std::cout << "---------" << std::endl;
        // std::cout << observation << "\n" << std::endl;
        // std::cout << observe_M_ << "\n" << std::endl;
        // std::cout << state_per <<"\n" << std::endl;
        // std::cout << observe_M_*state_per << std::endl;

        state_esti = state_per + kalman_G_ * (observation - observe_M_ * (state_per));

        // 5. 计算状态误差的协方差矩阵
        P_esti = (I - kalman_G_ * observe_M_) * P_per;

#ifdef DEBUG
        LOG(INFO) << "\n---------------" 
                  << "State prior estimate:\n" 
                  << state_per
                  << "\nP prior estimate:\n" 
                  << P_per
                  << "\nKalman Gain:\n" 
                  << kalman_G_
                  << "\nState estimate:\n" 
                  << state_esti
                  << "\nP estimate:\n" 
                  << P_esti << "\n";
#endif // DEBUG

        // 6. 更新值
        state = state_esti;

        // 7. 将state写入队列中
        states.push_back(state);
    }


    return true;
}

template<int states_num, int obvs_num>
KalmanFilter<states_num, obvs_num>::~KalmanFilter() {

}