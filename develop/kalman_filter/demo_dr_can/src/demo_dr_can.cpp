// LHopital
 
#include <iostream>
#include <fstream>
#include <sstream>

#include "KalmanFilter.hpp"
#include "gnuplot_i.hpp"

#include <eigen3/Eigen/Eigen>

#define GT_StdVector(states_num) States_StdVector(states_num)


template<int dimension>
void readData(const std::string& file, GT_StdVector(dimension)& datas);
template<int states_num, int obvs_num>
void readTESTData(
    const std::string& gt_file, 
    const std::string& obvs_file,
    GT_StdVector(states_num)& ground_truths,
    Obvs_StdVector(obvs_num)& observations
);

template<int states_num, int obvs_num>
void readTESTData(
    const std::string& gt_file, 
    const std::string& obvs_file,
    GT_StdVector(states_num)& ground_truths,
    Obvs_StdVector(obvs_num)& observations
) {

    LOG(INFO) << "Read Ground Truth: ";
    readData<states_num>(gt_file, ground_truths);
    
    LOG(INFO) << "Read Observations: ";
    readData<obvs_num>(obvs_file, observations);

}

// ATTENTION: 这里GT_StdVector与Obvs_StdVector是等价的，宏名不一样而已 
template<int dimension>
void readData(const std::string& file, GT_StdVector(dimension)& datas) {
    datas.clear();

    std::ifstream fin(file);
    std::string line;
    while(getline(fin, line)) {
        std::istringstream sin(line);
        std::string field;
        Eigen::Matrix<double, dimension, 1> vec;
        int k = 0;
        while(getline(sin, field, ' ')) {
            vec(k, 0) = std::stod(field);
            k++;
            if(k == dimension) {
                break;
            }
        }
        datas.push_back(vec);
        // assert(k == dimension); // 保证数据与预想的一致

#ifdef DEBUG
        // for(size_t i = 0; i < state_tmp.size(); i++) {
        //     std::cout << state_tmp[i] << " ";
        // }
        // std::cout << "\n" << std::endl;
        // std::cout << vec.transpose() << std::endl;
#endif // DEBUG
    }
}

void wait_for_key();

int main(int argc, char *argv[]) {

    // 初始化kalman filter
    Eigen::Matrix<double, 2, 2> A, Q;
    Eigen::Matrix<double, 2, 2> H, R;
    A << 1, 1, 0, 1;
    H << 1, 0, 0, 1;
    Q << 0.098, 0.001, 0.001, 0.121;
    R << 1, 0, 0, 1;

    // 创建kalman filter
    KalmanFilter<2, 2> kf(A, H, Q, R);
    // kf.initial(A, H, Q, R);

    // 读取数据
    GT_StdVector(2) gt_datas;
    Obvs_StdVector(2) obvs_datas;
    // readData<2>("/home/lho/MyProgramm/MyCar/develop/kalman_filter/demo_dr_can/test/data/Ground_Truth.txt", gt_datas);
    readTESTData<2, 2>(
        "/home/lho/MyProgramm/MyCar/develop/kalman_filter/demo_dr_can/test/data/Ground_Truth.txt",
        "/home/lho/MyProgramm/MyCar/develop/kalman_filter/demo_dr_can/test/data/Observation.txt",
        gt_datas, 
        obvs_datas
    );

#ifdef DEBUG
    std::cout << gt_datas.size() << std::endl;
    std::cout << obvs_datas.size() << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << gt_datas[i].transpose() << std::endl;
        std::cout << obvs_datas[i].transpose() << std::endl;
    }
#endif // DEBUG

    Eigen::Matrix<double, 2, 1> start_states;
    start_states << 0, 1;
    // Eigen::Matrix<double, 2, 2> states_matrix;
    // states_matrix << 1, 1, 0, 1;
    // Eigen::Matrix<double, 2, 2> obvs_matrix;
    // obvs_matrix << 1, 0, 0, 1;
    // Eigen::Matrix<double, 2, 2> states_cov_matrix;
    // states_cov_matrix << 0.
    // Eigen::Matrix<double, 2, 2> obvs_cov_matrix;
    States_StdVector(2) states;

    kf.estimate_test(obvs_datas, start_states, states);

    // 画图进行对比
    // 1. ground truth
    std::vector<double> position_gt, velocity_gt;
    for(size_t i = 0; i < gt_datas.size(); i++) {
        position_gt.push_back(gt_datas[i](0, 0));
        velocity_gt.push_back(gt_datas[i](1, 0));
    }
    // 2. estimate
    std::vector<double> position_esti, velocity_esti;
    for(size_t i = 0; i < states.size(); i++) {
        position_esti.push_back(states[i](0, 0));
        velocity_esti.push_back(states[i](1, 0));
    }
    // 3. observation
    std::vector<double> position_obvs, velocity_obvs;
    for(size_t i = 0; i < obvs_datas.size(); i++) {
        position_obvs.push_back(obvs_datas[i](0, 0));
        velocity_obvs.push_back(obvs_datas[i](1, 0));
    }

    std::cout << "groud truth size: " << gt_datas.size() << std::endl;
    std::cout << "observation size: " << obvs_datas.size() << std::endl;
    std::cout << "estimate size: " << states.size() << std::endl;

    // 画出veloctiy的gt、观测和估计
    Gnuplot g_velocity("lines");
    g_velocity << "set term qt";
    g_velocity.set_style("lines").plot_x(velocity_gt, "Ground Truth");
    g_velocity.set_style("lines").plot_x(velocity_obvs, "Observation");
    g_velocity.set_style("lines").plot_x(velocity_esti, "Estimation");
    g_velocity.showonscreen();

    // 画出position的gt、观测和估计
    Gnuplot g_position("lines");
    g_position << "set term qt";
    g_position.set_style("lines").plot_x(position_gt, "Ground Truth");
    g_position.set_style("lines").plot_x(position_obvs, "Observation");
    g_position.set_style("lines").plot_x(position_esti, "Estimation");
    g_position.showonscreen();

    wait_for_key();

    return 1;
}

void wait_for_key() {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
    cout << endl << "Press any key to continue..." << endl;

    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
    _getch();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    std::cout << std::endl << "Press ENTER to continue..." << std::endl;

    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
#endif
    return;
}