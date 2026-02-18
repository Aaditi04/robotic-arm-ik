#include <iostream>
#include <chrono>
#include "../src/kinematics/ML_IK.cpp"

int main()
{
    ML_IK ml("ik_model.onnx");

    // ===== Timing Start =====
    auto start = std::chrono::high_resolution_clock::now();

    auto res = ml.predict(0.3f,0.2f,0.4f);

    auto end = std::chrono::high_resolution_clock::now();
    // ===== Timing End =====

    double time =
    std::chrono::duration<double, std::micro>(end-start).count();

    std::cout<<"ML IK Output:\n";
    std::cout<<res[0]<<" "<<res[1]<<" "<<res[2]<<std::endl;

    std::cout<<"ML IK Time (microseconds): "
             << time << std::endl;

    return 0;
}

