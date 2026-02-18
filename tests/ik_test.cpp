#include <iostream>
#include <chrono>
#include <vector>

// Updated declaration
std::vector<std::vector<double>>
solveIK(double x,double y,double z);

void forwardKinematics(double t1, double t2, double t3,
                       double &x, double &y, double &z);

int main()
{
    double x=0.3, y=0.2, z=0.4;

    auto start = std::chrono::high_resolution_clock::now();

    auto sols = solveIK(x,y,z);

    auto end = std::chrono::high_resolution_clock::now();

    if(sols.empty())
    {
        std::cout<<"IK failed\n";
        return 0;
    }

    double time =
    std::chrono::duration<double, std::micro>(end-start).count();

    double t1 = sols[0][0];
    double t2 = sols[0][1];
    double t3 = sols[0][2];

    double fx,fy,fz;
    forwardKinematics(t1,t2,t3,fx,fy,fz);

    std::cout<<"Target: "<<x<<" "<<y<<" "<<z<<"\n";
    std::cout<<"FK: "<<fx<<" "<<fy<<" "<<fz<<"\n";

    std::cout<<"Analytical IK Time (microseconds): "
             << time << std::endl;

    return 0;
}



