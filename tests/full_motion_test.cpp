#include <iostream>
#include <vector>

bool solveIK(double x, double y, double z,
             double &t1, double &t2, double &t3);

std::vector<double> generateTrajectory(double q0, double qf,
                                       double T, int steps);

int main()
{
    double targetX = 0.3;
    double targetY = 0.2;
    double targetZ = 0.4;

    double t1, t2, t3;

    if(!solveIK(targetX, targetY, targetZ, t1, t2, t3))
    {
        std::cout<<"Target unreachable\n";
        return 0;
    }

    // Assume starting from zero position
    double c1 = 0, c2 = 0, c3 = 0;

    auto traj1 = generateTrajectory(c1, t1, 2.0, 20);
    auto traj2 = generateTrajectory(c2, t2, 2.0, 20);
    auto traj3 = generateTrajectory(c3, t3, 2.0, 20);

    for(int i=0;i<traj1.size();i++)
    {
        std::cout<<"Step "<<i<<" -> "
                 <<"t1: "<<traj1[i]<<" "
                 <<"t2: "<<traj2[i]<<" "
                 <<"t3: "<<traj3[i]<<"\n";
    }

    return 0;
}
