#include <iostream>
#include <vector>

std::vector<std::vector<double>>
solveIK(double x,double y,double z);

std::vector<double> generateTrajectory(double q0, double qf,
                                       double T, int steps);

int main()
{
    // Target position
    double x = 0.3;
    double y = 0.2;
    double z = 0.4;

    auto sols = solveIK(x,y,z);

    if(sols.empty())
    {
        std::cout<<"IK failed\n";
        return 0;
    }

    double t1 = sols[0][0];
    double t2 = sols[0][1];
    double t3 = sols[0][2];

    auto traj1 = generateTrajectory(0,t1,2.0,20);
    auto traj2 = generateTrajectory(0,t2,2.0,20);
    auto traj3 = generateTrajectory(0,t3,2.0,20);

    for(int i=0;i<traj1.size();i++)
    {
        std::cout<<"Step "<<i<<" -> "
                 <<traj1[i]<<" "
                 <<traj2[i]<<" "
                 <<traj3[i]<<"\n";
    }

    return 0;
}

