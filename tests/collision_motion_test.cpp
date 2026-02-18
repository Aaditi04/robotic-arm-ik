#include <iostream>
#include <vector>

bool solveIK(double x, double y, double z,
             double &t1, double &t2, double &t3);

void forwardKinematics(double t1, double t2, double t3,
                        double &x, double &y, double &z);

std::vector<double> generateTrajectory(double q0, double qf,
                                       double T, int steps);

bool checkCollision(double x, double y, double z,
                    double ox, double oy, double oz,
                    double radius);

int main()
{
    // Target
    double targetX = 0.3;
    double targetY = 0.2;
    double targetZ = 0.4;

    // Obstacle (place it somewhere in workspace)
    double obsX = 0.25;
    double obsY = 0.15;
    double obsZ = 0.35;
    double radius = 0.05;

    double t1, t2, t3;

    if(!solveIK(targetX, targetY, targetZ, t1, t2, t3))
    {
        std::cout<<"Target unreachable\n";
        return 0;
    }

    double c1=0, c2=0, c3=0;

    auto traj1 = generateTrajectory(c1, t1, 2.0, 20);
    auto traj2 = generateTrajectory(c2, t2, 2.0, 20);
    auto traj3 = generateTrajectory(c3, t3, 2.0, 20);

    for(int i=0;i<traj1.size();i++)
    {
        double fx, fy, fz;

        forwardKinematics(traj1[i], traj2[i], traj3[i],
                          fx, fy, fz);

        if(checkCollision(fx, fy, fz,
                          obsX, obsY, obsZ, radius))
        {
            std::cout<<"Collision detected at step "<<i<<"\n";
            return 0;
        }

        std::cout<<"Step "<<i<<" OK\n";
    }

    std::cout<<"Motion completed safely\n";

    return 0;
}
