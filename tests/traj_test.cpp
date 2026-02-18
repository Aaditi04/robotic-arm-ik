#include <iostream>
#include <vector>

std::vector<double> generateTrajectory(double q0, double qf,
                                       double T, int steps);

int main()
{
    auto traj = generateTrajectory(0.0, 1.57, 2.0, 10);

    for(double q : traj)
        std::cout << q << std::endl;

    return 0;
}
