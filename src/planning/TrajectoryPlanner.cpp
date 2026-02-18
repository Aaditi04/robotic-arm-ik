#include <vector>

std::vector<double> generateTrajectory(double q0, double qf, double T, int steps)
{
    std::vector<double> traj;

    double a0 = q0;
    double a1 = 0;
    double a2 = 3*(qf-q0)/(T*T);
    double a3 = -2*(qf-q0)/(T*T*T);

    for(int i=0;i<=steps;i++)
    {
        double t = (T*i)/steps;
        double q = a0 + a1*t + a2*t*t + a3*t*t*t;
        traj.push_back(q);
    }

    return traj;
}
