#include <cmath>

void forwardKinematics(double t1, double t2, double t3,
                        double &x, double &y, double &z)
{
    const double L1 = 0.3;
    const double L2 = 0.25;
    const double L3 = 0.15;

    double r = L2*cos(t2) + L3*cos(t2 + t3);

    x = r * cos(t1);
    y = r * sin(t1);
    z = L1 + L2*sin(t2) + L3*sin(t2 + t3);
}
