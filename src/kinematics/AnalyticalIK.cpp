#include <cmath>
#include <vector>

const double L1 = 0.3;
const double L2 = 0.25;
const double L3 = 0.15;

std::vector<std::vector<double>>
solveIK(double x,double y,double z)
{
    std::vector<std::vector<double>> solutions;

    double t1 = atan2(y,x);

    double r = sqrt(x*x + y*y);
    double zp = z - L1;
    double d = sqrt(r*r + zp*zp);

    if(d > (L2+L3) || d < fabs(L2-L3))
        return solutions;

    double cos_t3 =
        (d*d - L2*L2 - L3*L3) / (2*L2*L3);

    if(fabs(cos_t3) > 1.0)
        return solutions;

    double t3a = acos(cos_t3);      // elbow-down
    double t3b = -acos(cos_t3);     // elbow-up

    for(double t3 : {t3a,t3b})
    {
        double a = atan2(zp,r);
        double b = atan2(L3*sin(t3),
                         L2 + L3*cos(t3));

        double t2 = a - b;

        solutions.push_back({t1,t2,t3});
    }

    return solutions;
}


