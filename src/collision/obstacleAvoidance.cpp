#include <cmath>

bool checkCollision(double x, double y, double z,
                    double ox, double oy, double oz,
                    double radius)
{
    double dx = x - ox;
    double dy = y - oy;
    double dz = z - oz;

    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    return dist < radius;
}
