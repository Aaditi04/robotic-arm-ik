#include <iostream>
#include <vector>

// Updated IK declaration
std::vector<std::vector<double>>
solveIK(double x,double y,double z);

std::vector<double> generateTrajectory(double q0, double qf,
                                       double T, int steps);

bool checkCollision(double x, double y, double z,
                    double ox, double oy, double oz,
                    double radius);

void forwardKinematics(double t1, double t2, double t3,
                       double &x, double &y, double &z);

class Controller
{
public:
    void moveTo(double x, double y, double z)
    {
        auto sols = solveIK(x,y,z);

        if(sols.empty())
        {
            std::cout<<"Target unreachable\n";
            return;
        }

        // Choose first IK solution
        double t1 = sols[0][0];
        double t2 = sols[0][1];
        double t3 = sols[0][2];

        auto traj1 = generateTrajectory(0,t1,2.0,20);
        auto traj2 = generateTrajectory(0,t2,2.0,20);
        auto traj3 = generateTrajectory(0,t3,2.0,20);

        double obsX=0.25, obsY=0.15, obsZ=0.35;
        double radius=0.05;

        for(int i=0;i<traj1.size();i++)
        {
            double fx,fy,fz;

            forwardKinematics(traj1[i],traj2[i],traj3[i],
                               fx,fy,fz);

            if(checkCollision(fx,fy,fz,
                              obsX,obsY,obsZ,radius))
            {
                std::cout<<"Collision at step "<<i<<"\n";
                return;
            }

            std::cout<<"Step "<<i<<" -> "
                     <<traj1[i]<<" "
                     <<traj2[i]<<" "
                     <<traj3[i]<<"\n";
        }

        std::cout<<"Motion completed safely\n";
    }
};

