#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include <unistd.h>

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);
void ReadTrajectory(const string path, TrajectoryType &traj_e, TrajectoryType &traj_g);

string pc_file = "./compare.txt";

int main() {
    TrajectoryType traj_e, traj_g;
    // load the poses
    ReadTrajectory(pc_file, traj_e, traj_g);

    // show the point cloud
    // DrawTrajectory(traj_g, traj_e);

    // get centroids
    Vector3d p(0, 0, 0), pp(0, 0, 0);
    for (size_t i = 0; i < traj_e.size(); i++)
    {
        p += traj_g[i].translation();
        pp+= traj_e[i].translation();
    }

    p = p / traj_g.size();
    pp= pp/ traj_e.size();

    // move to centroids
    vector<Vector3d, Eigen::aligned_allocator<Vector3d>> q, qq;
    for (size_t i = 0; i < traj_e.size(); i++)
    {
        // Vector3d qi = traj_g[i].translation() - p;
        // Vector3d qqi= traj_e[i].translation() - pp;

        q.push_back(traj_g[i].translation() - p);
        qq.push_back(traj_e[i].translation() - pp);
    }

    // get W matrix
    Matrix3d W = Matrix3d::Zero();
    for (size_t i = 0; i < traj_e.size(); i++)
    {
        W += q[i] * qq[i].transpose();
    }
    
    // SVD
    JacobiSVD<Eigen::MatrixXd> svd(W, ComputeThinU | ComputeThinV );
    Matrix3d V = svd.matrixV(), U = svd.matrixU();
    
    // get rotation R
    Matrix3d R = U * V.transpose();

    // get translation t
    Vector3d t = p - R * pp;

    Sophus::SE3d T(R, t);

    // modify the estimation
    TrajectoryType traj;
    for (size_t i = 0; i < traj_e.size(); i++)
    {
        Sophus::SE3d p = T * traj_e[i];
        traj.push_back(p);
    }

    // show the point cloud
    DrawTrajectory(traj_g, traj);
    

    return 0;
}


// get this from previous homework
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}

// modify this from previous homework
void ReadTrajectory(const string path, TrajectoryType &traj_e, TrajectoryType &traj_g) {
    ifstream fin(path);

    while (!fin.eof()) {
        double te, te_x, te_y, te_z, qe_x, qe_y, qe_z, qe_w;
        double tg, tg_x, tg_y, tg_z, qg_x, qg_y, qg_z, qg_w;

        fin >> te >> te_x >> te_y >> te_z >> qe_x >> qe_y >> qe_z >> qe_w 
            >> tg >> tg_x >> tg_y >> tg_z >> qg_x >> qg_y >> qg_z >> qg_w;

        Sophus::SE3d pe(Eigen::Quaterniond(qe_w, qe_x, qe_y, qe_z), Eigen::Vector3d(te_x, te_y, te_z));
        Sophus::SE3d pg(Eigen::Quaterniond(qg_w, qg_x, qg_y, qg_z), Eigen::Vector3d(tg_x, tg_y, tg_z));
        
        traj_e.push_back(pe);
        traj_g.push_back(pg);
    }
}