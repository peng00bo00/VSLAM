#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include <unistd.h>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory files
string estimated = "../estimated.txt";
string ground_truth = "../groundtruth.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);
TrajectoryType ReadTrajectory(const string path);

int main(int argc, char **argv) {

    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> estimated_traj, ground_truth_traj;

    estimated_traj = ReadTrajectory(estimated);
    ground_truth_traj = ReadTrajectory(ground_truth);

    double rmse = 0;

    for (size_t i = 0; i < estimated_traj.size(); i++)
    {
        Sophus::SE3d p1 = estimated_traj[i];
        Sophus::SE3d p2 = ground_truth_traj[i];

        double e = (p2.inverse() * p1).log().norm();
        rmse += e * e;
    }

    rmse = rmse / estimated_traj.size();
    rmse = sqrt(rmse);

    cout << "RMSE: " << rmse << endl;

    // draw trajectory in pangolin
    DrawTrajectory(ground_truth_traj, estimated_traj);
    return 0;
}

/*******************************************************************************************/
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

TrajectoryType ReadTrajectory(const string path) {
    ifstream fin(path);
    TrajectoryType traj;

    if (!fin) {
        cerr << "Trajectory " << path << " not found." << endl;
        return traj;
    }

    while (!fin.eof()) {
        double t, tx, ty, tz, qx, qy, qz, qw;
        fin >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        Sophus::SE3d pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        
        traj.push_back(pose);
    }

    return traj;
}