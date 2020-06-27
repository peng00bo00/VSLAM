#include <iostream>

using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "./poses.txt";
string points_file = "./points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {return false;}

    bool write(std::ostream &os) const {return false;}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<float,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        auto v0 = (g2o::VertexSBAPointXYZ *) _vertices[0];
        auto v1 = (VertexSophus *) _vertices[1];

        Eigen::Vector3d P = (v1->estimate()) * (v0->estimate());

        float x = fx*P[0]/P[2] + cx;
        float y = fy*P[1]/P[2] + cy;

        for (int i = -2; i < 2; i++)
        {
            for (int j = -2; j < 2; j++)
            {
                int idx = 4 * (i+2) + (j+2);
                _error[idx] = origColor[idx] - GetPixelValue(targetImg, x+i, y+j);
            }
            
        }

        // END YOUR CODE HERE
    }

    // Let g2o compute jacobian for you

    virtual bool read(istream &in) override {return false;}

    virtual bool write(ostream &out) const override {return false;}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();


    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    // typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的
    // DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    // DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( std::unique_ptr<Block::LinearSolverType>(linearSolver) );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr) );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE

    /// vertices
    vector<g2o::VertexSBAPointXYZ *> vertex_points;
    vector<VertexSophus *> vertex_poses;

    for (size_t i = 0; i < poses.size(); i++)
    {
        VertexSophus* vpose = new VertexSophus();

        vpose -> setId(i);
        vpose -> setEstimate(poses[i]);

        optimizer.addVertex(vpose);

        vertex_poses.push_back(vpose);
    }

    for (size_t i = 0; i < points.size(); i++)
    {
        g2o::VertexSBAPointXYZ* vpoint = new g2o::VertexSBAPointXYZ();

        vpoint -> setId(poses.size()+i);
        vpoint -> setMarginalized(true);
        vpoint -> setEstimate(points[i]);

        optimizer.addVertex(vpoint);

        vertex_points.push_back(vpoint);
    }

    /// edges
    for (size_t i = 0; i < vertex_poses.size(); i++)
    {
        cv::Mat image = images[i];
        for (size_t j = 0; j < vertex_points.size(); j++)
        {
            Eigen::Vector3d P = (vertex_poses[i] -> estimate()) * (vertex_points[j] -> estimate());
            float x = fx*P[0]/P[2] + cx;
            float y = fy*P[1]/P[2] + cy;

            if (x-2 > 0 && y-2 > 0 && x+2 < image.cols && y+2 < image.rows)
            {
                EdgeDirectProjection* edge = new EdgeDirectProjection(color[i], image);

                edge->setVertex(1, vertex_poses[i]);
                edge->setVertex(0, vertex_points[j]);
                edge->setInformation(Eigen::Matrix<double,16,16>::Identity());

                g2o::RobustKernelHuber* huber = new g2o::RobustKernelHuber;
                huber -> setDelta(0.5);
                edge  -> setRobustKernel(huber);

                optimizer.addEdge(edge);
            }
            
        }
        
    }
    
    

    // // add pose vertex
    // for(int i = 0; i < poses.size(); ++i) {
    //     VertexSophus* v = new VertexSophus();
    //     v->setId(i);
    //     v->setEstimate(poses[i]);
    //     optimizer.addVertex(v);
    // }
    // // add point Vertex
    // for(int i = 0; i < points.size(); ++i) {
    //     double x = points[i][0];
    //     double y = points[i][1];
    //     double z = points[i][2];

    //     g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
    //     v->setId(poses.size() + i);
    //     v->setMarginalized(true);
    //     v->setEstimate(Eigen::Vector3d(x,y,z));
    //     optimizer.addVertex(v);
    // }
    // // add edges
    // vector<EdgeDirectProjection*>edges;
    // for(int i = 0; i < poses.size(); ++i) {
    //     for(int j = 0; j < points.size(); ++j) {

    //         g2o::VertexSBAPointXYZ* vp = static_cast< g2o::VertexSBAPointXYZ*>(optimizer.vertex(poses.size() + j));
    //         VertexSophus* vse = static_cast<VertexSophus*>(optimizer.vertex(i));

    //         Sophus::SE3d se3 = vse->estimate();
    //         Eigen::Vector3d p = vp->estimate();
    //         Eigen::Vector3d pt = se3 * p;
    //         float u = pt(0) / pt(2) * fx + cx;
    //         float v = pt(1) / pt(2) * fy + cy;

    //         if(u-2 < 0 || u+1 >= 640 || v-2 < 0 || v+1 >= 480) {
    //             continue;
    //         }

    //         EdgeDirectProjection* edge = new EdgeDirectProjection(color[i], images[i]);
    //         edge->setVertex(0, vp);
    //         edge->setVertex(1, vse);
    //         edge->setInformation(Eigen::Matrix<double,16,16>::Identity());
    //         g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //         rk->setDelta(0.5);
    //         edge->setRobustKernel(rk);
    //         optimizer.addEdge(edge);
    //         edges.push_back(edge);
    //     }
    // }
    // cout << "num of edges: " << edges.size() << endl;

    // END YOUR CODE HERE

    // perform optimization
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(200);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE

    for(int i = 0; i < poses.size(); ++i) {
        VertexSophus* vpose = static_cast<VertexSophus*>(optimizer.vertex(i));
        poses[i] = vpose->estimate();
    }
    for(int i = 0; i < points.size(); ++i) {
        g2o::VertexSBAPointXYZ* vpoint = static_cast< g2o::VertexSBAPointXYZ*>(optimizer.vertex(poses.size() + i));
        points[i] = vpoint->estimate();
    }
    // END YOUR CODE HERE

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
