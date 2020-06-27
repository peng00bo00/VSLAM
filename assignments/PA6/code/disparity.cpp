#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string left_img  = "./left.png";    // left image
string right_img = "./right.png";   // right image
string disparity_img = "./disparity.png";    // disparity image

// get this from optical_flow.cpp
float GetPixelValue(const cv::Mat &img, float x, float y) {
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

int main(int argc, char **argv) {

    // load images
    Mat left      = imread(left_img, 0);
    Mat right     = imread(right_img, 0);
    Mat disparity = imread(disparity_img, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kps;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(left, kps);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kps) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(left, right, pt1, pt2, status, error, cv::Size(16, 16));

    Mat optical_flows;
    cv::cvtColor(left, optical_flows, cv::COLOR_GRAY2BGR);
    int num = 0;
    int outliers = 0;
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            num++;
            cv::circle(optical_flows, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(optical_flows, pt1[i], pt2[i], cv::Scalar(0, 250, 0));

            double dx = pt1[i].x - pt2[i].x;
            double gt = GetPixelValue(disparity, pt1[i].x, pt1[i].y);
            double re = (dx-gt)/gt * 100;

            cout << "Disparity from Optical Flow: " << dx << ", GT: " << gt << ", Relative Error:" << re << "%" << endl;
            if (re > 10 or re < -10) outliers++;
        }
    }

    cout << "Outliers: " << outliers << ", Percent: " << (double) outliers / num * 100 << endl;

    Mat leftkps;
    cv::drawKeypoints(left, kps, leftkps, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("left image", left);
    cv::imshow("right image", right);
    cv::imshow("disparity image", disparity);
    cv::imshow("keypoints", leftkps);
    cv::imshow("optical_flows", optical_flows);
    cv::waitKey(0);

    return 0;
}