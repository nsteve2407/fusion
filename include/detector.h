// ROS dependencies
#include<ros/ros.h>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<vision_msgs/Detection2DArray.h>

// Tensprflow
#include<cppflow/cppflow.h>

// Other
#include<string>
#include<opencv2/core.hpp>
#include "opencv2/core/types_c.h"
#include<vector>
#include<eigen3/Eigen/Dense>

class Detector
{
    private:
    ros::NodeHandle nh;
    ros::Subscriber img_sub;
    ros::Publisher img_pub;
    ros::Publisher img_data_pub;
    std::shared_ptr<cppflow::model> detector;
    std::vector<uint8_t> img_vector;
    cv_bridge::CvImagePtr cvImgptr;
    cv::Mat reduced_size_image;
    int img_rows, img_cols, img_channels;
    std::vector<cppflow::tensor> detections;

    public:
    Detector(std::string model_path,std::string topic,ros::NodeHandle& n);
    void callback(const sensor_msgs::ImageConstPtr& img);
    cppflow::tensor Mat2Tensor(const cv::Mat& mat, int& rows, int& cols, int& channels);
    void parse_detections(std::vector<cppflow::tensor>& det);

};

