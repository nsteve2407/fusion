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

class Detector
{
    private:
    ros::Subscriber img_sub;
    ros::Publisher img_pub;
    ros::Publisher img_data_pub;
    cppflow::model detector;
    public:
    Detector(std::string model_path,std::string topic);
    void callback(const sensor_msgs::ImageConstPtr& img);

};

