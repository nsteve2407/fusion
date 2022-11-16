#include<detector.h>

Detector::Detector(std::string model_path,std::string topic,ros::NodeHandle& n)
{
    detector.reset(new cppflow::model(model_path));
    ROS_INFO("Detector model successfully loaded");
    nh = n;
    img_pub = nh.advertise<sensor_msgs::Image>("/img_with_bbox",100);
    img_data_pub = nh.advertise<vision_msgs::Detection2DArray>("/img_detections",100);
    img_sub = nh.subscribe<sensor_msgs::Image>(topic,100,&Detector::callback,this);
    ROS_INFO("Subscribers and Publishers successfully loaded");
}

void Detector::callback(const sensor_msgs::ImageConstPtr& img)
{

}