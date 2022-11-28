#include <iostream>
#include<detector.h>

int main(int argc,char** argv) 
{
    ros::init(argc,argv,"Detector");
    ros::NodeHandle nh;

    std::string model_full_path , img_topic;
    nh.getParam("/ssd_detector/img_topic",img_topic);
    nh.getParam("ssd_detector/ssd_model_path",model_full_path);

    std::shared_ptr<Detector> ssd(new Detector(model_full_path,img_topic,nh));

    while(ros::ok())
    {
        ros::spinOnce();
    }

    return 0;
}