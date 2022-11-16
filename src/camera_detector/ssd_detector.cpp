#include <iostream>
#include<detector.h>

int main(int argc,char** argv) 
{
    ros::init(argc,argv,"Detector");
    ros::NodeHandle nh;

    std::string model_full_path = "";
    std::string img_topic = "";

    std::shared_ptr<Detector> ssd(new Detector(model_full_path,img_topic,nh));

    return 0;
}