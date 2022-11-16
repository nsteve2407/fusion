#include<detector.h>

Detector::Detector(std::string model_path,std::string topic,ros::NodeHandle& n)
{
    detector.reset(new cppflow::model(model_path));
    ROS_INFO("Detector model successfully loaded");
    nh = n;
    nh.getParam("/ssd_detector/rows",img_rows);
    nh.getParam("/ssd_detector/columns",img_cols);
    nh.getParam("/ssd_detector/channels",img_channels);
    img_pub = nh.advertise<sensor_msgs::Image>("/img_with_bbox",100);
    img_data_pub = nh.advertise<vision_msgs::Detection2DArray>("/img_detections",100);
    img_sub = nh.subscribe<sensor_msgs::Image>(topic,100,&Detector::callback,this);
    ROS_INFO("Subscribers and Publishers successfully loaded");
}

cppflow::tensor Detector::Mat2Tensor(const cv::Mat& mat, int& rows, int& cols, int& channels)
{
    img_vector.assign(mat.data,mat.data+mat.total()*channels);
    auto img_tensor = cppflow::tensor(img_vector,{rows,cols,channels});
    img_tensor = cppflow::cast(img_tensor,TF_UINT8,TF_FLOAT);
    img_tensor = cppflow::expand_dims(img_tensor,0);
    
    return img_tensor;
}

void Detector::callback(const sensor_msgs::ImageConstPtr& img)
{
    // Convert Image msg to CV
    cvImgptr = cv_bridge::toCvCopy(img,sensor_msgs::image_encodings::BGR8);
    
    //Resize to 640x640
    cv::resize(cvImgptr->image,reduced_size_image,cv::Size(img_rows,img_cols),cv::INTER_LINEAR);

    cppflow::tensor img_tensor = Mat2Tensor(reduced_size_image,img_rows,img_cols,img_channels);

}