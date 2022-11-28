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
      
    // img_tensor = cppflow::cast(img_tensor,TF_FLOAT,TF_UINT8);
    img_tensor = cppflow::expand_dims(img_tensor,0);

    
    return img_tensor;
}

void Detector::callback(const sensor_msgs::ImageConstPtr& img)
{
    // Convert Image msg to CV   
   cvImgptr = cv_bridge::toCvCopy(img,"bgr8");
  
    //Resize to 640x640
    cv::resize(cvImgptr->image,reduced_size_image,cv::Size(img_rows,img_cols),cv::INTER_LINEAR);

    cppflow::tensor img_tensor = Mat2Tensor(reduced_size_image,img_rows,img_cols,img_channels);

    // Inference
    auto output = (*detector)({{"serving_default_input_tensor:0", img_tensor}},{"StatefulPartitionedCall:0",
                                                                                "StatefulPartitionedCall:1",
                                                                                "StatefulPartitionedCall:2",
                                                                                "StatefulPartitionedCall:3",
                                                                                "StatefulPartitionedCall:4",
                                                                                "StatefulPartitionedCall:5",
                                                                                "StatefulPartitionedCall:6",
                                                                                "StatefulPartitionedCall:7"});

    std::cout<<"Output:\n";
    std::cout<<output[4];

}