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
    detections.resize(3);
}

cppflow::tensor Detector::Mat2Tensor(const cv::Mat& mat, int& rows, int& cols, int& channels)
{
    img_vector.assign(mat.data,mat.data+mat.total()*channels);
    auto img_tensor = cppflow::tensor(img_vector,{rows,cols,channels});
      
    // img_tensor = cppflow::cast(img_tensor,TF_FLOAT,TF_UINT8);
    img_tensor = cppflow::expand_dims(img_tensor,0);

    
    return img_tensor;
}
void Detector::parse_detections(std::vector<cppflow::tensor>& det)
{
    std::vector<float> boxes_r = det[0].get_data<float>();
    std::vector<float> classes = det[1].get_data<float>();
    std::vector<float> scores = det[2].get_data<float>();


    for(int i=0;i<100;i++)
    {
        if(scores[i]>0.5)
        {
        int idx = std::max(0,(i*4) -1);
        cv::rectangle(cv::InputOutputArray(reduced_size_image),cv::Point(int(boxes_r[idx+1]*640),int(boxes_r[idx]*640)),
                    cv::Point(int(boxes_r[idx+3]*640),int(boxes_r[idx+2]*640)),cv::Scalar(255,0,0),3); 
        }

    }
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

    

    detections[0] = output[1]; // Detection boxes
    detections[1] = output[2]; //Detection classes
    detections[2] = output[4]; // Detection scores

    parse_detections(detections);
    cv_bridge::CvImage cbr;
    cbr.image = reduced_size_image;
    cbr.header = img->header;
    cbr.encoding = sensor_msgs::image_encodings::BGR8;
    img_pub.publish(cbr.toImageMsg());
}