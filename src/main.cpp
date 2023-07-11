#include "mrobot_cliff_detection.h"

CliffDetection::CliffDetection()
{
    ros::NodeHandle nh;
    listener_itof2base_link_ = new tf::TransformListener(nh);
    std::string itof_left_, itof_right_;
    bool itof_topic_left = ros::param::search("/navigation/devices/itof/itof_front_left", itof_left_);
    bool itof_topic_right = ros::param::search("/navigation/devices/itof/itof_front_right", itof_right_);
    if(itof_topic_left && itof_topic_right)
    {
       ros::param::get("/navigation/devices/itof/itof_front_left",itof_left_);
       ros::param::get("/navigation/devices/itof/itof_front_right",itof_right_);
       itof_left_frame_ = itof_left_.substr(itof_left_.find_last_of("/") + 1) + "_frame";
       ROS_INFO("the itof_left_frame_ is %s", itof_left_frame_.c_str());
       itof_right_frame_ = itof_right_.substr(itof_right_.find_last_of("/") + 1) + "_frame";
       ROS_INFO("the itof_right_frame_ is %s", itof_right_frame_.c_str());
    }
    nh.param<double>("cliff_detect_dis_", cliff_detect_dis_, 0.45);
    nh.param<double>("cliff_detect_slope_", cliff_detect_slope_, 0.17);
    nh.param<bool>("judge_cliff_swith2_", judge_cliff_swith2_,false);
    nh.param<int>("regiongrow_min_clustersize_", regiongrow_min_clustersize_,15);
    nh.param<std::string>("itof_left_cloud_", itof_left_cloud_, "/itof/itof_front_left/tof_frame/pointcloud");
    nh.param<std::string>("itof_right_cloud_", itof_right_cloud_, "/itof/itof_front_right/tof_frame/pointcloud");
    nh.param<std::string>("cliff_detect_request_", cliff_detect_request_, "cliff_and_escalator_service");
    itof_left_sub_ =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, itof_left_cloud_, 10);
    itof_right_sub_ =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, itof_right_cloud_, 10);
    itof_sync_ = new message_filters::Synchronizer<ItofCloudPolicy>(ItofCloudPolicy(10), *itof_left_sub_, *itof_right_sub_);
    itof_sync_->registerCallback(boost::bind(&CliffDetection::itofDataCallback, this, _1, _2));
    cliff_detect_client_ = nh.serviceClient<mrobot_srvs::JString>(cliff_detect_request_);
    voxel_filter_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/voxel_filter_cloud_",10);
    roi_area_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/roi_area_cloud", 10);
    project_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/project_cloud", 10);
    pass_filter_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/pass_filter_cloud", 10);
    region_cloud_color_ = nh.advertise<sensor_msgs::PointCloud2>("/region_color",10);
   
}

CliffDetection::~CliffDetection()
{

}

//去除无效点
void CliffDetection::clearNanFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in)
{
    std::vector<int> indices_src; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_in,*cloud_in, indices_src);
    return;
}

//点云下采样
void CliffDetection::dowmsampleFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    if (cloud_in->size() < 5) return;
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud_in);
    voxel.setLeafSize(0.05f,0.05f,0.05f);
    voxel.filter(*cloud_out);
    cloudVisualize(cloud_out, voxel_filter_cloud_);
    return;
}

//提取感兴趣区域
void CliffDetection::extractROIArea(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    //一阶段提取感兴趣区域数目小于10，疑似机器到达悬崖，直接进行二阶段的检测
    if(cloud_in->size() < 10)
    {
        ROS_INFO("the ROIAreaCloud number is %d, smaller thann 10, so start step2", cloud_in->size());
        return;
    }

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto &p:cloud_in->points)
    {
        if (p.z >= -0.1 && p.z <= 0.1 && (p.x * p.x + p.y * p.y <= 0.7 * 0.7))
        {
            cloud_out->push_back(p);
        }
    }
    // ROS_INFO("the size of roi_cloud_out is %d", cloud_out->size());
    cloudVisualize(cloud_out, roi_area_cloud_);
    return;
    // return cloud_out;    
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> CliffDetection::regionGrowing(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in)
{
    int KN_normal = 8; //设置默认输入参数
    double smoothnessThreshold = 30.0, curvatureThreshold = 0.05;
    //法线估计
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>); //创建一个指向kd树搜索对象的共享指针
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator; //创建法线估计对象
    normal_estimator.setSearchMethod(tree); //设置搜索方法
    normal_estimator.setInputCloud(cloud_in); //设置法线估计对象输入点集
    normal_estimator.setKSearch(KN_normal); //设置用于法向量估计的k近邻数目
    normal_estimator.compute(*normals); //计算并输出法向量

    //区域生长算法的5个参数
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg; //创建区域生长分割对象
    reg.setMinClusterSize(regiongrow_min_clustersize_); //设置一个聚类需要的最小点数
    reg.setMaxClusterSize(1000); //设置一个聚类需要的最大点数
    reg.setSearchMethod(tree); //设置搜索方法
    reg.setNumberOfNeighbours(8); //设置搜索的邻近点数目
    reg.setInputCloud(cloud_in); 
    reg.setInputNormals(normals); //设置输入点云的法向量
    reg.setSmoothnessThreshold(smoothnessThreshold / 180.0 * M_PI); //设置平滑阈值,法线差值阈值
    reg.setCurvatureThreshold(curvatureThreshold); //设置曲率阈值
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters); //获取聚类的结果，分割结果保存在点云索引的向量中
    // ROS_INFO("the number of regioncluster is %d", clusters.size());

    // for(int i = 0; i < clusters.size();i++)
    // {
    //     ROS_INFO("i  = %d, the number of cluster is %ld", i, clusters[i].indices.size());
    // }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> RegionGrow; //用于储存区域增长分割后的点云
    for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
            cloud_cluster->points.push_back(cloud_in->points[*pit]);
            cloud_cluster->width = cloud_cluster->points.size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            RegionGrow.push_back(cloud_cluster);
    }

    // ROS_WARN("the number of regionGrow is %d", RegionGrow.size());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr region_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
   // 分割完的平面渲染,并发布
    for (int i = 0; i < RegionGrow.size(); i++)
    {
        for (int j = 0; j < RegionGrow[i]->points.size(); j++)
        {
            pcl::PointXYZRGB p;
            p.x = RegionGrow[i]->points[j].x;
            p.y = RegionGrow[i]->points[j].y;
            p.z = RegionGrow[i]->points[j].z;
            p.r = 0;
            p.g = 255;
            p.b = 0;
            region_cloud_ptr->push_back(p);
        }
    }

    sensor_msgs::PointCloud2 region_cloud_sensor;  //创建新消息发布结果，消息类型为PointCloud2
    pcl::toROSMsg(*region_cloud_ptr, region_cloud_sensor); //此函数将pcl点云转换为ros的点云 
    region_cloud_sensor.header.stamp = ros::Time::now(); // 可选，用于在rviz中查看点云
    region_cloud_sensor.header.frame_id = "/base_link";// 在rviz中查看点云
    region_cloud_color_.publish(region_cloud_sensor); //发布结果 

    return RegionGrow;
}

void CliffDetection::computePlaneThetaAndCenter(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_in, std::vector<std::vector<double> > &vector, std::vector<Eigen::Vector4f> &regions_centroid) //计算区域生长算法分割得到的每个平面的法向量与质心
{
    // ROS_WARN("the number of cloud_in is %d", clouds_in.size());
    if (clouds_in.size() < 1) 
    {
        ROS_INFO("the number of regiongrow plane is %ld", clouds_in.size());
        return;
    }
    vector.clear(); //每个平面初始化
    regions_centroid.clear(); //每个平面质心初始化
    //求每个分割平面的法向量
    for (int i  = 0; i < clouds_in.size(); i++)
    {
        // pcl::PointCloud<pcl::PointXYZ>::Ptr plane_segment(new pcl::PointCloud<pcl::PointXYZ>());
        // 创建一个分割器
        pcl::SACSegmentation <pcl::PointXYZ> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); 
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        seg.setOptimizeCoefficients (true);     // 可选，设置对估计的模型进行优化处理
        seg.setModelType (pcl::SACMODEL_PLANE); //设置分割的模型类型，按照平面进行提取
        seg.setMethodType (pcl::SAC_RANSAC);    //设置所用随机参数估计方法
        seg.setMaxIterations(100);    //最大迭代次数
        seg.setDistanceThreshold(0.02);   //与平面距离小于distanceThreshold的点作为局内点考虑

        //从点云中分割出最大的平面组成部分
        seg.setInputCloud(clouds_in[i]);   //输入点云
        seg.segment(*inliers, *coefficients);   //存储结果到点集合inliers及存储平面模型系数coefficients
        
        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset" << std::endl;
        }

        //输出平面模型的系数n1, n2, n3, n4
        double n1, n2, n3, n4;
        n1 = coefficients->values[0];
        n2 = coefficients->values[1];
        n3 = coefficients->values[2];
        n4 = coefficients->values[3];
    
        Eigen::Vector3d z;
        z[0] = 0;
        z[1] = 0;
        z[2] = 1;

        Eigen::Vector3d v(n1, n2, n3);
        v.normalize();
        double theta = acos(z.dot(v));  //z与v的点乘运算，保证平面法线朝向

        double thetaThreshold = 0.5 * M_PI + 0.1;  //保证ransac方法得到的平面法向量始终朝上

        if (fabs(theta) > thetaThreshold)
        {
            n1 = -n1;
            n2 = -n2;
            n3 = -n3;
            n4 = -n4;
            if(theta>0)
            {
                theta = theta - M_PI ;
            }
            else 
            {
                theta = theta + M_PI ;
            }

        }
        std::vector<double> plane_coefficient(5);
        //存储每个平面的法向量以及与地平面间的夹角theta
        plane_coefficient.push_back(n1);
        plane_coefficient.push_back(n2);
        plane_coefficient.push_back(n3);
        plane_coefficient.push_back(n4);
        plane_coefficient.push_back(theta);
        vector.push_back(plane_coefficient);
    }

    for (int i = 0; i < clouds_in.size(); i++)
    {
        Eigen::Vector4f region_centroid;
        pcl::compute3DCentroid(*clouds_in[i], region_centroid);
        regions_centroid.push_back(region_centroid);
        //ROS_INFO("the z value of regions_centroid is %f", regions_centroid[2]);
    }
 
   return; 
}

//直通滤波
pcl::PointCloud<pcl::PointXYZ>::Ptr CliffDetection::passthroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, 
                                       double x1, double x2, 
                                       double y1, double y2, 
                                       double z1, double z2
                                       )
{
    // ROS_WARN("the size of cloud_in is %ld", cloud_in->size());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pass_filter(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(cloud_in);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(x1, x2);
    pass_x.setFilterLimitsNegative(false);
    pass_x.filter(*pass_filter);
    //ROS_INFO("the size of pass_x is %ld", pass_filter->size());

    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(pass_filter);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(y1, y2);
    pass_y.setFilterLimitsNegative(false);
    pass_y.filter(*pass_filter);
    //ROS_INFO("the size of pass_y is %ld", pass_filter->size());

    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(pass_filter);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(z1, z2);
    pass_z.setFilterLimitsNegative(false);
    pass_z.filter(*pass_filter);
    //ROS_INFO("the size of pass_z is %ld", pass_filter->size());

    cloudVisualize(pass_filter, pass_filter_cloud_);
    return pass_filter;
}

//投影到xoy平面,取其轮廓，并可视化
void CliffDetection::projectToPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    if (cloud_in->size() < 10)
    {
        ROS_INFO("the number of projectToPlane points is %d, smaller than 10", cloud_in->size());
    }

    pcl::PointXYZ project_cloud;
    for (const auto& p: *cloud_in)
    {
        project_cloud.x = p.x;
        project_cloud.y = p.y;
        project_cloud.z = 0.0;
        cloud_out->push_back(project_cloud);
    }
    cloudVisualize(cloud_out, project_cloud_);

    return;
}


void CliffDetection::computeCloudCenter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, Eigen::Vector4f &vector)
{
    if(cloud_in->size() < 10)
    {
        ROS_INFO("the number of computeCloudCenter points is %d, smaller than 10", cloud_in->size());
        return;
    }

    pcl::compute3DCentroid(*cloud_in, vector);
    // ROS_INFO("the centroid of project cloud is %f, %f, %f", vector[0],vector[1], vector[2]);
    return;
}

void CliffDetection::cloudVisualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, ros::Publisher pub)
{
    sensor_msgs::PointCloud2 msgs_cloud;
    pcl::toROSMsg(*cloud_in, msgs_cloud);
    msgs_cloud.header.frame_id = "/base_link";
    msgs_cloud.header.stamp = ros::Time::now();
    pub.publish(msgs_cloud);
    return;
}

//tf变换
void CliffDetection::coordinateTransform(const std::string &target_frame, const std::string &source_frame, const tf::TransformListener *listener, tf::StampedTransform &transform)
{
    try
    {
        listener->waitForTransform(target_frame, source_frame, ros::Time(0), ros::Duration(3.0));
        listener->lookupTransform(target_frame, source_frame, ros::Time(0), transform);
    }
    catch(tf::TransformException &ex)
    {
        ROS_ERROR("%s", ex.what());
        ros::Duration(1.0).sleep();
    }
    return;
}

//TF to Affin3f  从tf变换转换到了变换矩阵
Eigen::Matrix4f CliffDetection::transformTf(const tf::StampedTransform& transform)
{
    //获得平移
    Eigen::Translation3f t(transform.getOrigin().getX(), transform.getOrigin().getY(), transform.getOrigin().getZ());
    //获得旋转
    double roll, pitch, yaw;
    tf::Matrix3x3(transform.getRotation()).getEulerYPR(yaw, pitch, roll);
    Eigen::AngleAxisf rot_x_btol(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rot_y_btol(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rot_z_btol(yaw, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    transform_matrix = (t * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
    return transform_matrix;
}

//计算距离
bool CliffDetection::computeDistance(Eigen::Vector4f &center)
{
    double distance = sqrt((center[0] * center[0]) + (center[1] * center[1]));
   // ROS_INFO("the distance is %f", distance);

    if(distance < cliff_detect_dis_)
    {
        ROS_WARN("robot is approach the cliff area");
        Json::Value value_in;
        value_in["data"]["event"] = "robot_approach_cliff_area";
        value_in["data"]["level"] = 0;
        cliffDetectCall(value_in);
        // robotApproachCliff(value_in);
        return true;
    }

    return false;
}

bool CliffDetection::cliffDetectCall(Json::Value &msgs)
{
    Json::Value res_value;
    Json::Reader reader;
    try
    {
        mrobot_srvs::JString req;
        Json::StyledWriter writer;
        req.request.request = writer.write(msgs);

        cliff_detect_client_.call(req);

        if(req.response.success)
        {
            return true;
        }
        else
        {
            return false;
        }
        
    }
    catch(...)
    {
        ROS_WARN("cliff_detect error!");
    }

    return false;
}

//主要处理函数
void CliffDetection::cliffDetectMainLoop(const std::string &itof_frame_name, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_trans_ptr, double y1, double y2) 
{
try
{
    tf::StampedTransform transform_itof2base_link;
    coordinateTransform("/base_link",itof_frame_name,listener_itof2base_link_, transform_itof2base_link);
    clearNanFilter(cloud_trans_ptr);
    //ROS_WARN("the number of nan_cloud is %d", cloud_trans_ptr->points.size());
    dowmsampleFilter(cloud_trans_ptr,cloud_trans_ptr);
    //ROS_WARN("the number of dowmsampleFilter is %d", cloud_trans_ptr->points.size());
    Eigen::Affine3f itofTobase_link = Eigen::Affine3f::Identity();  //itof->base_link的变换矩阵 
    itofTobase_link = transformTf(transform_itof2base_link);  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud_trans_ptr, *cloud_transformed_ptr, itofTobase_link);      //itof变换到base_link
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in2(new pcl::PointCloud<pcl::PointXYZ>());
   // ROS_INFO("the size of cloud_transformed is %ld", cloud_transformed_ptr->size());

    // 一阶段
    extractROIArea(cloud_transformed_ptr, roi_cloud_ptr); //一阶段提取感兴趣区域,以机器人为中心,裁减 0.7m 内数据
    //ROS_INFO("the number of roi_cloud is %ld", roi_cloud_ptr->size());
    pcl::PointCloud<pcl::PointXYZ>::Ptr project_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    projectToPlane(roi_cloud_ptr,project_cloud_ptr);
    // ROS_INFO("the size of project_cloud_ptr is %ld", project_cloud_ptr->size());
    Eigen::Vector4f center = Eigen::Vector4f::Identity();
    computeCloudCenter(project_cloud_ptr,center);
    if((center[0] * center[0] + center[1] * center[1]) == 1.0) //机器被人抬走直接放在悬崖区域,直接进行二阶段检测
    {
        judge_cliff_swith2_ = true;
    }
    else
    {
        judge_cliff_swith2_ = computeDistance(center); //计算质心与机器人之间的距离,满足设置条件作为疑似悬崖区域,并且开启第二阶段检测
    }
   
    //二阶段: 第二次提取感兴趣区域，提取机器人1m*1m*1m内区域内点云数据
    if(judge_cliff_swith2_)
    {
        pass_filter_y1_ = y1;
        pass_filter_y2_ = y2;
        pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud_ptr2(new pcl::PointCloud<pcl::PointXYZ>()); 
        // ROS_WARN("the size of cloud_transformed_ptr is %ld", cloud_transformed_ptr->size());
        roi_cloud_ptr2 = passthroughFilter(cloud_transformed_ptr,0.0, 1.0, pass_filter_y1_, pass_filter_y2_, -0.5, 0.5);
        //ROS_WARN("the size of roi_cloud_pr2 is %ld", roi_cloud_ptr2->size());
        std::vector<Eigen::Vector4f> vector_theta;
        std::vector<std::vector<double> > plane_theta;
        if(roi_cloud_ptr2->size() < 5)
        {
            ROS_INFO("the ROI_Cloud points is %ld, so stop region_grow_function", roi_cloud_ptr2->size());
            return;
         }
        region_grow_planes_ = regionGrowing(roi_cloud_ptr2); //区域增长分割算法分割平面

        // TODO: ROS_INFO("the number of region_grow_planes is %d", region_grow_planes_.size());

        computePlaneThetaAndCenter(region_grow_planes_, plane_theta, vector_theta); //计算每个平面法向量与(0,0,1)间的夹角以及每个平面的质心
        // ROS_WARN("the size of region_grow_planes_ is %d, plane_theta is %d, vector_theta is %d", region_grow_planes_.size(), plane_theta.size(), vector_theta.size());
        //遍历每个平面的夹角以及质心
        for(int i = 0; i <= region_grow_planes_.size(); i++)
        {
            for(int j = 0; j < plane_theta.size(); j++)
           {
                if(plane_theta[j][4] <= cliff_detect_slope_)
                {
                    if(vector_theta[j][2] <= 0.0)   //质心坐标的z值小于0       
                    {
                        ROS_WARN("robot is get the cliff area");
                        Json::Value value_in;
                        value_in["data"]["event"] = "robot_get_cliff_area";
                        value_in["data"]["level"] = 1;
                        cliffDetectCall(value_in);
                        // robotReachCliff(value_in);
                    }
                }
            }
        }
        
    }
    
    return;
}
catch(pcl::PCLException &e)
{
    ROS_INFO_STREAM("abnormal process1...");
}
catch(std::exception &e)
{
    ROS_INFO_STREAM("abnormal process2....");
}  
 
}

void CliffDetection::itofDataCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud1_in, const sensor_msgs::PointCloud2::ConstPtr &cloud2_in)
{   
    //左侧itof处理
    pcl::PointCloud<pcl::PointXYZ>::Ptr leftcloud_trans_ptr(new pcl::PointCloud<pcl::PointXYZ>()); //左侧itof
    pcl::fromROSMsg(*cloud1_in, *leftcloud_trans_ptr);
    cliffDetectMainLoop(itof_left_frame_, leftcloud_trans_ptr,0.0, 1.0);

    //右侧itof处理
    pcl::PointCloud<pcl::PointXYZ>::Ptr rightcloud_trans_ptr(new pcl::PointCloud<pcl::PointXYZ>()); //右侧itof
    pcl::fromROSMsg(*cloud2_in, *rightcloud_trans_ptr);
    cliffDetectMainLoop(itof_right_frame_, rightcloud_trans_ptr, -1.0, 0.0);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cliff_detect");
    CliffDetection itof_cliff_detect;
    ros::spin();
    return 0;
}
