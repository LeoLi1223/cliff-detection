#ifndef MROBOT_CLIFF_DETECTION_
#define MROBOT_CLIFF_DETECTION_

//ros lib
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//pcl lib
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/region_growing.h>

//Eigen lib
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

//c++ lib
#include <boost/thread/thread.hpp>
#include <iostream>

//json lib
#include <jsoncpp/json/json.h>

//other lib
#include <mrobot_srvs/JString.h>
#include <mrobot_srvs/JStringRequest.h>
#include <mrobot_srvs/JStringResponse.h>

using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,sensor_msgs::PointCloud2> ItofCloudPolicy;

class CliffDetection
{
public:
    CliffDetection();
    ~CliffDetection();
    void itofDataCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud1_in, const sensor_msgs::PointCloud2::ConstPtr &cloud2_in);
    void clearNanFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in);
    void dowmsampleFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out);
    void projectToPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out); //投影到xoy平面
    void extractROIArea(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out); //截取以base_link为中心，0.6米内的数据
    void computeCloudCenter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, Eigen::Vector4f &vector); //求点云质心
    bool computeDistance(Eigen::Vector4f &center); //计算平面质心与机器人之间的距离，一阶段检测上报事件, 机器减速运动
    pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, double x1, double x2, double y1, double y2, double z1, double z2); //直通滤波
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> regionGrowing(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in); //区域增长算法，判断机器人1m内平面是否存在地面以下数据
    void computePlaneThetaAndCenter(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_in, std::vector<std::vector<double> > &vector, std::vector<Eigen::Vector4f> &regions_centroid); //计算区域生长算法分割得到的每个平面的法向量与质心，二阶段检测上报事件,机器人停止运动
    void cloudVisualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, ros::Publisher pub);
    void coordinateTransform(const std::string &target_frame, const std::string &source_frame, const tf::TransformListener *listener, tf::StampedTransform &transform);
    Eigen::Matrix4f transformTf(const tf::StampedTransform& transform);
    void cliffDetectMainLoop(const std::string &itof_frame_name, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_trans_ptr, double y1, double y2); //处理主要功能函数
    bool cliffDetectCall(Json::Value &msgs); //响应函数
    
private:
    ros::Publisher voxel_filter_cloud_; //点云下采样后的点云数据
    ros::Publisher roi_area_cloud_; //一阶段感兴趣区域
    ros::Publisher project_cloud_; //投影后点云数据
    ros::Publisher pass_filter_cloud_;  //二阶段直通滤波后的点云数据
    ros::Publisher region_cloud_color_; //分割完带有颜色信息的平面 
    ros::ServiceClient cliff_detect_client_; //服务客户端
    std::string itof_left_frame_; //左侧itof的frame名
    std::string itof_right_frame_;
    std::string itof_left_cloud_; //左侧itof点云名
    std::string itof_right_cloud_; 
    std::string cliff_detect_request_;
    double cliff_detect_dis_; //距离
    double cliff_detect_slope_; //角度
    double pass_filter_y1_, pass_filter_y2_;
    int regiongrow_min_clustersize_;  //区域增长聚类点云需要满足的最小数目
    bool judge_cliff_swith2_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> region_grow_planes_;
    tf::TransformListener *listener_itof2base_link_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *itof_left_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *itof_right_sub_;
    message_filters::Synchronizer<ItofCloudPolicy>*  itof_sync_;
};


#endif