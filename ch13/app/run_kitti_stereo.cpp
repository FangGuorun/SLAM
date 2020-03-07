//
// Created by gaoxiang on 19-5-4.
//
using namespace std;
#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"
#include <iostream>     // std::cout, std::fixed
#include <iomanip>		// std::setprecision

DEFINE_string(config_file, "/home/guorunfang/slam_course/hw8/hw8/config/default.yaml", "config file path");
//保存轨迹
void SaveTrajectoryKITTI(const string &filename,const myslam::Map::FramesType frames);


int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();
    cout << endl << "Saving camera trajectory to " << "/home/guorunfang/slam_course/hw8/CameraTrajectory_Tcw.txt" << " ..." << endl;
    //要得到所有未丢失帧的位姿，就先找输入或输出是每一帧的函数 ，找不到再找输入或输出能组成所有帧的函数
    //找一个容器存储每一个位姿，然后到这里提取出来？
    SaveTrajectoryKITTI("/home/guorunfang/slam_course/hw8/CameraTrajectory_Twc_23_30.txt",vo->map_->frames_);
    //SaveTrajectoryKITTI("/home/guorunfang/slam_course/hw8/CameraTrajectory_twc2_raw.txt",vo->map_->frames_raw);

    return 0;
}


void SaveTrajectoryKITTI(const string &filename,const myslam::Map::FramesType frames){

    ///keyframes_.matrix的格式与kitti相符

    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    int number = 0;
    cout<<"frames.size() = " << frames.size() << endl;
    for(size_t i=0; i<frames.size(); ++i){
        number++;
        auto Tcw = frames.at(i)->pose_;
        //auto Rcw = Tcw.rotationMatrix();
        //auto tcw = Tcw.translation();

        auto Twc = Tcw.inverse().matrix();
        //auto Rwc = Twc.rotationMatrix();
        //auto twc = Twc.translation();
        f << setprecision(9) << Twc(0,0) << " " << Twc(0,1)  << " " << Twc(0,2) << " "  << Twc(0,3) << " " <<
          Twc(1,0) << " " << Twc(1,1)  << " " << Twc(1,2) << " "  << Twc(1,3)/Twc(2,3) << " " <<
          Twc(2,0) << " " << Twc(2,1)  << " " << Twc(2,2) << " "  << Twc(2,3) << endl;
        //f << setprecision(9) << Rcw(0,0) << " " << Rcw(0,1)  << " " << Rcw(0,2) << " "  << tcw(0) << " " <<
        //  Rcw(1,0) << " " << Rcw(1,1)  << " " << Rcw(1,2) << " "  << tcw(1) << " " <<
        //  Rcw(2,0) << " " << Rcw(2,1)  << " " << Rcw(2,2) << " "  << tcw(2) << endl;
        //cout<<"iter = " << i << ",  " << "number  = " << number<< endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}
