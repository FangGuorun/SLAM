//
// Created by gaoxiang on 19-5-4.
//
using namespace std;
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"
#include <iostream>     // std::cout, std::fixed
#include <iomanip>		// std::setprecision
namespace myslam {

    VisualOdometry::VisualOdometry(std::string &config_path)
            : config_file_path_(config_path) {}

    bool VisualOdometry::Init() {
        // read from config file
        if (Config::SetParameterFile(config_file_path_) == false) {
            return false;
        }

        dataset_ =
                Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
        CHECK_EQ(dataset_->Init(), true);

        // create components and links
        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);

        frontend_->SetBackend(backend_);
        frontend_->SetMap(map_);
        frontend_->SetViewer(viewer_);
        frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

        backend_->SetMap(map_);
        backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

        viewer_->SetMap(map_);

        return true;
    }

    void VisualOdometry::Run() {
        while (1) {
            LOG(INFO) << "VO is running";
            if (Step() == false) {
                break;
            }
        }

        backend_->Stop();
        viewer_->Close();

        LOG(INFO) << "VO exit";
    }

    //void VisualOdometry::SaveTrajectoryKITTI(const string &filename, myslam::Frame::Ptr keyframe){

      //  static int number = 0;
       // number++;
        ///keyframes_.matrix的格式与kitti相符
       // auto Tcw = keyframe->pose_;
        //auto Rcw = Tcw.rotationMatrix();
        //auto tcw = Tcw.translation();
      //  auto Twc = Tcw.inverse().matrix3x4();
        //auto Rwc = Twc.rotationMatrix();
        //auto twc = Twc.translation();

        //f << setprecision(9) << Twc(0,0) << " " << Twc(0,1)  << " " << Twc(0,2) << " "  << Twc(0,3) << " " <<
        //  Twc(1,0) << " " << Twc(1,1)  << " " << Twc(1,2) << " "  << Twc(1,3) << " " <<
        //  Twc(2,0) << " " << Twc(2,1)  << " " << Twc(2,2) << " "  << Twc(2,3) << endl;

        //f << setprecision(9) << Rcw(0,0) << " " << Rcw(0,1)  << " " << Rcw(0,2) << " "  << tcw(0) << " " <<
        //  Rcw(1,0) << " " << Rcw(1,1)  << " " << Rcw(1,2) << " "  << tcw(1) << " " <<
        //  Rcw(2,0) << " " << Rcw(2,1)  << " " << Rcw(2,2) << " "  << tcw(2) << endl;

       // f.close();
       // cout << endl << "frames count = " << number << endl;
   // }

    bool VisualOdometry::Step() {
        Frame::Ptr new_frame = dataset_->NextFrame();
        if (new_frame == nullptr) return false;

        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->AddFrame(new_frame);
        auto t2 = std::chrono::steady_clock::now();
        auto time_used =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
        //new_frame->pose_;
        //SaveTrajectoryKITTI("/home/guorunfang/slam_course/hw8/CameraTrajectory_twc.txt",new_frame);
        return success;
    }


}  // namespace myslam


