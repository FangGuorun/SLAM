#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {

    struct Frame;

    struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
    struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0;  // ID
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();  // Position in world
        std::mutex data_mutex_;
        int observed_times_ = 0;  // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_; //obserbations: list容器存储观测到该地图点的特征点

        MapPoint() {}

        MapPoint(long id, Vec3 position);

        Vec3 Pos() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Vec3 &pos) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        //地图点被特征点观测到的次数+1
        //feature:观测到该地图点的特征点
        void AddObservation(std::shared_ptr<Feature> feature) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);                       //添加观测到该地图点的特征点feature到observations中
            observed_times_++;                                      //该地图点被观测到的次数+1
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
