/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {

    void Map::InsertKeyFrame(Frame::Ptr frame) {
        current_frame_ = frame; //这句话？
        if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {             //如果是新的关键帧（没有这个ID）
            keyframes_.insert(make_pair(frame->keyframe_id_, frame));               //插入关键帧到地图keyframes_
            active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));        //插入关键帧到窗口active_keyframes_
        } else {
            keyframes_[frame->keyframe_id_] = frame;                                //如果是已有的关键帧（有这个ID）
            active_keyframes_[frame->keyframe_id_] = frame;                         //在地图keyframes_和窗口active_keyframes_中更新这个ID的关键帧
        }

        if (active_keyframes_.size() > num_active_keyframes_) {                     //如果窗口中的关键帧数量超过了窗口限制
            RemoveOldKeyframe();                                                    //移除旧的关键帧（移除规则自己定）
        }
    }

    void Map::InsertFrame(Frame::Ptr frame) {
        current_frame_ = frame; ///这句话？
        if (frames_.find(frame->frame_id_) == frames_.end()) {             //如果是新的帧（没有这个ID）
            frames_.insert(make_pair(frame->frame_id_, frame));

        } else {
            frames_[frame->frame_id_] = frame;                                //如果是已有的帧（有这个ID）//在地图frames_和窗口active_keyframes_中更新这个ID的关键帧
        }

    }

    void Map::InsertRawFrame(Frame::Ptr frame) {
        if (frames_.find(frame->frame_id_) == frames_.end()) {             //如果是新的帧（没有这个ID）
            frames_.insert(make_pair(frame->frame_id_, frame));

        } else {
            frames_[frame->frame_id_] = frame;                                //如果是已有的帧（有这个ID）//在地图frames_和窗口active_keyframes_中更新这个ID的关键帧
        }

    }

    void Map::InsertMapPoint(MapPoint::Ptr map_point) {
        if (landmarks_.find(map_point->id_) == landmarks_.end()) {                  //如果是新的路标点（没有这个ID）
            landmarks_.insert(make_pair(map_point->id_, map_point));                //插入地图点到地图landmarks_
            active_landmarks_.insert(make_pair(map_point->id_, map_point));         //插入地图点到窗口active_landmarks_
        } else {                                                                    //如果是已有的路标点（有这个ID）
            landmarks_[map_point->id_] = map_point;                                 //在地图landmarks_和窗口active_landmarks_中更新这个ID的地图点
            active_landmarks_[map_point->id_] = map_point;
        }
    }

    void Map::RemoveOldKeyframe() {
        if (current_frame_ == nullptr) return;
        // 寻找与当前帧最近与最远的两个关键帧
        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame_->Pose().inverse();                    //当前帧c-->w的位姿
        for (auto& kf : active_keyframes_) {                            //遍历关键帧，寻找与当前帧最近与最远的两个关键帧
            if (kf.second == current_frame_) continue;
            auto dis = (kf.second->Pose() * Twc).log().norm();          //遍历的关键帧与当前帧的距离
            if (dis > max_dis) {                                        //找最远的，没什么好说的；记录最远距离和最远帧ID
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {                                        //找最近的，没什么好说的；记录最近距离和最近帧ID
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        const double min_dis_th = 0.2;  // 最近阈值
        Frame::Ptr frame_to_remove = nullptr;
        if (min_dis < min_dis_th) {
            // 如果存在很近的帧，优先删掉最近的
            frame_to_remove = keyframes_.at(min_kf_id);                 //.at( ), 访问哈希表，括号里是哈希表的key;
            // at访问的是哈希表的某个整个元素<first,second>
        } else {
            // 删掉最远的
            frame_to_remove = keyframes_.at(max_kf_id);
        }

        LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
        // remove keyframe and landmark observation
        active_keyframes_.erase(frame_to_remove->keyframe_id_);         //.erase(key)//从窗口active_keyframes中删除要删除的帧
        for (auto feat : frame_to_remove->features_left_) {             //遍历要被删除的帧中的特征，找到这些特征对应的地图点，删除这些特征对地图点的观测
            auto mp = feat->map_point_.lock();                          //遍历包括遍历左图与右图
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }
        for (auto feat : frame_to_remove->features_right_) {
            if (feat == nullptr) continue;
            auto mp = feat->map_point_.lock();
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }

        CleanMap();                                                     //从observations中删完观测到地图点的特征之后，清理地图
        //清理地图：从窗口中删除被0个特征观测到的路标点
        //即如果某个路标点没有再被特征观测到了，那么从窗口中删掉它
    }

    void Map::CleanMap() {
        int cnt_landmark_removed = 0;
        for (auto iter = active_landmarks_.begin();
             iter != active_landmarks_.end();) {
            if (iter->second->observed_times_ == 0) {
                iter = active_landmarks_.erase(iter);                   //erase(iterator):删除这个迭代器后，返回一个指向删除后位置的迭代器(删除后自动++）
                cnt_landmark_removed++;
            }
            else {
                ++iter;
            }
        }
        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }

}  // namespace myslam
