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


#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {

    MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

    MapPoint::Ptr MapPoint::CreateNewMappoint() {           //创建新地图点:1.工厂id（静态初始化）  2.新地图点指针 3.新地图点对应工厂id+1     返回新地图点指针
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end();    //遍历观测，找到要删除的那个观测
             iter++) {                                                          //observations_里的元素就是std::weak_ptr<Feature>
            if (iter->lock() == feat) {
                observations_.erase(iter);
                feat->map_point_.reset();           //weak_ptr.reset():释放指针
                observed_times_--;
                break;
            }
        }
    }

}  // namespace myslam