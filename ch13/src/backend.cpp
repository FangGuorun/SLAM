//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

    Backend::Backend() {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }

    void Backend::UpdateMap() {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.notify_one();
    }

    void Backend::Stop() {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::BackendLoop() {
        while (backend_running_.load()) {
            std::unique_lock<std::mutex> lock(data_mutex_);
            map_update_.wait(lock);

            /// 后端仅优化激活的Frames和Landmarks
            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_kfs, active_landmarks);
        }
    }

    void Backend::Optimize(Map::KeyframesType &keyframes,
                           Map::LandmarksType &landmarks) {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
                LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(
                        g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // pose 顶点，使用Keyframe id
        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_kf_id = 0;
        for (auto &keyframe : keyframes) {          //遍历关键帧，设置顶点，计数关键帧数量（后面路标点ID设置需要加上关键帧总数量）
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
            vertex_pose->setId(kf->keyframe_id_);
            vertex_pose->setEstimate(kf->Pose());
            optimizer.addVertex(vertex_pose);
            if (kf->keyframe_id_ > max_kf_id) {
                max_kf_id = kf->keyframe_id_;
            }

            vertices.insert({kf->keyframe_id_, vertex_pose});
        }

        // 路标顶点，使用路标id索引
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        // K 和左右外参
        Mat33 K = cam_left_->K();
        SE3 left_ext = cam_left_->pose();
        SE3 right_ext = cam_right_->pose();

        // edges
        int index = 1;
        double chi2_th = 5.991;  // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

        for (auto &landmark : landmarks) {                      //遍历地图中的landmark
            if (landmark.second->is_outlier_) continue;         //如果landmark是外点，不做处理
            unsigned long landmark_id = landmark.second->id_;
            auto observations = landmark.second->GetObs();
            for (auto &obs : observations) {                    //遍历这个landmark对应的所有观测(所有观测到这个landmark的特征)
                if (obs.lock() == nullptr) continue;            //obs:观测到地图点的单个特征(弱指针）
                auto feat = obs.lock();
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;//如果特征是外点或者特征没有对应帧

                auto frame = feat->frame_.lock();
                EdgeProjection *edge = nullptr;
                if (feat->is_on_left_image_) {                                    //判断特征在左图还是右图
                    edge = new EdgeProjection(K, left_ext);                       //在左图，传给二元边左目相机外参
                } else {
                    edge = new EdgeProjection(K, right_ext);                      //在右图，传给二元边右目相机外参
                }

                // 如果landmark还没有被加入优化，则新加一个顶点
                if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) { //vertices_landmarks(key是待优化的landmarkID，value是待优化的landmark顶点)
                    // 一开始为空，要从零开始添加//添加进去即被加入优化
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(landmark.second->Pos());                             //优化值：路标点坐标
                    v->setId(landmark_id + max_kf_id + 1);                              //优化ID：路标点在关键帧后面优化，所以要加上关键帧数量再加路标点自己的计数
                    v->setMarginalized(true);                                           //true,则会将这个顶点边缘化（这里是路标点）
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v);
                }
                ///一个边对应一个特征（观测），一个路标点(顶点2)（路标点位姿变换得到估计值2）, 一个当前帧位姿(顶点1)(估计值1)  边(error) = 观测值 - 估计值
                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
                edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark            //哈希表.at()函数，返回second
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge);

                index++;
            }
        }

        // do optimization and eliminate the outliers
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        //while (iteration < 5) {
        while (iteration < 10) {
            cnt_outlier = 0;
            cnt_inlier = 0;
            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges_and_features) {   //遍历上面得到的edges_and_features
                if (ef.first->chi2() > chi2_th) {   //判断边的误差是否大于阈值，大于的话，当前帧外点数目+1（这条边对应的特征为外点），否则内点（（这条边对应的特征为内点）数目+1
                    cnt_outlier++;
                } else {
                    cnt_inlier++;
                }
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5) {               //如果内点的比例大于一半，那么达到要求，迭代停止
                break;
            } else {
                chi2_th *= 2;                       //否则提高误差阈值(内点筛选力度)，继续迭代
                iteration++;
            }
        }

        for (auto &ef : edges_and_features) {           //迭代完成后，遍历edges_and_features，如果边的误差过大，把这条边对应的特征标记为外点，并且把对应地图点该特征的观测移除
            if (ef.first->chi2() > chi2_th) {
                ef.second->is_outlier_ = true;
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            } else {
                ef.second->is_outlier_ = false;         //如果边的误差没那么大，则不删特征和其地图点的观测
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier;

        // Set pose and lanrmark position               //遍历两个顶点容器，在窗口active_kfs,和active_landmarks中分别设置优化好的位姿和路标点坐标
        for (auto &v : vertices) {
            keyframes.at(v.first)->SetPose(v.second->estimate()); //keyframes:传入的active_kfs
        }
        for (auto &v : vertices_landmarks) {
            landmarks.at(v.first)->SetPos(v.second->estimate());  //landmarks:传入的active_landmarks
        }
    }

}  // namespace myslam // namespace myslam