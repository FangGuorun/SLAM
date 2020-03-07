//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

    Frontend::Frontend() {
        gftt_ =
                cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
    }

    bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
        current_frame_ = frame;//传入帧为当前帧


        switch (status_) {              //三种跟踪状态
            case FrontendStatus::INITING:
                StereoInit();
                break;
            case FrontendStatus::TRACKING_GOOD:    //GOOD后没break,所以GOOD和BAD都会进入Track();break;
            case FrontendStatus::TRACKING_BAD:
                Track();
                break;
            case FrontendStatus::LOST:
                Reset();
                break;
        }

        //添加所有未丢失的帧到Frames_
        if (status_ != FrontendStatus::LOST) {
            current_frame_->SetFrame();
            map_->InsertFrame(current_frame_);
        }

        last_frame_ = current_frame_;   //当前帧变为次新帧
        return true;
    }

    bool Frontend::Track() {
        if (last_frame_) {              //last_frame==nullptr的情况:如果还没有次新帧，即初始化完的第一次T。
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
            ///last_frame_->Pose(): p_l_w     w-->l
            ///relative_motion: p_l_w * p_ll_w.inv() == p_l_ll    ll:次次新帧（上上帧）  ***标记问题 ，已解决
            ///现在没有当前帧的位姿，所以只能用已有的结果(上上一帧ll和上一帧l的相对位姿)来作为当前帧近似的初始值
            ///默认 相邻帧的变化没有那么大，所以 这样的近似是合理的
            ///只是提供一个比较相近的初值，总比完全不合理的初值要好
        }

        int num_track_last = TrackLastFrame();          //找到了好的当前帧的特征并存放在current_frame_->features_left_中
        tracking_inliers_ = EstimateCurrentPose();      //根据找到的好的特征，追踪上一帧，估计(优化)当前帧的位姿(pose_c_w)；同时得到当前帧与上一帧匹配的较好的关键帧的数目(内点）

        ///跟踪的三种情况:
        ///GOOD: 跟踪内点 > 跟踪特征good阈值
        ///BAD:  跟踪内点 > 跟踪特征bad阈值
        ///LOST: 其余情况( < 跟踪特征bad阈值)
        if (tracking_inliers_ > num_features_tracking_) {
            // tracking good
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if (tracking_inliers_ > num_features_tracking_bad_) {
            // tracking bad
            status_ = FrontendStatus::TRACKING_BAD;
        } else {
            // lost
            status_ = FrontendStatus::LOST;
        }
        //进行是否添加关键帧的判别
        InsertKeyframe();
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
        //relative_motion = p_c_w * p_l_w.inv()
        if (viewer_) viewer_->AddCurrentFrame(current_frame_);
        return true;
    }

    bool Frontend::InsertKeyframe() {
        if (tracking_inliers_ >= num_features_needed_for_keyframe_) {   //如果当前帧与上一阵的匹配点足够，那么不添加关键帧，只添加普通帧
            // still have enough features, don't insert keyframe,just insert frame
            //current_frame_->SetFrame();
            //map_->InsertFrame(current_frame_);//添加一帧

            return false;
        }
        //current frames is a frame!!(always)
        //current_frame_->SetFrame();
        //map_->InsertFrame(current_frame_);

        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        ///InsertKeyFrame(current_frame_)： 1.插入关键帧到地图keyframes_中
        ///                                 2.    2.1移除窗口中旧的关键帧(移除规则自己定，如移除与当前帧最近与最远的两个关键帧)（窗口是active_keyframes_）
        ///                                       2.2移除关键帧后，移除关键帧中 所有特征 对其 对应的路标点 的 观测（从map_point_中移除对应特征的观测:feature->map_point_.lock()->RemoveObservation(feature))
        ///                                       2.3清理地图，移除不再有特征观测到的地图点
        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_;

        SetObservationsForKeyFrame();
        ///设置对关键帧中特征的观测:
        /// 如果特征观测到了某个地图点(feat->map_point_.lock() != nullptr)，那么将这个特征添加到这个地图点的观测observations_中
        DetectFeatures();           // detect new features//和初始化中检测特征点的函数相同

        // track in right image
        FindFeaturesInRight();      //找到关键点在右图中的对应点
        // triangulate map points
        TriangulateNewPoints();     //三角化新的地图点并加入到地图中去
        // update backend because we have a new keyframe
        backend_->UpdateMap();      //后端更新，因为加入了新的关键帧

        if (viewer_) viewer_->UpdateMap();

        return true;
    }
//Frontend::SetObservationsForKeyFrame(): 设置关键帧中各个特征点对应的地图点的观测:地图点被观测到几次and地图点被哪些特征点观测
    void Frontend::SetObservationsForKeyFrame() {
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock(); //map_point是一个weak_ptr，map_point.lock:返回一个shared_ptr，指向map_point这个weak_ptr指向的对象。
            if (mp) mp->AddObservation(feat); //如果特征观测到了地图点，调用AddObservation函数
        }
    }
//void Frontend::SetObservationsForKeyFrame() {
    //  for(auto &feat : current_frame_->features_left_ ){
    //      auto mp = feat->map_point_.lock();
    //      if (mp) mp->AddObservation(feat);
    //  }
//}


    int Frontend::TriangulateNewPoints() {
        std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
        SE3 current_pose_Twc = current_frame_->Pose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_left_[i]->map_point_.expired() &&          //weak_ptr<T> w,声明一个weak_ptr,指向T型对象。w.expired():如果w指向的对象还存在,返回false;否则，已经释放，返回true
                current_frame_->features_right_[i] != nullptr) {                    //这里是左图的特征点没有对应的地图点(未关联地图点)，而且关键点在右图存在对应点(存在右图匹配点)，那么尝试三角化
                // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
                std::vector<Vec3> points{
                        camera_left_->pixel2camera(                                     //像素坐标经过内参，得到相机坐标系下坐标points(左目和右目)
                                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                                     current_frame_->features_left_[i]->position_.pt.y)),
                        camera_right_->pixel2camera(
                                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                                     current_frame_->features_right_[i]->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0) {        //如果可以三角化并且深度大于0
                    auto new_map_point = MapPoint::CreateNewMappoint();             //创建一个地图点
                    pworld = current_pose_Twc * pworld;
                    new_map_point->SetPos(pworld);                                  //设置地图点的坐标(世界坐标系）
                    new_map_point->AddObservation(                                  //添加左目特征到对地图点的观测
                            current_frame_->features_left_[i]);
                    new_map_point->AddObservation(                                  //添加右目特征到对地图点的观测
                            current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;  //把地图点关联到左目特征
                    current_frame_->features_right_[i]->map_point_ = new_map_point; //把地图点关联到右目特征
                    map_->InsertMapPoint(new_map_point);                            //插入地图点到地图(landmarks_）和窗口(active_landmarks_)
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    int Frontend::EstimateCurrentPose() {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType;                       //块求解器:构建H和b(线性求解器所需要的)
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>     //线性求解器:求解H*delta_x=b
                LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(
                        g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());   //估计当前帧的位姿
        optimizer.addVertex(vertex_pose);

        // K
        Mat33 K = camera_left_->K();

        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            auto mp = current_frame_->features_left_[i]->map_point_.lock();
            if (mp) {
                features.push_back(current_frame_->features_left_[i]);
                EdgeProjectionPoseOnly *edge =
                        new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(                                            //观测值(测量值),相机观测，肯定是2维的（像素坐标）
                        toVec2(current_frame_->features_left_[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        //for (int iteration = 0; iteration < 4; ++iteration) {       //通过调整vertex众多特征的error来调整vertex的位姿pose
        for (int iteration = 0; iteration < 8; ++iteration) {
            vertex_pose->setEstimate(current_frame_->Pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            //每迭代优化一次，统计一次内外点(重投影误差过大的点为外点)
            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]->is_outlier_) {             //is_outlier初始值false
                    e->computeError();                      //计算的是重投影误差
                    ///观测值measurement是特征匹配得来的像素值(2维) //估计值estimate是位姿变换得来的像素值(2维): p_pixel = K*T*pos3d/p_pixel[2];   error = measurement - p_pixel.head<2>;

                }
                if (e->chi2() > chi2_th) {                  //e->chi2():表征error大小的一个东西，类似error经过信息矩阵////error过大，
                    features[i]->is_outlier_ = true;        //error过大，这条边对应的特征被设为外点
                    e->setLevel(1);                         //setLevel(1)：野值边，1代表不优化
                    cnt_outlier++;
                } else {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);                         //setLevel(0):0代表内点
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat : features) {
            if (feat->is_outlier_) {                    //如果确认某个feature是外点，那么reset它
                feat->map_point_.reset();
                feat->is_outlier_ = false;  // maybe we can still use it in future
            }
        }
        return features.size() - cnt_outlier;   //返回好的特征的数量     //什么是好的特征？当前帧追踪上一帧，重投影误差小于一定阈值的特征
    }

    int Frontend::TrackLastFrame() {
        //和初始化阶段类似，只是初始化时是右图追踪左图，track阶段是当前帧追踪上一帧
        // use LK flow to estimate points in the current image from the last image
        // GOOD track, only use left image.
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_) {       //遍历上一帧的左图特征，将上一帧特征对应的地图点投影到当前帧的左图，作为当前帧关键点的初值。
            if (kp->map_point_.lock()) {                     //kps_last
                // use project point
                auto mp = kp->map_point_.lock();
                auto px =
                        camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {                                        //如果上一帧关键点没有对应的地图点，那么上一帧关键点作为当前帧左图关键点的初值
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
                last_frame_->left_img_, current_frame_->left_img_, kps_last,
                kps_current, status, error, cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i) {            //status:当前帧和上一帧的光流匹配结果,好的流为1，不好的流为0
            if (status[i]) {                                    //整合关键点为特征指针，设置特征对应地图点，添加特征到特征哈希表
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }

        //for (size_t i=0; i<status.size(); ++i) {
        //    if(status[i]) {
        //        cv::KeyPoint kp(kps_current[i], 7);
        //        Feature::Ptr feature(new Feature(current_frame_, kp));
        //        feature->map_point_ = last_frame_->features_left_[i]->map_point_;
        //        current_frame_->features_left_.push_back(feature);
        //        num_good_pts++;
        //    }
        //}

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    bool Frontend::StereoInit() {                               //双目初始化
        int num_features_left = DetectFeatures();               //DetectFeatures():返回检测到的特征数量.这里是左图
        ///DetectFeatures()：检测左图中的关键点并添加到current_frame->features_left中，并返回检测到的特征数量;
        int num_coor_features = FindFeaturesInRight();          //FindFeaturesInRight()：返回右图中与左图匹配的特征数量。
        if (num_coor_features < num_features_init_) {           //如果匹配的特征数量(右图中好的关键点数量)小于初始化的特征数量阈值，初始化失败
            return false;
        }

        bool build_map_success = BuildInitMap();                //BuildInitMap():返回是否成功创建地图。
        ///BuildInitMap()：创建地图（初始化地图），添加关键帧到
        if (build_map_success) {                                //如果地图创建成功，
            status_ = FrontendStatus::TRACKING_GOOD;            //则status更新为good
            if (viewer_) {
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
            return true;
        }
        return false;
    }

    int Frontend::DetectFeatures() {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);       //创建一个左图大小的Mask，Mat(Size size, int type, const Scalar& s);CV_8UC1,1代表单通道
        for (auto &feat : current_frame_->features_left_) {
            ///初始化时应该还没有特征，怎么遍历阿？
            ///答： 初始化的时候，没有特征，features_left_为空，那遍历就直接退出来了，那mask就是全图了。之后有了特征了，mask再变化
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
        }

        ///      CV_EXPORTS_W void rectangle(    InputOutputArray img, Point pt1, Point pt2,
        ///                                      const Scalar& color, int thickness = 1,
        ///                                      int lineType = LINE_8, int shift = 0);
        ///      cv::rectangle(mask,p1,p2,0,CV_FILLED)，每次循环，都在mask中取p1（左上顶点）和p2(右下顶点）构成的长方形
        ///      多次循环后，mask中就有很多很多长方形，
        ///      那么，检测特征点时就不在整个图检测，而只在mask中所有的长方形里去搜索
        ///      这里是，特征点左边取10，特征点右边取10，特征点在中间构成的长方形，
        ///      (检测每个特征点都在所有的长方形中去检测)
        ///      好处:1.缩小特征点检测范围，加速 2.避免了无关区域产生特征点的误匹配(如空白区域，墙面等）

        std::vector<cv::KeyPoint> keypoints;        //std::vector<cv::KeyPoint> keypoints存放keypoint
        gftt_->detect(current_frame_->left_img_, keypoints, mask);//检测出current_frame_->left_img_图像中mask区域内的keypoint存放在keypoints中

        ///      CV_WRAP virtual void detect( InputArray image,
        ///                                  CV_OUT std::vector<KeyPoint>& keypoints,
        ///                                 InputArray mask=noArray() );
        ///     Detector->detect(image, keypoints, mask) ,image:图像；keypoints:存放detect出的keypoint的vector；mask：只在mask区域内搜寻特征点，不管其它区域

        int cnt_detected = 0;
        for (auto &kp : keypoints) {                              //把检测到的gftt keypoint添加到当前帧的左图特征current_frame->features_left中去
            current_frame_->features_left_.push_back(
                    Feature::Ptr(new Feature(current_frame_, kp)));   //Feature::Ptr(new Feature(frame, keypoint)) //括号内是指针来创建指针
            cnt_detected++;                                       //当前帧检测到的关键点数目+1（当前帧追踪到的关键点数目+1？）（关键点能被当前帧检测到的数目+1）
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;                                      //返回当前帧检测到的关键点数目
    }

    int Frontend::FindFeaturesInRight() {       //1.找到右图的特征坐标。怎么找呢？由左图特征得到地图点，地图点投影到右图，得到右图的特征坐标。
        //2.计算左图到右图的光流跟踪，得到右图中好的关键点的数量
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right; //存放左图关键点坐标，右图关键点坐标，类型为cv::Point2f
        for (auto &kp : current_frame_->features_left_) {  //遍历左图特征,来添加左、右图关键点坐标
            kps_left.push_back(kp->position_.pt); //添加左图关键点坐标到kps_left// kp : std::shared_ptr<Feature>(特征类指针);  kp->position : 左图KeyPoint;  kp->position.pt : coordinate(坐标)
            auto mp = kp->map_point_.lock(); //kp->map_point是weak_ptr,调用lock()函数使用其对应的shared_ptr//mp:shared_ptr<MapPoint>，如果mp不为空，说明kp->map_point不为空，即关键点对应的地图点存在
            if (mp) {                        //如果该关键点对应的地图点存在，使用地图点在右图的投影坐标作为右图关键点坐标（就是之后的观测坐标（由特征匹配得到的特征坐标作为观测坐标）），并添加到kps_right
                // use projected points as initial guess
                auto px =
                        camera_right_->world2pixel(mp->pos_, current_frame_->Pose());//这里的pose是从W到C
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            } else {                         //否则，使用与左图的关键点相同的坐标，添加到kps_right
                // use same pixel in left image
                kps_right.push_back(kp->position_.pt);
            }
        }



        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(                                                         //计算左图到右图的光流跟踪
                current_frame_->left_img_, current_frame_->right_img_, kps_left,
                kps_right, status, error, cv::Size(11, 11), 3,                                //status[i]:  输出状态向量(无符号字符)，如果对应特征的流被找到，则元素设为1，否则设为0
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;       //num_good_pts
        //for (size_t i = 0; i < status.size(); ++i) {
        //   if (status[i]) {
        //       cv::KeyPoint kp(kps_right[i], 7);
        //      Feature::Ptr feat(new Feature(current_frame_, kp));
        //       feat->is_on_left_image_ = false;
        //      current_frame_->features_right_.push_back(feat);
        //     num_good_pts++;
        //  } else {
        //      current_frame_->features_right_.push_back(nullptr);
        //  }
        //}

        //复写
        for (size_t i=0; i<status.size(); ++i){                     //遍历status,

            if (status[i]){                                         //如果第i个特征的流被找到，
                cv::KeyPoint kp(kps_right[i], 7);                   //第i个右图关键点作为好的关键点，整合为特征，添加到当前帧的右图特征中去
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;                    //标示为该特征提在右图
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;                                     //好的关键点数目+1
            }

            else{                                                   //如果第i个特征的流没被找到，则没有对应的右图关键点
                current_frame_->features_right_.push_back(nullptr); //对应的当前帧右图特征添加为空指针
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;                                        //返回好的关键点的数目(好的关键点:status[i]==1,即能检测到左图到右图的流的关键点)
    }

    bool Frontend::BuildInitMap() {         // 1.通过左右图关键点的三角化得到地图点
        // 2.添加所有的地图点到地图；添加当前帧到地图
        std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
        size_t cnt_init_landmarks = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_right_[i] == nullptr) continue;
            // create map point from triangulation
            std::vector<Vec3> points{
                    camera_left_->pixel2camera(
                            Vec2(current_frame_->features_left_[i]->position_.pt.x,
                                 current_frame_->features_left_[i]->position_.pt.y)),
                    camera_right_->pixel2camera(
                            Vec2(current_frame_->features_right_[i]->position_.pt.x,
                                 current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {                    //如果三角化成功并且深度>0，那么创建新的地图点，并添加到地图中去
                auto new_map_point = MapPoint::CreateNewMappoint();
                new_map_point->SetPos(pworld);                                              //设置地图点世界坐标系坐标
                new_map_point->AddObservation(current_frame_->features_left_[i]);           //地图点被左图特征点i观测到，观测次数+1,
                new_map_point->AddObservation(current_frame_->features_right_[i]);          //地图点被右图特征点i观测到，观测次数+1,
                current_frame_->features_left_[i]->map_point_ = new_map_point;              //设置新地图点为左图特征点i对应的地图点
                current_frame_->features_right_[i]->map_point_ = new_map_point;             //设置新地图点为右图特征点i对应的地图点
                cnt_init_landmarks++;                                                       //初始化地图中的路标点个数+1
                map_->InsertMapPoint(new_map_point);
            }
        }



        current_frame_->SetKeyFrame();          //设置当前帧为关键帧
        map_->InsertKeyFrame(current_frame_);   //添加当前帧到地图
        backend_->UpdateMap();                  //后端，更新地图（这个看后端的时候再看）

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                  << " map points";

        return true;
    }

    bool Frontend::Reset() {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }

}  // namespace myslam