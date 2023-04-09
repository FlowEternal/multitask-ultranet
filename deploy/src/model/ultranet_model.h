// ——————————————————————————————————————————————————————————————————————————————
// File Name	:ultranet_model.h
// Abstract 	:ultranet
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/09/12
// ——————————————————————————————————————————————————————————————————————————————
#ifndef ONNX_Ultranet_MODEL_H
#define ONNX_Ultranet_MODEL_H

#include <string>
#include <algorithm>  

#if defined (_WINDOWS)
#include <Windows.h>
#else
#include <sys/time.h> 
#endif

// —————————————————————————
// ———————— 接口头文件 ———————
// —————————————————————————
#include <Ultranet.h>

// ——————————————————————————
// ———————— ONNX头文件 ———————
// ——————————————————————————
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

#define USING_TENSORRT_ARM true
#if not defined (_WINDOWS)
#if USING_TENSORRT_ARM
#include <tensorrt_provider_factory.h>
#endif
#endif 

// ————————————————————————————
// ———————— xml解析头文件 ———————
// ————————————————————————————
#include "tinystr.h"  
#include "tinyxml.h"

// ————————————————————————————————————————
// ———————— Network输入输出名称宏定义 ————————
// ————————————————————————————————————————
#define NET_INPUT_NAME							"input"
#define OUTPUT_SEG								"output_seg"
#define OUTPUT_DETECTION_TOPK_SCORES			"topk_scores"
#define OUTPUT_DETECTION_TOPK_INDEX				"topk_inds"
#define OUTPUT_DETECTION_WH_PREDICT				"wh_pred"
#define OUTPUT_DETECTION_OFFSET_PREDICT			"offset_pred"
#define OUTPUT_LANE_CONFIDENCE					"out_confidence"
#define OUTPUT_LANE_OFFSET						"out_offset"
#define OUTPUT_LANE_INSTANCE					"out_instance"

// ————————————————————————————————
// ———————— Network维度定义 ————————
// ————————————————————————————————
#define NET_INPUT_HEIGHT				640
#define NET_INPUT_WIDTH					640
#define NET_INPUT_CHANNEL				3
#define RATIO_SCALE						16
#define LANE_FEAT_DIM					4			// 车道线聚类维度
#define TOPK_NUM						100			// 目标检测topk
#define FEAT_MAP_WIDTH					(NET_INPUT_WIDTH/RATIO_SCALE)
#define FEAT_MAP_HEIGHT					(NET_INPUT_HEIGHT/RATIO_SCALE)
static constexpr const int INPUT_WIDTH = NET_INPUT_WIDTH;
static constexpr const int INPUT_HEIGHT = NET_INPUT_HEIGHT;
static constexpr const int INPUT_CHANNEL = NET_INPUT_CHANNEL;
static constexpr const int FEAT_WIDTH = FEAT_MAP_WIDTH;
static constexpr const int FEAT_HEIGHT = FEAT_MAP_HEIGHT;
static constexpr const int LANE_FEATURE_DIM = LANE_FEAT_DIM;
static constexpr const int Topk_num = TOPK_NUM;

struct tunning_param
{
	// 车道线
	float lane_conf_thres = 0.8;
	float lane_cluster_thres = 0.002;
	float min_pts_num = 6;

	// 目标检测
	float det_conf_thres = 0.35;
	float det_iou_thres = 0.4;

	// 模型路径
	std::string model_path = "";
};

namespace ultranet 
{

	namespace ultranet_detection 
	{

		class ultranet_model
		{

		public:
			// default constructor
			ultranet_model(std::string cfg_path);

			bool parse_xml_cfg(std::string cfg_path);

			void detect(const cv::Mat& input_image,cv::Mat& visual_image, Output_Info output_info);

		private:

			// inference engine
			Ort::Session session_{ nullptr };
			Ort::Env env_{nullptr};

			// timer
			std::chrono::steady_clock::time_point tic;
			std::chrono::steady_clock::time_point tac;
			std::chrono::steady_clock::time_point tic_inner;
			std::chrono::steady_clock::time_point tac_inner;
			double time_used = 0;

			// image original size
			cv::Size _m_input_node_size_host = cv::Size(INPUT_WIDTH,INPUT_HEIGHT);

			// original image dimension
			float org_img_width = 0.0f;
			float org_img_height = 0.0f;

			// tunning param
			tunning_param params;

			// ———————————————————————————————————————
			// ———————————  input related ———————————
			// ———————————————————————————————————————
			std::array<float, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL> input_image_{};
			std::array<int64_t, 4> input_image_shape_{ 1,INPUT_CHANNEL,INPUT_HEIGHT, INPUT_WIDTH };
			Ort::Value input_image_tensor_{ nullptr };

			// 前处理
			void preprocess(const cv::Mat& input_image, cv::Mat& output_image, cv::Mat &visual_img);

			// ———————————————————————————————————————
			// ———————————  lane detection ———————————
			// ———————————————————————————————————————
			// out_confidence
			std::array<float, 1 * FEAT_HEIGHT * FEAT_WIDTH> out_confidence_{};
			std::array<int64_t, 4> out_confidence_shape_{ 1, 1, FEAT_HEIGHT, FEAT_WIDTH };
			Ort::Value out_confidence_tensor_{ nullptr };

			// out_offset
			std::array<float, 2 * FEAT_HEIGHT * FEAT_WIDTH> out_offset_{};
			std::array<int64_t, 4> out_offset_shape_{ 1, 2, FEAT_HEIGHT, FEAT_WIDTH };
			Ort::Value out_offset_tensor_{ nullptr };

			// out_instance
			std::array<float, LANE_FEATURE_DIM * FEAT_HEIGHT * FEAT_WIDTH> out_instance_{};
			std::array<int64_t, 4> out_instance_shape_{ 1, LANE_FEATURE_DIM, FEAT_HEIGHT, FEAT_WIDTH };
			Ort::Value out_instance_tensor_{ nullptr };

			// 车道线后处理
			void postprocess_lane(float * out_confidence_ptr,
								  float * out_offset_ptr,
								  float * out_instance_ptr,
								  std::vector< Lane_Info > & process_result,
								  cv::Mat & visual_image);

			void draw_lane_line(Lane_Info & one_lane, cv::Mat& visual_img);

			// ———————————————————————————————————————
			// ———————————  segmentation  ———————————
			// ———————————————————————————————————————
			std::array<int64_t, INPUT_WIDTH * INPUT_HEIGHT> seg_results_{};
			std::array<int64_t, 3> seg_output_shape_{ 1,INPUT_HEIGHT, INPUT_WIDTH };
			Ort::Value seg_output_tensor_{ nullptr };

			// 分割后处理
			void postprocess_seg(int64_t* output_data_ptr, cv::Mat& visual_img, cv::Mat& seg_mask);

			// ——————————————————————————————————————————
			// ———————————  object detection  ———————————
			// ——————————————————————————————————————————
			// topk_scores
			std::array<float, 1 * Topk_num > det_topk_scores_{};
			std::array<int64_t, 2> det_topk_scores_shape_{ 1,Topk_num };
			Ort::Value det_topk_scores_tensor_{ nullptr };

			// topk_inds
			std::array<int64_t, 1 * Topk_num > det_topk_inds_{};
			std::array<int64_t, 2> det_topk_inds_shape_{ 1,Topk_num };
			Ort::Value det_topk_inds_tensor_{ nullptr };

			// wh_pred
			std::array<float, 1 * 2 * FEAT_HEIGHT * FEAT_WIDTH> det_wh_pred_{};
			std::array<int64_t, 4> det_wh_pred_shape_{ 1, 2, FEAT_HEIGHT, FEAT_WIDTH };
			Ort::Value det_wh_pred_tensor_{ nullptr };

			// offset_pred
			std::array<float, 1 * 2 * FEAT_HEIGHT * FEAT_WIDTH> det_offset_pred_{};
			std::array<int64_t, 4> det_offset_pred_shape_{ 1, 2, FEAT_HEIGHT, FEAT_WIDTH };
			Ort::Value det_offset_pred_tensor_{ nullptr };

			// 目标检测后处理
			void postprocess_detection(float* output_topk_scores_ptr,
									   int64 * output_topk_inds_ptr,
									   float * output_wh_pred_ptr,
									   float * output_offset_pred_ptr,
									   std::vector<Detection_Info>& detect_result,
									   cv::Mat& visual_img);


		};

	}

}


// utility function
float cal_iou(cv::Rect rect1, cv::Rect rect2);
bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect);
static float get_iou_value(cv::Rect2f rect1, cv::Rect2f rect2);
void nms_boxes_detect(std::vector<Detection_Info> & nms_detect_infos, std::vector<int> & index_choose, float threshold_iou);
bool sort_lane_y(MyPoint lsh, MyPoint rsh);

#if defined (_WINDOWS)
wchar_t * char2wchar(const char* cchar);
#endif

#endif //ONNX_Ultranet_MODEL_H
