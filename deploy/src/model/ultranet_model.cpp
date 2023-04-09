// ——————————————————————————————————————————————————————————————————————————————
// File Name	:ultranet_model.cpp
// Abstract 	:ultranet
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/09/12
// ——————————————————————————————————————————————————————————————————————————————

#include "ultranet_model.h"

// —————————————————————————————————————
// ———————————seg相关的显示参数———————————
// —————————————————————————————————————
std::vector<cv::Scalar> draw_color_vec = {
	cv::Scalar(0, 0, 0),
	cv::Scalar(128, 0, 128),
	cv::Scalar(128, 0, 128),
};

// —————————————————————————————————————
// ———————————det相关的显示参数———————————
// —————————————————————————————————————
std::vector<std::string> detect_vec = {
					  "roadtext",
					  "pedestrian",
					  "guidearrow",
					  "traffic",
					  "obstacle",
					  "vehicle_wheel",
					  "roadsign",
					  "vehicle",
					  "vehicle_light"
};

std::vector<cv::Scalar> detect_color_list = {
	cv::Scalar(0,252,124),
	cv::Scalar(0,255,127),
	cv::Scalar(255,255,0),
	cv::Scalar(220,245,245),
	cv::Scalar(255, 255, 240),
	cv::Scalar(205, 235, 255),
	cv::Scalar(196, 228, 255),
	cv::Scalar(212, 255, 127),
	cv::Scalar(226, 43, 138),
	cv::Scalar(135, 184, 222),
};

// —————————————————————————————————————
// ———————————lane相关的显示参数———————————
// —————————————————————————————————————
// lane pt显示参数
#define LANE_COLOR			cv::Scalar(0,255,0)
#define LANE_PT_SIZE		4
// lane box显示参数
#define WIDTH_BOX_BASE		float(20)		
#define HEIGHT_BOX_BASE		float(20)	
#define TEXT_SCALE			1					// "Lane"有四个字符
#define BOX_IOU_THRESHOLD	0.000001			// 不让box重合
// lane 文本显示参数
#define FONT_SCALE_TXT		0.7			
#define THICKNESS_TXT		2					
#define FONT_TYPE			cv::FONT_HERSHEY_COMPLEX
#define TEXT_COLOR			cv::Scalar(0,0,0)


namespace ultranet
{

	namespace ultranet_detection
	{

		ultranet_model::ultranet_model(std::string cfg_path)
		{

			env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Default");
			Ort::SessionOptions session_option;
			session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#if not defined (_WINDOWS)
			if (USING_TENSORRT_ARM)
			{
				Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
			}
#endif
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));


			// 解析配置文件
			parse_xml_cfg(cfg_path);


			// 测量模型加载时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(1) Start Model Loading" << std::endl;

#if defined (_WINDOWS)
			const ORTCHAR_T* model_path_convert = char2wchar(params.model_path.c_str());
			session_ = Ort::Session(env_, model_path_convert, session_option);

#else
			session_ = Ort::Session(env_, model_path.c_str(), session_option);
#endif 


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Model Loading Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << std::endl;
			// 测量模型加载时间 end


		}

		bool ultranet_model::parse_xml_cfg(std::string cfg_path)
		{
			//读取xml文件中的参数值
			TiXmlDocument* document = new TiXmlDocument();
			if (!document->LoadFile(cfg_path.c_str()))
			{
				std::cout << "无法加载xml文件！" << std::endl;
				std::cin.get();
				return false;
			}
			TiXmlElement* root_element = document->RootElement();		//根目录
			TiXmlElement* next_element = root_element->FirstChildElement();		//根目录下的第一个节点层

			while (next_element != NULL)		//判断有没有读完
			{


				// 获取网络输入大小
				if (next_element->ValueTStr() == "network")
				{

					TiXmlElement* sub_element = next_element->FirstChildElement();

					while (sub_element != NULL)
					{

						if (sub_element->ValueTStr() == "detection")
						{
							TiXmlElement* end_element = sub_element->FirstChildElement();
							params.det_conf_thres = atof(end_element->GetText());

							end_element = end_element->NextSiblingElement();
							params.det_iou_thres = atof(end_element->GetText());

						}

						if (sub_element->ValueTStr() == "lane")
						{
							TiXmlElement* end_element = sub_element->FirstChildElement();
							params.lane_conf_thres = atof(end_element->GetText());

							end_element = end_element->NextSiblingElement();
							params.lane_cluster_thres = atof(end_element->GetText());

							end_element = end_element->NextSiblingElement();
							params.min_pts_num = atoi(end_element->GetText());

						}

						sub_element = sub_element->NextSiblingElement();

					}



				}

				if (next_element->ValueTStr() == "demo")
				{

					TiXmlElement* sub_element = next_element->FirstChildElement();

					while (sub_element != NULL)
					{

						if (sub_element->ValueTStr() == "test_model_path")
						{
							params.model_path = sub_element->GetText();

						}


						sub_element = sub_element->NextSiblingElement();

					}



				}



				next_element = next_element->NextSiblingElement();
			}

			//释放内存
			delete document;
		}


		void ultranet_model::preprocess(const cv::Mat &input_image, cv::Mat& output_image, cv::Mat & visual_img)
		{

			// start
			tic_inner = std::chrono::steady_clock::now();

			if (input_image.size() != _m_input_node_size_host)
			{
				cv::resize(input_image, output_image, _m_input_node_size_host, 0, 0, cv::INTER_LINEAR);
			}


			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Image Resize Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << "-- Original Image Size: Height = " << input_image.size().height << ", Width = " << input_image.size().width << std::endl;
			std::cout << "-- Resized Image Size: Height = " << output_image.size().height << ", Width = " << output_image.size().width << std::endl;
			// end


			// start
			tic_inner = std::chrono::steady_clock::now();
			visual_img = output_image.clone();
			if (output_image.type() != CV_32FC3)
			{
				// 首先转化为RGB
				cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
				// 然后转化为float32
				output_image.convertTo(output_image, CV_32FC3);
				// 进行normalization
				cv::divide(output_image, cv::Scalar(255.0f, 255.0f, 255.0f), output_image);
				cv::subtract(output_image, cv::Scalar(0.485, 0.456, 0.406), output_image);
				cv::divide(output_image, cv::Scalar(0.229, 0.224, 0.225), output_image);
			}

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Converting Resized Image To Float32 Time Cost: " << time_used << "ms!" << std::endl;
			// end


		}

		void ultranet_model::detect(const cv::Mat& input_image,
			cv::Mat& visual_image,Output_Info output_info)
		{

			// 这里首先做一个可视化备份并获取原始图像尺寸
			org_img_height = input_image.rows;
			org_img_width = input_image.cols;
			std::cout << std::endl;
			std::cout << "(1) Start Input Tensor Preprocess" << std::endl;
			// —————————————————————————————————————
			// ——————————— prepare input ———————————
			// —————————————————————————————————————
			cv::Mat input_image_copy;
			input_image.copyTo(input_image_copy);

			// 测量preprocess时间 start
			tic = std::chrono::steady_clock::now();
			preprocess(input_image, input_image_copy, visual_image);
			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Preprocess Time Cost: " << time_used << "ms!" << std::endl;
			// 测量preprocess时间 end



			// 测量填充tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(2) Start Input Tensor Filling" << std::endl;
			float* input_image_ptr = input_image_.data();
#if defined(_WINDOWS)
			fill(input_image_.begin(), input_image_.end(), 0.f);
#else
#endif	
			const int row = INPUT_HEIGHT;
			const int col = INPUT_WIDTH;
			const int channel = INPUT_CHANNEL;

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					input_image_ptr[0 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 0]);
					input_image_ptr[1 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 1]);
					input_image_ptr[2 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 2]);
				}
			}

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Filling Time Cost: " << time_used << "ms!" << std::endl;
			// 测量填充tensor时间 end




			// 测量创建ORT Tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(3) Start Tensor Convert" << std::endl;

			// ——————————————————————————————
			// ——————————— tensor ———————————
			// ——————————————————————————————
			const char* input_names[] = { NET_INPUT_NAME };
			const char* output_names[] = { 
				OUTPUT_SEG, 
				OUTPUT_DETECTION_TOPK_SCORES,
				OUTPUT_DETECTION_TOPK_INDEX,
				OUTPUT_DETECTION_WH_PREDICT,
				OUTPUT_DETECTION_OFFSET_PREDICT,
				OUTPUT_LANE_CONFIDENCE,
				OUTPUT_LANE_OFFSET,
				OUTPUT_LANE_INSTANCE
			};

			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
			input_image_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				input_image_.data(),
				input_image_.size(),
				input_image_shape_.data(),
				input_image_shape_.size());

			std::vector<Ort::Value> inputs_tensor;
			std::vector<Ort::Value> outputs_tensor;
			inputs_tensor.push_back(std::move(input_image_tensor_));

			////////////////////////////////
			// output seg
			seg_output_tensor_ = Ort::Value::CreateTensor<int64>(memory_info,
				seg_results_.data(),
				seg_results_.size(),
				seg_output_shape_.data(),
				seg_output_shape_.size());

			outputs_tensor.push_back(std::move(seg_output_tensor_));

			////////////////////////////////
			// output topk_scores
			det_topk_scores_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_topk_scores_.data(),
				det_topk_scores_.size(),
				det_topk_scores_shape_.data(),
				det_topk_scores_shape_.size());


			// output topk_inds
			det_topk_inds_tensor_ = Ort::Value::CreateTensor<int64>(memory_info,
				det_topk_inds_.data(),
				det_topk_inds_.size(),
				det_topk_inds_shape_.data(),
				det_topk_inds_shape_.size());


			// output det_wh_pred_
			det_wh_pred_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_wh_pred_.data(),
				det_wh_pred_.size(),
				det_wh_pred_shape_.data(),
				det_wh_pred_shape_.size());

			// output det_offset_pred_
			det_offset_pred_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_offset_pred_.data(),
				det_offset_pred_.size(),
				det_offset_pred_shape_.data(),
				det_offset_pred_shape_.size());

			outputs_tensor.push_back(std::move(det_topk_scores_tensor_));
			outputs_tensor.push_back(std::move(det_topk_inds_tensor_));
			outputs_tensor.push_back(std::move(det_wh_pred_tensor_));
			outputs_tensor.push_back(std::move(det_offset_pred_tensor_));


			////////////////////////////////
			// out_confidence_
			out_confidence_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				out_confidence_.data(),
				out_confidence_.size(),
				out_confidence_shape_.data(),
				out_confidence_shape_.size());

			// out_offset_
			out_offset_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				out_offset_.data(),
				out_offset_.size(),
				out_offset_shape_.data(),
				out_offset_shape_.size());

			// out_instance_
			out_instance_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				out_instance_.data(),
				out_instance_.size(),
				out_instance_shape_.data(),
				out_instance_shape_.size());


			outputs_tensor.push_back(std::move(out_confidence_tensor_));
			outputs_tensor.push_back(std::move(out_offset_tensor_));
			outputs_tensor.push_back(std::move(out_instance_tensor_));


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Tensor Convert Time Cost: " << time_used << "ms!" << std::endl;
			// 测量创建ORT Tensor时间 end


			// Single Forward Inference start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(4) Start Single Forward Inference" << std::endl;

			session_.Run(Ort::RunOptions{ nullptr },
				input_names,
				inputs_tensor.data(), inputs_tensor.size(),
				output_names,
				outputs_tensor.data(), outputs_tensor.size());

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Single Forward Inference Time Cost: " << time_used << "ms!" << std::endl;
			// Single Forward Inference end


			// PostProcess start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(5) Start PostProcessing" << std::endl;

			// ————————————————————————————
			// ———————— 处理输出张量 ————————
			// ————————————————————————————

			// 处理语义分割
			int64* output_seg_ptr = outputs_tensor[0].GetTensorMutableData<int64>();
			postprocess_seg(output_seg_ptr, visual_image, output_info.seg_mask);

			// 处理目标
			float* output_topk_scores_ptr = outputs_tensor[1].GetTensorMutableData<float>();
			int64* output_topk_inds_ptr = outputs_tensor[2].GetTensorMutableData<int64>();
			float* output_wh_pred_ptr = outputs_tensor[3].GetTensorMutableData<float>();
			float* output_offset_pred_ptr = outputs_tensor[4].GetTensorMutableData<float>();

			postprocess_detection(output_topk_scores_ptr,
				output_topk_inds_ptr,
				output_wh_pred_ptr, 
				output_offset_pred_ptr,
				output_info.detector,
				visual_image);



			// 处理车道线
			float* out_confidence_ptr = outputs_tensor[5].GetTensorMutableData<float>();
			float* out_offset_ptr = outputs_tensor[6].GetTensorMutableData<float>();
			float* out_instance_ptr = outputs_tensor[7].GetTensorMutableData<float>();
			postprocess_lane(out_confidence_ptr, out_offset_ptr, out_instance_ptr, output_info.lanes, visual_image);


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--PostProcessing Time Cost: " << time_used << "ms!" << std::endl;
			// Postprocess end

		}



		void ultranet_model::postprocess_lane(float * out_confidence_ptr,
			float * out_offset_ptr,
			float * out_instance_ptr,
			std::vector< Lane_Info > & process_result,
			cv::Mat & visual_image)
		{

			std::vector<float>		conf_list;
			std::vector<cv::Point>	candidate_pts;
			std::vector<std::array<float, LANE_FEAT_DIM>> candidate_features;

			// 首先过滤出大于置信阈值的所有在图像内的点
			for (int row = 0; row < FEAT_HEIGHT; row++)
			{
				for (int col = 0; col < FEAT_WIDTH; col++)
				{
					float score = *(out_confidence_ptr + FEAT_WIDTH * row + col);
					if (score < params.lane_conf_thres)
					{
						continue;
					}

					float offset_x = *(out_offset_ptr + 0 * FEAT_WIDTH*FEAT_HEIGHT + FEAT_WIDTH * row + col);
					float offset_y = *(out_offset_ptr + 1 * FEAT_WIDTH*FEAT_HEIGHT + FEAT_WIDTH * row + col);
					
					float pt_x = (col + offset_x)*RATIO_SCALE;
					float pt_y = (row + offset_y)*RATIO_SCALE;

					if (!((pt_x>=0) && (pt_x < NET_INPUT_WIDTH) && (pt_y >= 0) && (pt_y < NET_INPUT_HEIGHT)))
					{
						continue;
					}

					// feature
					std::array<float, LANE_FEAT_DIM> feature;
					for (int k = 0; k < LANE_FEAT_DIM;k++)
					{
						feature[k] = *(out_instance_ptr + k * FEAT_WIDTH*FEAT_HEIGHT + FEAT_WIDTH * row + col);
					}

					candidate_pts.push_back(cv::Point(pt_x, pt_y));
					candidate_features.push_back(feature);
					conf_list.push_back(score);
				}
			}

			// 根据feature进行聚类操作
			std::vector<std::array<float, LANE_FEAT_DIM>> assist_features;
			std::vector<std::vector<MyPoint>>			  lanes;

			for (int iter_index=0;iter_index<candidate_pts.size();iter_index++)
			{
				if (assist_features.size() == 0)
				{
					std::vector<MyPoint> one_lane;
					MyPoint tmp_pt;
					tmp_pt.pt = candidate_pts[iter_index];
					tmp_pt.conf_ = conf_list[iter_index];
					one_lane.push_back(tmp_pt);

					lanes.push_back(one_lane);
					assist_features.push_back(candidate_features[iter_index]);
				}

				else
				{
					int min_feature_index = -1;
					float	min_feature_dis = 10000;
					for (int feat_idx=0;feat_idx<assist_features.size();feat_idx++)
					{
						double distance = 0.0f;
						for (int m = 0; m < LANE_FEAT_DIM;m++)
						{
							distance += pow(assist_features[feat_idx][m] - candidate_features[iter_index][m],4);
						}

						distance = sqrt(distance);
						if (distance < min_feature_dis)
						{
							min_feature_dis = distance;
							min_feature_index = feat_idx;
						}
						
					}

					if (min_feature_dis < params.lane_cluster_thres)
					{
						// 同一类
						for (int m = 0; m < LANE_FEAT_DIM; m++)
						{
							assist_features[min_feature_index][m]= (assist_features[min_feature_index][m]*lanes[min_feature_index].size()
								+ candidate_features[iter_index][m])/(lanes[min_feature_index].size() + 1);
						}


						MyPoint tmp_pt;
						tmp_pt.pt = candidate_pts[iter_index];
						tmp_pt.conf_ = conf_list[iter_index];
						lanes[min_feature_index].push_back(tmp_pt);

					}

					else
					{
						// 不同类
						std::vector<MyPoint> one_lane;
						MyPoint tmp_pt;
						tmp_pt.pt = candidate_pts[iter_index];
						tmp_pt.conf_ = conf_list[iter_index];
						one_lane.push_back(tmp_pt);
						lanes.push_back(one_lane);
						assist_features.push_back(candidate_features[iter_index]);


					}


				}

			}

			// 准备出结果
			for (int index = 0; index < lanes.size(); index++)
			{
				std::vector<MyPoint> one_lane_info = lanes[index];

				if (one_lane_info.size() < params.min_pts_num)
				{
					continue;
				}

				Lane_Info output_one_lane_info;

				std::vector<MyPoint> output_one_lane_pts;

				float avg_score = 0.0;
				for (int pt_index = 0; pt_index < one_lane_info.size() ; pt_index++)
				{
					// scale back
					int scaled_x_coord = float(one_lane_info[pt_index].pt.x) / NET_INPUT_WIDTH * org_img_width;
					int scaled_y_coord = float(one_lane_info[pt_index].pt.y) / NET_INPUT_HEIGHT * org_img_height;
					MyPoint my_point;
					my_point.pt = cv::Point(scaled_x_coord, scaled_y_coord);
					my_point.conf_ = one_lane_info[pt_index].conf_;
					output_one_lane_pts.push_back(my_point);

					avg_score += one_lane_info[pt_index].conf_;

				}

				sort(output_one_lane_pts.begin(), output_one_lane_pts.end(), sort_lane_y);

				output_one_lane_info.lane_pts = output_one_lane_pts;
				output_one_lane_info.conf_score = avg_score/ one_lane_info.size(); // 平均置信度作为lane整体score
				process_result.push_back(output_one_lane_info);

			}


			// —————————————————————————————————————
			// ——————————— 可视化显示车道线 ———————————
			// —————————————————————————————————————
			std::vector<cv::Rect>	box_list;
			for (int lane_idx = 0; lane_idx < process_result.size(); lane_idx++)
			{

				std::vector<MyPoint> & tmp_pts = process_result[lane_idx].lane_pts;

				// —————————————————————————————————————
				// ——————————— 车道线点连成线 —————————————
				// —————————————————————————————————————
				draw_lane_line(process_result[lane_idx], visual_image);

				std::string info_type = "Lane"; // 这里用choose_lane_info的type

				int width_box = int(WIDTH_BOX_BASE / TEXT_SCALE * (info_type.length()));
				int height_box = int(HEIGHT_BOX_BASE);

				// —————————————————————————————————————————
				// ——————————— 显示车道线的置信度 —————————————
				// —————————————————————————————————————————
				float cof_score_round = process_result[lane_idx].conf_score;
				std::string info_conf = std::to_string(cof_score_round).substr(0, 4);


				// 这里开始进行box的画图
				// 保证每个box不相交
				float text_x = 0;
				float text_y = 0;
				int counter = -1;
				cv::Rect text_box;
				cv::Point pt1;
				cv::Point pt2;

				do
				{

					counter++; // 从0 开始
					if (counter >= process_result[lane_idx].lane_pts.size())
					{
						break;
					}

					text_x = process_result[lane_idx].lane_pts[counter].pt.x / org_img_width * NET_INPUT_WIDTH;
					text_y = process_result[lane_idx].lane_pts[counter].pt.y / org_img_height * NET_INPUT_HEIGHT;
					pt1 = cv::Point(text_x, text_y);
					pt2 = cv::Point(text_x + width_box, text_y - 2 * height_box);
					text_box = cv::Rect(pt1, pt2);

				} while ((text_x + width_box >= NET_INPUT_WIDTH) || (is_overlap_with_any(box_list, text_box)));

				box_list.push_back(text_box);
				cv::rectangle(visual_image, text_box, LANE_COLOR, -1); 
				cv::Point text_center_conf = cv::Point(text_x, text_y);
				cv::Point text_center_type = cv::Point(text_x, text_y - height_box);

				// line type
				cv::putText(visual_image, info_conf, text_center_conf,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);

				// confidence
				cv::putText(visual_image, info_type, text_center_type,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);


			}



		}

		void ultranet_model::draw_lane_line(Lane_Info & one_lane, cv::Mat & visual_img)
		{
			std::vector<MyPoint> tmp_pts = one_lane.lane_pts;

			for (int idx = 0; idx < tmp_pts.size(); idx++)
			{

				float coord_x_ = tmp_pts[idx].pt.x / org_img_width * NET_INPUT_WIDTH;
				float coord_y_ = tmp_pts[idx].pt.y / org_img_height * NET_INPUT_HEIGHT;
				cv::Point2f pt = cv::Point2f(coord_x_, coord_y_);
				cv::circle(visual_img, pt, LANE_PT_SIZE, LANE_COLOR, -1);

			}
		}

		// ——————————————————————————————————
		// ———————— 语义分割decode函数 ————————
		// ——————————————————————————————————
		void ultranet_model::postprocess_seg(int64_t* output_data_ptr, cv::Mat& visual_img, cv::Mat& seg_mask)
		{
			seg_mask = cv::Mat::zeros(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC1);
			uchar* seg_mask_ptr = seg_mask.data;
			cv::Mat org_img = visual_img.clone();

			// 进行softmax和期望值计算
			for (int height = 0; height < seg_mask.rows; height++)
			{
				for (int width = 0; width < seg_mask.cols; width++)
				{

					int64 tmp_cls_id = *(output_data_ptr + height * seg_mask.cols + width);
					*(seg_mask_ptr + height * seg_mask.cols + width) = (uchar)tmp_cls_id;

				}


			}


			// 可视化
			for (int height = 0; height < visual_img.rows; height++)
			{
				for (int width = 0; width < visual_img.cols; width++)
				{

					int index = *((uchar*)seg_mask.data + height * visual_img.cols + width);
					visual_img.ptr<uchar>(height)[width * 3 + 0] = draw_color_vec[index].val[0];
					visual_img.ptr<uchar>(height)[width * 3 + 1] = draw_color_vec[index].val[1];
					visual_img.ptr<uchar>(height)[width * 3 + 2] = draw_color_vec[index].val[2];

				}


			}


			// resize回原尺寸并可视化
			cv::resize(seg_mask, seg_mask, cv::Size(org_img_width, org_img_height), cv::INTER_NEAREST);

			// mask和原图进行叠加
			cv::addWeighted(org_img, 0.8, visual_img, 0.5, 0.0, visual_img);



		}


		void ultranet_model::postprocess_detection(float* output_topk_scores_ptr,
			int64 * output_topk_inds_ptr,
			float * output_wh_pred_ptr,
			float * output_offset_pred_ptr,
			std::vector<Detection_Info>& detect_result,
			cv::Mat& visual_img)
		{

			std::vector<Detection_Info> detect_infos;

			for (int propose_index = 0; propose_index < TOPK_NUM; propose_index++)
			{
				float  tmp_score = *(output_topk_scores_ptr + propose_index);
				int64  tmp_index = *(output_topk_inds_ptr + propose_index);

				if (tmp_score < params.det_conf_thres)
				{
					continue;
				}

				//
				int tmp_cls_index = tmp_index / (FEAT_WIDTH * FEAT_HEIGHT);
				int tmp_width_height_index = tmp_index % (FEAT_WIDTH * FEAT_HEIGHT);

				Detection_Info one_proposal;
				one_proposal.conf_score = tmp_score;
				one_proposal.class_name = detect_vec[tmp_cls_index];
				one_proposal.class_id = tmp_cls_index;


				// get center cell index
				int tmp_height_index = tmp_width_height_index / FEAT_WIDTH;
				int tmp_width_index = tmp_width_height_index % FEAT_WIDTH;


				// get width and hegiht
				float tmp_relative_width = *(output_wh_pred_ptr + 0 * FEAT_WIDTH * FEAT_HEIGHT + tmp_width_height_index);
				float tmp_relative_height = *(output_wh_pred_ptr + 1 * FEAT_WIDTH * FEAT_HEIGHT + tmp_width_height_index);


				// get offset
				float tmp_relative_offset_x = *(output_offset_pred_ptr + 0 * FEAT_WIDTH * FEAT_HEIGHT + tmp_width_height_index);
				float tmp_relative_offset_y = *(output_offset_pred_ptr + 1 * FEAT_WIDTH * FEAT_HEIGHT + tmp_width_height_index);

				// get precise location 
				float center_x = tmp_width_index + tmp_relative_offset_x;
				float center_y = tmp_height_index + tmp_relative_offset_y;

				// get x1, y1, x2, y2
				float xmin = (center_x - tmp_relative_width / 2.0)*RATIO_SCALE;
				float xmax = (center_x + tmp_relative_width / 2.0)*RATIO_SCALE;
				float ymin = (center_y - tmp_relative_height / 2.0)*RATIO_SCALE;
				float ymax = (center_y + tmp_relative_height / 2.0)*RATIO_SCALE;

				// clip out of box
				if (xmin < 0)
				{
					xmin = 0;
				}

				if (ymin < 0)
				{
					ymin = 0;
				}

				if (xmax > NET_INPUT_WIDTH - 1)
				{
					xmax = NET_INPUT_WIDTH - 1;
				}


				if (ymax > NET_INPUT_HEIGHT - 1)
				{
					ymax = NET_INPUT_HEIGHT - 1;
				}

				one_proposal.x1 = xmin;
				one_proposal.y1 = ymin;
				one_proposal.x2 = xmax;
				one_proposal.y2 = ymax;

				detect_infos.push_back(one_proposal);

			}


			std::vector<int>  index_choose;
			nms_boxes_detect(detect_infos, index_choose,params.det_iou_thres);

			std::vector<Detection_Info> detect_infos_refine;
			for (int i = 0;i<index_choose.size();i++)
			{
				detect_infos_refine.push_back(detect_infos[index_choose[i]]);
			}

			// 输出结果
			for (int j = 0; j < detect_infos_refine.size(); j++)
			{
				Detection_Info tmp_info;
				tmp_info.class_id = detect_infos_refine[j].class_id;
				tmp_info.class_name = detect_infos_refine[j].class_name;
				tmp_info.conf_score = detect_infos_refine[j].conf_score;
				tmp_info.x1 = detect_infos_refine[j].x1 / float(NET_INPUT_WIDTH) * org_img_width;
				tmp_info.x2 = detect_infos_refine[j].x2 / float(NET_INPUT_WIDTH) * org_img_width;

				tmp_info.y1 = detect_infos_refine[j].y1 / float(NET_INPUT_HEIGHT) * org_img_height;
				tmp_info.y2 = detect_infos_refine[j].y2 / float(NET_INPUT_HEIGHT) * org_img_height;

				detect_result.push_back(tmp_info);

			}

			// 可视化
			for (int j = 0; j < detect_infos_refine.size(); j++)
			{

				std::string class_name = detect_infos_refine[j].class_name;
				int class_id = detect_infos_refine[j].class_id;

				float conf_score = detect_infos_refine[j].conf_score;
				int x1 = detect_infos_refine[j].x1;
				int y1 = detect_infos_refine[j].y1;
				int x2 = detect_infos_refine[j].x2;
				int y2 = detect_infos_refine[j].y2;


				// 开始显示
				int t1 = int(round(0.003 * MAX(NET_INPUT_HEIGHT, NET_INPUT_WIDTH)));
				cv::Scalar color = detect_color_list[class_id];
				cv::Point pt1 = cv::Point(x1, y1);
				cv::Point pt2 = cv::Point(x2, y2);
				rectangle(visual_img, pt1, pt2, color, t1);

				int tf = MAX( t1 - 2, 2);

				std::string conf_score_str = std::to_string(int(conf_score * 100));
				int baseline;
				cv::Size s_size = cv::getTextSize(conf_score_str, cv::FONT_HERSHEY_SIMPLEX, (float(t1) / 3), tf, &baseline);
				cv::Size t_size = cv::getTextSize(class_name, cv::FONT_HERSHEY_SIMPLEX, float(t1) / 3, tf, &baseline);
				cv::Point pt3 = cv::Point(x1 + t_size.width + s_size.width + 15, y1 - t_size.height - 3);
				cv::rectangle(visual_img, pt1, pt3, color, -1);


				std::string txt_str = class_name + ""+ conf_score_str + "%";
				cv::putText(visual_img, txt_str, cv::Point(x1, y1 - 2), 0, float(t1) / 3, cv::Scalar(0, 0, 0),tf, cv::FONT_HERSHEY_SIMPLEX);
					

			}

		}



	}


}


#pragma region =============================== utility function===============================

bool sort_lane_y(MyPoint lsh, MyPoint rsh)
{
	if (lsh.pt.y > rsh.pt.y)
		return true;
	else
		return false;
}

#if defined (_WINDOWS)
wchar_t * char2wchar(const char* cchar)
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}
#endif

// box iou
float cal_iou(cv::Rect rect1, cv::Rect rect2)
{
	//计算两个矩形的交集
	cv::Rect rect_intersect = rect1 & rect2;
	float area_intersect = rect_intersect.area();

	//计算两个举行的并集
	cv::Rect rect_union = rect1 | rect2;
	float area_union = rect_union.area();

	//计算IOU
	double IOU = area_intersect * 1.0 / area_union;

	return IOU;
}

bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect)
{

	for (int index = 0; index < box_list.size(); index++)
	{
		float iou = cal_iou(box_list[index], target_rect);
		if (iou > BOX_IOU_THRESHOLD)
		{
			return true;
		}
	}

	return false;

}

// nms部分
typedef struct {
	cv::Rect2f box;
	float score;
	int index;
}BBOX;

static float get_iou_value(cv::Rect2f rect1, cv::Rect2f rect2)
{
	float xx1, yy1, xx2, yy2;

	xx1 = MAX(rect1.x, rect2.x);
	yy1 = MAX(rect1.y, rect2.y);
	xx2 = MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
	yy2 = MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

	float insection_width, insection_height;
	insection_width = MAX(0, xx2 - xx1 + 1);
	insection_height = MAX(0, yy2 - yy1 + 1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(rect1.width*rect1.height + rect2.width*rect2.height - insection_area);
	iou = insection_area / union_area;
	return iou;
}

bool cmpScore_object(BBOX lsh, BBOX rsh) {
	if (lsh.score > rsh.score)
		return true;
	else
		return false;
}

//input:  boxes: 原始检测框集合;
//input:  score：confidence * class_prob
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output: indices  经过上面两个阈值过滤后剩下的检测框的index
void nms_boxes_detect(std::vector<Detection_Info> & nms_detect_infos, std::vector<int> & index_choose, float threshold_iou)
{
	BBOX bbox;
	std::vector<BBOX> bboxes;
	int i, j;
	for (i = 0; i < nms_detect_infos.size(); i++)
	{
		Detection_Info tmp_info = nms_detect_infos[i];
		float _x = tmp_info.x1;
		float _y = tmp_info.y1;
		float _width = tmp_info.x2 - tmp_info.x1;
		float _height = tmp_info.y2 - tmp_info.y1;
		int class_index = tmp_info.class_id;

		cv::Rect2f tmp_rect(_x + class_index * NET_INPUT_WIDTH, _y + class_index * NET_INPUT_WIDTH, _width, _height);
		bbox.box = tmp_rect;
		bbox.score = tmp_info.conf_score;
		bbox.index = i;
		bboxes.push_back(bbox);
	}

	sort(bboxes.begin(), bboxes.end(), cmpScore_object);

	int updated_size = bboxes.size();
	for (i = 0; i < updated_size; i++)
	{

		index_choose.push_back(bboxes[i].index);
		for (j = i + 1; j < updated_size; j++)
		{

			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > threshold_iou)
			{

				bboxes.erase(bboxes.begin() + j);
				updated_size = bboxes.size();
				j--;

			}
		}
	}


}
#pragma endregion
