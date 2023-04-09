// ———————————————————————————————————
// ———————— Ultranet 头文件 ———————————
// ———————————————————————————————————
#include <Ultranet.h>
#include "tinystr.h"  
#include "tinyxml.h"

// param struct
struct demo_param
{

	int net_input_width = 0;
	int net_input_height = 0;

	int test_mode = 1;
	std::string test_model_path = "";
	std::string test_img_path = "";
	std::string test_video_path = "";
	std::string save_video_path = "";
	int camera_id = 0;

	// output video
	int gpu_warm_up = 10;
	int iteration = 20;
	int output_fps = 1;
	int weight_ms = 5;

};

bool read_demo_cfgs(std::string cfgs_path, demo_param& params);

int main(int argc, char* argv[])
{
	// 读入配置文件
	std::string cfg_path = "../data/cfgs.xml";
	demo_param params;
	read_demo_cfgs(cfg_path, params);


	// 输入参数
	int mode = params.test_mode;
	std::string test_img_path = params.test_img_path;
	std::string test_video_path = params.test_video_path;
	int camera_id = params.camera_id;

	// 计时参数
	std::chrono::steady_clock::time_point tic;
	std::chrono::steady_clock::time_point tac;
	double time_used = 0.0;
	double time_used_total = 0.0;


	// 调节参数
	int output_fps = params.output_fps;
	int weight_ms = params.weight_ms;
	int iteration = params.iteration;
	int gpu_warm_up = params.gpu_warm_up;

	cv::namedWindow("visual", cv::WINDOW_FREERATIO);

	// video writer 
	cv::VideoWriter video_writer;
	std::cout << output_fps;
	video_writer.open(params.save_video_path, 
		cv::VideoWriter::fourcc('M', 'P', '4', '2'), output_fps, cv::Size(params.net_input_width, params.net_input_height),true);


	// ———————————————————————————
	// ———————— 算法初始化 ————————
	// ———————————————————————————
	IN void* handle;
	int ret = Ultranet_Init(&handle, cfg_path);

	// 变量定义
	cv::Mat						src_image;
	cv::Mat						visual_img;
	cv::VideoCapture			video_reader;

	if (mode == 2)
	{
		video_reader.open(test_video_path);
	}

	if (mode ==3)
	{
		video_reader.open(camera_id);

	}


	unsigned int counter = 0;
	while (true)
	{


		// —————————————————————————
		// ———————— 准备输入 ————————
		// —————————————————————————
		counter++;
		if (mode ==1)
		{
			if (counter < (iteration + gpu_warm_up))
			{

				src_image = cv::imread(test_img_path, cv::IMREAD_COLOR);

			}

			else
			{
				return 0;
			}
		}

		else
		{
			video_reader >> src_image;

			if (src_image.empty())
			{
				break;
			}
		}


		tic = std::chrono::steady_clock::now();

		// ——————————————————————————————
		// ———————— 进行多任务检测 ————————
		// ——————————————————————————————
		std::cout << "++++++++++++++++++++ ITER " << counter << " ++++++++++++++++++++" << std::endl;
		OUT cv::Mat visual_img;
		OUT Output_Info process_result;
		ret = Ultranet_Detect(handle, src_image, visual_img, process_result);

		tac = std::chrono::steady_clock::now();
		time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
		std::cout << "Hydranet_Detect Interface Total Time Cost: " << time_used << "ms!" << std::endl;

		if (counter > gpu_warm_up)
		{

			time_used_total += time_used;
			double average_time = (time_used_total / (counter - gpu_warm_up) );
			std::cout << "Hydranet_Detect Interface Average Time Cost: " << average_time << " ms" << std::endl;

		}

		std::cout << std::endl;
		std::cout << std::endl;

		cv::imshow("visual", visual_img);
		char c = cv::waitKey(weight_ms);
		if (c == 27) break;
		std::cout << std::endl;

		video_writer << visual_img;

	}

	// ————————————————————————————
	// ———————— 算法反初始化 ————————
	// ————————————————————————————
	ret = Ultranet_Uinit(handle);
	return 0;

}





bool read_demo_cfgs(std::string cfgs_path, demo_param& params)
{

	//读取xml文件中的参数值
	TiXmlDocument* document = new TiXmlDocument();
	if (!document->LoadFile(cfgs_path.c_str()))
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

				if (sub_element->ValueTStr() == "general")
				{
					TiXmlElement* end_element = sub_element->FirstChildElement();
					params.net_input_width = atoi(end_element->GetText());

					end_element = end_element->NextSiblingElement();
					params.net_input_height = atoi(end_element->GetText());

				}

				sub_element = sub_element->NextSiblingElement();

			}



		}

		if (next_element->ValueTStr() == "demo")
		{

			// test_mode
			TiXmlElement* end_element = next_element->FirstChildElement();
			params.test_mode = atoi(end_element->GetText());

			// test_model_path
			end_element = end_element->NextSiblingElement();
			params.test_model_path = end_element->GetText();

			// test_img_path
			end_element = end_element->NextSiblingElement();
			params.test_img_path = end_element->GetText();

			// test_video_path
			end_element = end_element->NextSiblingElement();
			params.test_video_path = end_element->GetText();

			// save_video_path
			end_element = end_element->NextSiblingElement();
			params.save_video_path = end_element->GetText();

			// camera id
			end_element = end_element->NextSiblingElement();
			params.camera_id = atoi(end_element->GetText());

			// gpu warm up
			end_element = end_element->NextSiblingElement();
			params.gpu_warm_up = atoi(end_element->GetText());

			// iterations
			end_element = end_element->NextSiblingElement();
			params.iteration = atoi(end_element->GetText());

			// output_fps
			end_element = end_element->NextSiblingElement();
			params.output_fps = atoi(end_element->GetText());

			// weight_ms
			end_element = end_element->NextSiblingElement();
			params.weight_ms = atoi(end_element->GetText());

		}



		next_element = next_element->NextSiblingElement();
	}

	//释放内存
	delete document;
	return true;
}