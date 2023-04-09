#include "Ultranet.h"
#include <ultranet_model.h>

using ultranet::ultranet_detection::ultranet_model;

ULTRANET_API int Ultranet_Init(IN void **handle, std::string cfg_path)
{

	ultranet_model * detector = new ultranet_model(cfg_path);

	if (detector == NULL)
	{
		std::cout << "输入的Handle指针为空" << std::endl;
		return -1;
	}

	*handle = (void *)detector;

	return 0;
}

ULTRANET_API int Ultranet_Detect(IN void *handle,
							     IN cv::Mat& input_image,
							     OUT cv::Mat& visual_image,
								 OUT Output_Info & output)
{

	ultranet_model * detector = (ultranet_model *)handle;
	detector->detect(input_image, visual_image, output);
	return 0;

}


ULTRANET_API int Ultranet_Uinit(IN void * handle)
{

	ultranet_model * detector = (ultranet_model *)handle;
	detector->~ultranet_model();
	return 0;

}
