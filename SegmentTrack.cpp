#include "Header.h"

int main(int argv, char** argc)
{
	//opens webcam
	cv::VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return -1;
	}

	cv::Mat frame;
	byGradient = true;
	name = "Using Gradient";
	cv::namedWindow(name);
	cv::setMouseCallback(name, mouseCallback);
	trackbar(empty_handle, name);

	while (vid.read(frame))
	{
		cv::Mat segmentedImage = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		cv::Mat selected_contours_image = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		
		cv::Mat preProcImage = preProcess(frame);

		if (byGradient == false) {
			segmentedImage = segmentationByAdaptThreshold(preProcImage, selected_contours_image, segmentedImage);
		}

		if (byGradient == true) {
			segmentedImage = segmentByCanny(preProcImage, selected_contours_image, segmentedImage);
		}

		cv::imshow(name, segmentedImage);
		cv::imshow("Selected contours", selected_contours_image);
		cv::imshow("Result", frame);
		cv::waitKey(1);
	}
	return -1;
}

