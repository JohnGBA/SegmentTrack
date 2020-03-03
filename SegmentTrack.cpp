#include "Header.h"

int main(int argv, char** argc)
{
	//opens webcam
	cv::VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return -1;
	}

	cv::Mat frame, result;
	byGradient = true;
	name = "Using Gradient";
	cv::namedWindow(name);
	cv::setMouseCallback(name, mouseCallback);
	trackbar(emptyHandle, name);

	while (vid.read(frame))
	{
		cv::Mat segmentedImage = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		cv::Mat selectedContoursImage = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);	

		if (byGradient == false) {
			segmentedImage = segmentByAdaptThreshold(frame, selectedContoursImage);
		}

		if (byGradient == true) {
			segmentedImage = segmentByCanny(frame, selectedContoursImage);
		}

		cv::imshow(name, segmentedImage);
		cv::imshow("Selected contours", selectedContoursImage);
		cv::imshow("Resulting Image", frame);
		cv::waitKey(1);
	}
	return -1;
}

