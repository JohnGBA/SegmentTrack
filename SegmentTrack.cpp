#include "Header.h"

cv::Mat grad_x, grad_y;
cv::Mat median_blurred_image;
cv::Point Landmark;
cv::Point TrackPoint;
std::string name;
int RADIUS = 160;
bool pause = true;
bool trigger = true;
bool reset = true;
bool byRadius = false;
bool byCanny = true;
std::vector < cv::Point > P(1, cv::Point(0, 0));
std::vector <int> sizes(1, 0);
std::vector <double> DIST(1, 0);
std::vector < std::vector<cv::Point> > contours;

int main(int argv, char** argc)
{
	//opens webcam
	cv::VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return -1;
	}

	name = "Using Canny";
	cv::namedWindow(name);
	cv::setMouseCallback(name, mouse_callback);
	trackbar(empty_handle, name);

	while (vid.read(im.frame))
	{
		cv::Mat final_image = cv::Mat::zeros(im.frame.rows, im.frame.cols, CV_8UC3);
		cv::Mat selected_contours_image = cv::Mat::zeros(im.frame.rows, im.frame.cols, CV_8UC3);
		cv::cvtColor(im.frame, im.frame, CV_BGR2GRAY);
		medianBlur(im.frame, median_blurred_image, 5);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(4);

		cv::Mat CLAHE_image;
		clahe->apply(median_blurred_image, CLAHE_image);

		if (byCanny == false) {
			final_image = segmentationByAdaptThreshold(CLAHE_image, selected_contours_image, final_image);
		}

		if (byCanny == true) {
			final_image = segmentByCanny(CLAHE_image, selected_contours_image, final_image);
		}

		cv::imshow(name, final_image);
		cv::imshow("Selected contours", selected_contours_image);
		cv::imshow("Result", im.frame);
		cv::waitKey(1);
	}
	return -1;
}

