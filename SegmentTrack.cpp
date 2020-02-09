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
bool byCanny = false;

std::vector < cv::Point > P(1, cv::Point(0, 0));
std::vector <int> sizes(1, 0);
std::vector <double> DIST(1, 0);
std::vector < std::vector<cv::Point> > contours;

int main(int argv, char** argc)
{
	cv::VideoCapture vid(0);
	//cv::VideoCapture vid("Test01.avi");
	if (!vid.isOpened())
	{
		return -1;
	}

	if (byCanny == false) {
		name = "Adapt threshold binarized image";
		cv::namedWindow(name);
		cv::setMouseCallback(name, mouse_callback);
	}
	else {
		name = "Canny binarized image";
		cv::namedWindow(name);
		cv::setMouseCallback(name, mouse_callback);
		otsu_trackbar(empty_handle, name);
	}

	while (vid.read(im.frame))
	{
		cv::Mat static_image = cv::Mat::zeros(im.frame.rows, im.frame.cols, CV_8UC3);
		cv::Mat selected_contours_image = cv::Mat::zeros(im.frame.rows, im.frame.cols, CV_8UC3);
		cv::Mat selected_contours_image_backup = cv::Mat::zeros(im.frame.rows, im.frame.cols, CV_8UC3);

		cvtColor(im.frame, im.frame, CV_BGR2GRAY);
		medianBlur(im.frame, median_blurred_image, 5);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(4);

		cv::Mat CLAHE_image;
		clahe->apply(median_blurred_image, CLAHE_image);
		if (byCanny == false) {
			cv::Mat adapt_thresh_image;
			cv::adaptiveThreshold(CLAHE_image, adapt_thresh_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 25, 8);
			filterSpeckles(adapt_thresh_image, 0, 1600, 0);
			findContours(adapt_thresh_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			eraseContours(contours, 100);

			std::vector<cv::Point> centroids(contours.size());

			findCentroids_by_Contour(contours, centroids);
			drawContours(static_image, contours, -1, cv::Scalar(255, 255, 255));
			drawCentroids(static_image, centroids);

			if (pause == true){
				pauseProgram(static_image, contours, centroids, name);
			}
			std::vector <cv::Point> tracked_centroid;
			std::vector <std::vector < cv::Point > > tracked_contour;

			validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, 3, tracked_centroid, tracked_contour);

			if (byRadius == false)
				drawResults(static_image, selected_contours_image, tracked_contour, tracked_centroid);
			else {
				selectContours(contours, centroids, Landmark, RADIUS, NULL);
				drawResults(static_image, selected_contours_image, contours, centroids);
			}
		}

		if (byCanny == true) {
			canny_with_otsu(CLAHE_image);
			findContours(im.imageCanny, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			eraseContours(contours, 100);

			std::vector<cv::Point> centroids(contours.size());

			findCentroids_by_Contour(contours, centroids);
			drawContours(static_image, contours, -1, cv::Scalar(255, 255, 255));
			drawCentroids(static_image, centroids);
			if (pause == true) {
				cannyPause(static_image, contours, centroids, name);
			}

			std::vector <cv::Point> tracked_centroid;
			std::vector <std::vector < cv::Point > > tracked_contour;

			validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, 3, tracked_centroid, tracked_contour);

			if (byRadius == false)
				drawResults(static_image, selected_contours_image, tracked_contour, tracked_centroid);
			else {
				selectContours(contours, centroids, Landmark, RADIUS, NULL);
				drawResults(static_image, selected_contours_image, contours, centroids);
			}
		}
		imshow(name, static_image);
		imshow("Selected contours", selected_contours_image);
		imshow("Result", im.frame);
		cv::waitKey(1);
	}
	return -1;
}

