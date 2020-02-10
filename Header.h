#pragma once
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include <stdint.h>
#include "opencv2/video/tracking.hpp" 
#include <sstream>
#include <fstream>

extern cv::Point Idmax, Idmin;
extern int low_t, high_t;
extern double m;
extern std::string name;
extern int RADIUS;
extern bool pause;
extern bool trigger;
extern bool reset;
extern bool byRadius;
extern bool byCanny;
extern std::vector <int> sizes;
extern std::vector <double> DIST;
extern std::vector < std::vector<cv::Point> > contours;
extern std::vector < cv::Point > P;
extern cv::Point Landmark;
extern cv::Point TrackPoint;
extern images im;
extern parameters_t par;

struct images {
public:
	cv::Mat frame;
	cv::Mat imageCanny;
	cv::Mat grad_y;
	cv::Mat grad_x;
	cv::Mat grad;
	cv::Mat grad_scaled;
	std::string window;
	double min, max;
};
struct parameters_t {
	int low_lim = 255;
	int high_lim = 255;
	int low_slider = 0;
	int high_slider = 100;
	int H_otsu_slider = 200;
	int L_otsu_slider = 100;
	int otsu_lim = 600;
};

void ApplySobel(const cv::Mat& image, double scale_fac = 1);
void otsu(const cv::Mat& image, int& threshold1);
void empty_handle(int, void*);
void otsu_trackbar(void(*functocall)(int, void*), std::string image_window);
void canny_with_otsu(cv::Mat& image);
void ballsContour(cv::Mat& image, std::vector< std::vector<cv::Point> >& c, cv::Scalar color);
void eraseContours(std::vector< std::vector<cv::Point> >& c, int limit);
void findCentroids_by_Contour(std::vector< std::vector<cv::Point> >& c, std::vector<cv::Point>& centroids);
void drawCentroids(cv::Mat& image, std::vector<cv::Point>& v);
void selectContours(std::vector< std::vector<cv::Point> >& c, std::vector<cv::Point>& centroids, cv::Point& landmark, int max_distance, int Nb_centroids);
void mouse_callback(int  event, int  x, int  y, int  flag, void* param);
void pauseProgram(cv::Mat& MASK, std::vector< std::vector<cv::Point> > c, std::vector<cv::Point> centroids, std::string name);
void cannyPause(cv::Mat& MASK, std::vector< std::vector<cv::Point> > c, std::vector<cv::Point> centroids, std::string name);
void newPoint(std::vector <cv::Point>& p, cv::Point& point);
void newSize(std::vector <int>& sizes, int size);
void newDIST(std::vector <double>& distances, double& distance);
void calculateDistances(std::vector< std::vector<cv::Point> >& points, std::vector<cv::Point>& centroids, std::vector <double>& distances);
void get_Points(std::vector< std::vector<cv::Point> >& c, std::vector< std::vector<cv::Point> >& p, int Nb_points);
void validationOfCentroid(std::vector< std::vector<cv::Point> >& c, std::vector <cv::Point>& centroids, int Nb_cent, int param_px, double param_size, double param_distance, int param_crit, std::vector <cv::Point>& TrackPoint, std::vector <std::vector  < cv::Point >>& trackedContour);
void drawResults(cv::Mat& image, cv::Mat& selectedContours, std::vector < std::vector<cv::Point> > c, std::vector< cv::Point > centroids);
