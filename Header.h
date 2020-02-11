#pragma once
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include <stdint.h>
#include "opencv2/video/tracking.hpp" 
#include <sstream>
#include <fstream>

extern std::string name;
extern bool byGradient;

std::array <cv::Mat, 4> findGradients(const cv::Mat& image);
void otsu(const cv::Mat& image, int& threshold1);
void empty_handle(int, void*);
void trackbar(void(*functocall)(int, void*), std::string image_window);
cv::Mat canny_with_otsu(cv::Mat& image);
void drawCirclesInSelectedContours(cv::Mat& image, std::vector< std::vector<cv::Point> >& c, cv::Scalar color);
void eraseContours(std::vector< std::vector<cv::Point> >& contours, int limit);
void findCentroids(std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Point>& centroids);
void drawCentroids(cv::Mat& image, std::vector<cv::Point>& v);
void selectContours(std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Point>& centroids, cv::Point& landmark, int max_distance, int Nb_centroids);
void mouseCallback(int  event, int  x, int  y, int  flag, void* param);
void pauseProgram(cv::Mat& MASK, std::vector< std::vector<cv::Point> > c, std::vector<cv::Point> centroids, std::string name);
void cannyPause(cv::Mat preProcImage, cv::Mat& segmentedImage, std::vector< std::vector<cv::Point> > contours, std::vector<cv::Point> centroids, std::string name);
void newPoint(std::vector <cv::Point>& p, cv::Point& pt);
void newSize(std::vector <int>& sizes, int size);
void newDist(std::vector <double>& distances, double& distance);
void calculateMeanDistances(std::vector< std::vector<cv::Point> >& points, std::vector<cv::Point>& centroids, std::vector <double>& distances);
void getPoints(std::vector< std::vector<cv::Point> >& c, std::vector< std::vector<cv::Point> >& p, int Nb_points);
void validationOfCentroid(std::vector< std::vector<cv::Point> >& c, std::vector <cv::Point>& centroids, int Nb_cent, int param_px, double param_size, double param_distance, std::vector <cv::Point>& trackPoint, std::vector <std::vector  < cv::Point >>& trackedContour);
void drawResults(cv::Mat& preProcImage, cv::Mat& image, cv::Mat& selectedContours, std::vector < std::vector<cv::Point> > c, std::vector< cv::Point > centroids);
cv::Mat segmentationByAdaptThreshold(cv::Mat preProcImage, cv::Mat selected_contours_image, cv::Mat segmentedImage);
cv::Mat segmentByCanny(cv::Mat preProcImage, cv::Mat selected_contours_image, cv::Mat segmentedImage);
cv::Mat preProcess(cv::Mat frame);