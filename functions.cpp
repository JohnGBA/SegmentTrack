#include "Header.h"

cv::Point landmark;
std::string name;
int radius = 160;
bool pause = true;
bool trigger = true;
bool reset = true;
bool byRadius = false;
bool byGradient = true;
int lowThresholdSlider = 200;
int highThresholdSlider = 100;
int sliderLimit = 600;

cv::Mat segmentByAdaptThreshold(cv::Mat& frame, cv::Mat selectedContoursImage)
{
	cv::Mat preProcessedImage = preProcess(frame);
	static int sizes = 0;
	static double dist = 0;
	static cv::Point pt = cv::Point(0, 0);
	cv::Mat segmentedImage = cv::Mat::zeros(preProcessedImage.rows, preProcessedImage.cols, CV_8UC3);
	std::vector < std::vector<cv::Point> > contours;
	cv::Mat adaptThresholdImage;
	cv::adaptiveThreshold(preProcessedImage, adaptThresholdImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 25, 8);
	filterSpeckles(adaptThresholdImage, 0, 1600, 0);
	findContours(adaptThresholdImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	eraseContours(contours, 100);

	std::vector<cv::Point> centroids(contours.size());

	findCentroids(contours, centroids);
	drawContours(segmentedImage, contours, -1, cv::Scalar(255, 255, 255));
	drawCentroids(segmentedImage, centroids);

	if (pause == true) {
		pauseProgram(segmentedImage, contours, centroids, name);
	}

	std::vector <cv::Point> trackedCentroid;
	std::vector <std::vector < cv::Point > > trackedContour;
	validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, trackedCentroid, trackedContour, pt, sizes, dist);

	if (byRadius == false)
		drawResults(frame, segmentedImage, selectedContoursImage, trackedContour, trackedCentroid, pt);
	else {
		selectContours(contours, centroids, landmark, radius, NULL);
		drawResults(frame, segmentedImage, selectedContoursImage, contours, centroids, pt);
	}
	return segmentedImage;
}

cv::Mat segmentByCanny(cv::Mat& frame, cv::Mat selectedContoursImage)
{
	cv::Mat preProcessedImage = preProcess(frame);
	static int sizes = 0;
	static double dist = 0;
	static cv::Point pt = cv::Point(0, 0);
	cv::Mat segmentedImage = cv::Mat::zeros(preProcessedImage.rows, preProcessedImage.cols, CV_8UC3);
	std::vector < std::vector<cv::Point> > contours;
	cv::Mat cannyImage = cannyUsingOtsu(preProcessedImage);
	findContours(cannyImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	eraseContours(contours, 100);
	std::vector<cv::Point> centroids(contours.size());

	findCentroids(contours, centroids);
	drawContours(segmentedImage, contours, -1, cv::Scalar(255, 255, 255));
	drawCentroids(segmentedImage, centroids);
	if (pause == true) {
		cannyPause(preProcessedImage, segmentedImage, contours, centroids, name);
	}

	std::vector <cv::Point> trackedCentroid;
	std::vector <std::vector < cv::Point > > trackedContour;
	validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, trackedCentroid, trackedContour, pt, sizes, dist);

	if (byRadius == false)
		drawResults(frame, segmentedImage, selectedContoursImage, trackedContour, trackedCentroid, pt);		
	else {
		selectContours(contours, centroids, landmark, radius, NULL);
		drawResults(frame, segmentedImage, selectedContoursImage, contours, centroids, pt);
	}
	return segmentedImage;
}

cv::Mat preProcess(cv::Mat frame) 
{
	cv::cvtColor(frame, frame, CV_BGR2GRAY);
	medianBlur(frame, frame, 3);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat preProcessedImage;
	clahe->apply(frame, preProcessedImage);
	return preProcessedImage;
}

void trackbar(void(*functocall)(int, void*), std::string windowName)
{
	cv::namedWindow(windowName);
	cv::setMouseCallback(windowName, mouseCallback, NULL);
	cv::createTrackbar("Low slider", windowName, &highThresholdSlider, sliderLimit, functocall);
	cv::createTrackbar("High slider", windowName, &lowThresholdSlider, sliderLimit, functocall);
}

std::array <cv::Mat,4> findGradients(const cv::Mat& image)
{
	//Gradient on x and y and magnitude 
	std::array <cv::Mat, 4> gradient;
	double scale = 1;
	int delta = 0;
	cv::Mat gradX, gradY;
	cv::Mat gradXFloat , gradYFloat;
	cv::Mat grad;
	cv::Mat gradScaled;

	Sobel(image, gradX, CV_16SC1, 1, 0, 5, scale, delta, cv::BORDER_DEFAULT);
	Sobel(image, gradY, CV_16SC1, 0, 1, 5, scale, delta, cv::BORDER_DEFAULT);  // tipo 16S para usar no canny algo
	Sobel(image, gradXFloat, CV_32F, 1, 0, 5, scale, delta, cv::BORDER_DEFAULT);
	Sobel(image, gradYFloat, CV_32F, 0, 1, 5, scale, delta, cv::BORDER_DEFAULT);   // tipo 32 F para calcular magnitude do gradiente para dar imshow no gradiente.
	
	gradX.convertTo(gradX, -1, 2);  //// optional and converTo keeps the number of channels of the image. Othewise use cvtColor.
	gradY.convertTo(gradY, -1, 2);  //// optional
	
	cv::magnitude(gradXFloat, gradYFloat, grad);
	cv::normalize(grad, gradScaled, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	gradient[0] = gradX; 
	gradient[1] = gradY;
	gradient[2] = grad;
	gradient[3] = gradScaled;
	return gradient;
}

void otsu(const cv::Mat& image, int& threshold1) {
	cv::Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = range;
	bool uniform = true;
	bool accum = false;
	int t;
	float sum = 0.0, wb = 0.0, wf = 0.0, mb, mf, sumb = 0.0, max = 0.0, between;
	calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accum);

	float* p = (float*)hist.ptr(0);
	for (int i = 0; i < 256; ++i)
	{
		sum += i * p[i];
	}

	for (t = 0; t < 256; t++) {
		wb += p[t];
		if (wb == 0)
			continue;
		wf = image.rows * image.cols - wb;
		if (wf == 0)
			break;
		sumb += (float)(t * p[t]);
		mb = sumb / wb;
		mf = (sum - sumb) / wf;
		between = (float)wb * (float)wf * (mb - mf) * (mb - mf);
		if (between > max) {
			threshold1 = t;
			max = between;
		}
	}
}

void emptyHandle(int, void*) {}

cv::Mat cannyUsingOtsu(cv::Mat& image) {
	double min, max;
	cv::Point Idmax, Idmin;
	static int OTSU;
	cv::Mat cannyImage;
	std::array <cv::Mat, 4> gradient = findGradients(image); // Gera im.gradX/y para ser usado no Canny (16S) e tambem gradX/y (32F) que gera im.grad ( valor >> 255 )
						  // que depois eh normalizado em im.gradScaled e ser vizualizado 255 e usado para achar parametro OTSU.

	otsu(gradient[3], OTSU);
	cv::minMaxLoc(gradient[2], &min, &max, &Idmin, &Idmax);

	double multiplyingFactor = 2 * max / 255; // the multiplying number has to be the same as the 3rd argument of contertTo inside findGradients
	double adjustedOtsu = multiplyingFactor * OTSU;
	static int lowThreshold;
	static int highThreshold;
	lowThreshold = adjustedOtsu * highThresholdSlider / 100;
	highThreshold = adjustedOtsu * lowThresholdSlider / 100; // we divide by 100 so that 0 to 500 means 0 to 5 passing through values in between
	cv::Canny(gradient[0], gradient[1], cannyImage, lowThreshold, highThreshold, true);
	return cannyImage;
}

void drawCirclesInSelectedContours(cv::Mat& image, std::vector< std::vector<cv::Point> >& contours,
	cv::Scalar color)   // fills a contours with equally spread circles
{
	size_t capacity1 = contours.size();

	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = contours[i].size();

		for (int j = 0; j < capacity2; j = j + 10) {
			circle(image, contours[i][j], 2, color, 1);
		}
	}
}

void eraseContours(std::vector< std::vector<cv::Point> >& contours, int limit)
{
	size_t capacity1 = contours.size();
	int count = 0;
	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = contours[i].size();

		for (int j = 0; j < capacity2; j += 10) {
			if (capacity2 > (size_t)limit)
				;
			else {
				contours.erase(contours.begin() + i);
				i -= 1;
				break;
			}
		}

		count++;
		if (count == capacity1)
			break;
	}
}

void findCentroids(std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Point>& centroids)
{
	size_t capacity1 = contours.size();
	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = contours[i].size();

		for (int j = 0; j < capacity2; j++) {
			centroids[i].x += contours[i][j].x;
			centroids[i].y += contours[i][j].y;
		}
		centroids[i].x = (centroids[i].x) / capacity2;
		centroids[i].y = (centroids[i].y) / capacity2;
	}
}

void drawCentroids(cv::Mat& image, std::vector<cv::Point>& centroids)
{
	size_t capacity1 = centroids.size();
	for (int i = 0; i < capacity1; i++) {
		circle(image, centroids[i], 4, cv::Scalar(0, 0, 255), 2);
	}
}

void selectContours(std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Point>& centroids, 
	cv::Point& landmark, int maxDistance, int NbCentroids)
{
	std::vector< double > dist(centroids.size());
	std::vector< std::vector<cv::Point> > contoursTemp;
	std::vector< cv::Point > centroidsTemp;

	size_t centroidsSize = centroids.size();

	for (int i = 0; i < centroidsSize; i++) {
		dist[i] = cv::norm(centroids[i] - landmark);
	}

	int IDmin = 0;
	int IDmax = std::distance(dist.begin(), std::max_element(dist.begin(), dist.end()));

	if (maxDistance != NULL) {
		for (int i = 0; i < centroidsSize; i++) {
			if (dist[i] < maxDistance) {
				contoursTemp.push_back(contours[i]);
				centroidsTemp.push_back(centroids[i]);
			}
		}
	}

	if (NbCentroids != NULL) {
		for (int i = 0; i < NbCentroids; i++) {
			IDmin = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
			contoursTemp.push_back(contours[IDmin]);
			centroidsTemp.push_back(centroids[IDmin]);
			dist[IDmin] = dist[IDmax];
		}
	}

	centroids = centroidsTemp;
	contours = contoursTemp;
}

void mouseCallback(int  event, int  x, int  y, int  flag, void* param)
{
	if (flag == cv::EVENT_FLAG_LBUTTON)
	{
		landmark = cv::Point(x, y);
		trigger = true;
	}
	if (flag == cv::EVENT_FLAG_CTRLKEY)
	{
		radius -= 2;
	}
	if (flag == cv::EVENT_FLAG_SHIFTKEY)
	{
		radius += 2;
	}
	if (flag == cv::EVENT_FLAG_ALTKEY)
	{
		pause = true;
	}
	if (flag == cv::EVENT_FLAG_RBUTTON)
	{
		reset = true;
	}
	if (flag == (cv::EVENT_FLAG_CTRLKEY + cv::EVENT_FLAG_SHIFTKEY))
	{
		byRadius = !byRadius;
		cv::waitKey(10);
	}

	if (flag == (cv::EVENT_FLAG_CTRLKEY + cv::EVENT_FLAG_ALTKEY))
	{
		byGradient = !byGradient;
		cv::destroyWindow(name);

		if (byGradient == false) {
			name = "Using Adaptive Threshold";
			cv::namedWindow(name);
			cv::setMouseCallback(name, mouseCallback);
		}
		else {
			name = "Using Gradient";
			cv::namedWindow(name);
			cv::setMouseCallback(name, mouseCallback);
			trackbar(emptyHandle, name);
		}

		cv::waitKey(30); // give time to load the window
	}
}

void pauseProgram(cv::Mat& segmentedImage, std::vector< std::vector<cv::Point> > contours,
	std::vector<cv::Point> centroids, std::string name) 
{
	cv::Mat selectedContoursImage = cv::Mat::zeros(segmentedImage.rows, segmentedImage.cols, CV_8UC3);
	cv::Mat segmentedImageCopy = segmentedImage.clone();
	std::vector< std::vector<cv::Point> > contoursTemp = contours;
	std::vector<cv::Point> centroidsTemp(contours.size());
	centroidsTemp = centroids;

	cv::imshow(name, segmentedImage);

	while (cv::waitKey() != 27) {
		if (byRadius == false)
			selectContours(contours, centroids, landmark, NULL, 1);

		else {
			//draws the radius green circle and select the contours inside it
			selectContours(contours, centroids, landmark, radius, NULL);
			circle(segmentedImage, landmark, radius, cv::Scalar(0, 255, 0), 1);
		}

		findCentroids(contours, centroids); 
		drawContours(selectedContoursImage, contours, -1, cv::Scalar(255, 255, 255));
		drawCentroids(selectedContoursImage, centroids);

		//draws the landmark point
		circle(segmentedImage, landmark, 5, cv::Scalar(255, 0, 255), 2);
		cv::imshow(name, segmentedImage);
		cv::imshow("Selected contour(s)", selectedContoursImage);

		//resets data
		contours = contoursTemp;
		centroids = centroidsTemp;
		selectedContoursImage = cv::Mat::zeros(segmentedImage.rows, segmentedImage.cols, CV_8UC3);
		segmentedImage = segmentedImageCopy.clone();
	}
	pause = false;
	cv::destroyWindow("Selected contour(s)");
}

void cannyPause(cv::Mat preProcImage, cv::Mat& segmentedImage, std::vector< std::vector<cv::Point> > contours, 
	std::vector<cv::Point> centroids, std::string name)
{
	cv::Mat selectedContoursImage = cv::Mat::zeros(segmentedImage.rows, segmentedImage.cols, CV_8UC3);
	cv::Mat segmentedImageCopy = segmentedImage.clone();
	std::vector< std::vector<cv::Point> > contoursTemp = contours;
	std::vector<cv::Point> centroidsTemp(contours.size());
	centroidsTemp = centroids;
	cv::Mat cannyImage;

	cv::imshow(name, segmentedImage);

	while (cv::waitKey() != 27) {
		cannyImage = cannyUsingOtsu(preProcImage);
		findContours(cannyImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		eraseContours(contours, 100);
		std::vector<cv::Point> centroids(contours.size());

		if (byRadius == false)
			selectContours(contours, centroids, landmark, NULL, 1);
		else
			selectContours(contours, centroids, landmark, radius, NULL);

		selectContours(contours, centroids, landmark, radius, NULL);
		findCentroids(contours, centroids);
		drawContours(selectedContoursImage, contours, -1, cv::Scalar(255, 255, 255));
		drawCentroids(selectedContoursImage, centroids);
		circle(segmentedImage, landmark, radius, cv::Scalar(0, 255, 0), 1);
		circle(segmentedImage, landmark, 5, cv::Scalar(255, 0, 255), 2);

		cv::imshow("Selected contour(s)", selectedContoursImage);
		cv::imshow(name, segmentedImage);

		//resets data
		contours = contoursTemp;
		centroids = centroidsTemp;
		selectedContoursImage = cv::Mat::zeros(segmentedImage.rows, segmentedImage.cols, CV_8UC3);
		segmentedImage = segmentedImageCopy.clone();
	}
	pause = false;
	cv::destroyWindow("Selected contour(s)");
}

void getPoints(std::vector< std::vector<cv::Point> >& contours, 
	std::vector< std::vector<cv::Point> >& allContoursPoints, int nbPointsToGet)
{
	size_t nbContours = contours.size();
	std::vector < cv::Point > points;

	for (int i = 0; i < nbContours; i++) {
		size_t sizeContour = contours[i].size();
		int step = cvRound(((int)sizeContour / nbPointsToGet));

		for (int j = 0; j < sizeContour; j = j + step) {
			points.push_back(contours[i][j]);
		}
		allContoursPoints.push_back(points);
		points.clear();
	}
}

void calculateMeanDistances(std::vector< std::vector<cv::Point> >& points, 
	std::vector<cv::Point>& centroids, std::vector <double>& meanDistances)
{
	size_t nbCentroids = centroids.size();
	double sum = 0;

	for (int i = 0; i < nbCentroids; i++) {
		size_t nbPoints = points[i].size();

		for (int j = 0; j < nbPoints; j++) {
			sum += cv::norm(points[i][j] - centroids[i]);
		}

		sum = sum / nbPoints;
		meanDistances.push_back(sum);
		sum = 0;
	}
}

void drawResults(cv::Mat& frame, cv::Mat& segmentedImage, cv::Mat& selectedContours,
	std::vector < std::vector<cv::Point> > contours, std::vector< cv::Point > centroids, cv::Point pt)
{
	if (byRadius == true)
		circle(segmentedImage, landmark, radius, cv::Scalar(0, 255, 0), 1);

	circle(segmentedImage, landmark, 5, cv::Scalar(255, 0, 255), 2);
	circle(segmentedImage, pt, 6, cv::Scalar(255, 0, 0), 2);
	drawContours(selectedContours, contours, -1, cv::Scalar(255, 255, 255));
	drawCentroids(selectedContours, centroids);
	if (frame.channels() < 3)
		cv::cvtColor(frame, frame, CV_GRAY2BGR);
	drawCirclesInSelectedContours(frame, contours, cv::Scalar(0, 0, 255));
}

void validationOfCentroid(std::vector< std::vector<cv::Point> >& contours, std::vector <cv::Point>& centroids,
	int nbCentroids, int proximityConstraint, double sizeConstraint, double meanDistanceConstraint,
	std::vector <cv::Point>& TrackedCentroid, std::vector <std::vector  < cv::Point >>& trackedContour,
	cv::Point& pt, int& sizes, double& dist)
{
	cv::Point trackPoint;
	cv::Point distance;
	double relativeSize = 0;
	double relativeChange = 0;
	int criteria = 0;
	std::vector < double > meanDistances;
	double distanceC = 0;
	int sizeC = 0;
	std::vector <int> sizesC;
	int IdCentroid = 0;
	std::vector < std::vector<cv::Point> > points;

	selectContours(contours, centroids, landmark, NULL, nbCentroids); 
	getPoints(contours, points, 10); // points[i] are in same order as contours[i] and size[i] etc..
	calculateMeanDistances(points, centroids, meanDistances);

	for (int i = 0; i < nbCentroids; i++) {
		trackPoint = centroids[i];
		distanceC = meanDistances[i];
		sizeC = (int)contours[i].size();
		sizesC.push_back(sizeC);

		if (trigger)  // trigger  == true when we change trackPoint
		{
			if (reset == true) {
				pt = trackPoint;
				sizes = sizeC;
				dist = distanceC;
				reset = false;
			}
			else
				trigger = false;
		}

		if ((norm(trackPoint - pt) < proximityConstraint)) {
			criteria += 1;
		}

		relativeSize = abs((double)(sizeC - sizes) / sizes);

		if ((relativeSize < sizeConstraint)) {
			criteria += 1;
		}

		relativeChange = abs((double)(distanceC - dist) / dist);

		if ((relativeChange < meanDistanceConstraint)) {
			criteria += 1;
		}

		if (criteria >= 3) {
			IdCentroid = i;
			sizes = sizeC;
			dist = distanceC;
			distance = trackPoint - pt;
			pt = trackPoint;
			landmark += distance;
			break;
		}
	}
	trackedContour.push_back(contours[IdCentroid]);
	TrackedCentroid.push_back(centroids[IdCentroid]);
}