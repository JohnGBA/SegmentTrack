#include "Header.h"

images im;
parameters_t par;
cv::Point Idmax, Idmin;
int low_t, high_t;
double m = 2;  // scale factor in Sobel function

cv::Mat segmentationByAdaptThreshold(cv::Mat CLAHE_image, cv::Mat selected_contours_image, cv::Mat final_image)
{
	cv::Mat adapt_thresh_image;
	cv::adaptiveThreshold(CLAHE_image, adapt_thresh_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 25, 8);
	filterSpeckles(adapt_thresh_image, 0, 1600, 0);
	findContours(adapt_thresh_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	eraseContours(contours, 100);

	std::vector<cv::Point> centroids(contours.size());

	findCentroids_by_Contour(contours, centroids);
	drawContours(final_image, contours, -1, cv::Scalar(255, 255, 255));
	drawCentroids(final_image, centroids);

	if (pause == true) {
		pauseProgram(final_image, contours, centroids, name);
	}

	std::vector <cv::Point> tracked_centroid;
	std::vector <std::vector < cv::Point > > tracked_contour;
	validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, 3, tracked_centroid, tracked_contour);

	if (byRadius == false)
		drawResults(final_image, selected_contours_image, tracked_contour, tracked_centroid);
	else {
		selectContours(contours, centroids, Landmark, RADIUS, NULL);
		drawResults(final_image, selected_contours_image, contours, centroids);
	}
	return final_image;
}

cv::Mat segmentByCanny(cv::Mat CLAHE_image, cv::Mat selected_contours_image, cv::Mat final_image)
{

	canny_with_otsu(CLAHE_image);
	findContours(im.imageCanny, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	eraseContours(contours, 100);
	std::vector<cv::Point> centroids(contours.size());

	findCentroids_by_Contour(contours, centroids);
	drawContours(final_image, contours, -1, cv::Scalar(255, 255, 255));
	drawCentroids(final_image, centroids);
	if (pause == true) {
		cannyPause(final_image, contours, centroids, name);
	}

	std::vector <cv::Point> tracked_centroid;
	std::vector <std::vector < cv::Point > > tracked_contour;
	validationOfCentroid(contours, centroids, centroids.size(), 30, 0.2, 0.25, 3, tracked_centroid, tracked_contour);

	if (byRadius == false)
		drawResults(final_image, selected_contours_image, tracked_contour, tracked_centroid);
	else {
		selectContours(contours, centroids, Landmark, RADIUS, NULL);
		drawResults(final_image, selected_contours_image, contours, centroids);
	}
	return final_image;
}

void trackbar(void(*functocall)(int, void*), std::string image_window)
{
	im.window = image_window;
	cv::namedWindow(image_window);
	cv::setMouseCallback(image_window, mouse_callback, NULL);
	cv::createTrackbar("Low slider", image_window, &par.L_otsu_slider, par.otsu_lim, functocall);
	cv::createTrackbar("High slider", image_window, &par.H_otsu_slider, par.otsu_lim, functocall);
}

void applySobel(const cv::Mat& image, double scale_fac)
{
	double scale = 1;
	int delta = 0;
	cv::Mat grad_x, grad_y;

	Sobel(image, im.grad_x, CV_16SC1, 1, 0, 5, scale, delta, cv::BORDER_DEFAULT);
	Sobel(image, im.grad_y, CV_16SC1, 0, 1, 5, scale, delta, cv::BORDER_DEFAULT);  // tipo 16S para usar no canny algo
	Sobel(image, grad_x, CV_32F, 1, 0, 5, scale, delta, cv::BORDER_DEFAULT);
	Sobel(image, grad_y, CV_32F, 0, 1, 5, scale, delta, cv::BORDER_DEFAULT);   // tipo 32 F para calcular magnitude do gradiente para dar imshow no gradiente.
	if (scale_fac != 1) {
		im.grad_x.convertTo(im.grad_x, -1, scale_fac);  //// optional and converTo keeps the number of channels of the image, untouched. If not , use cvtColor.
		im.grad_y.convertTo(im.grad_y, -1, scale_fac);  //// optional
	}

	cv::magnitude(grad_x, grad_y, im.grad);
	cv::normalize(im.grad, im.grad_scaled, 0, 255, cv::NORM_MINMAX, CV_8UC1);
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

void empty_handle(int, void*) {}

void canny_with_otsu(cv::Mat& image) {
	static int count = 1;
	static int OTSU;
	applySobel(image, m); // Gera im.grad_x/y para ser usado no Canny (16S) e tambem grad_x/y (32F) que gera im.grad ( valor >> 255 )
						  // que depois eh normalizado em im.grad_scaled e ser vizualizado 255 e usado para achar parametro OTSU.
	otsu(im.grad_scaled, OTSU);
	cv::minMaxLoc(im.grad, &im.min, &im.max, &Idmin, &Idmax);

	double fac = m * im.max / 255;
	double OTSU_adapt = fac * OTSU;
	static int low_t;
	static int high_t;
	low_t = OTSU_adapt * par.L_otsu_slider / 100;
	high_t = OTSU_adapt * par.H_otsu_slider / 100; // we divide by 100 so that 0 to 500 means 0 to 5 passing through values in between
	cv::Canny(im.grad_x, im.grad_y, im.imageCanny, low_t, high_t, true);
}

void ballsContour(cv::Mat& image, std::vector< std::vector<cv::Point> >& c, cv::Scalar color)   // fills a contours with equally spread circles
{
	size_t capacity1 = c.size();

	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = c[i].size();

		for (int j = 0; j < capacity2; j = j + 10) {
			circle(image, c[i][j], 2, color, 1);
		}
	}
}

void eraseContours(std::vector< std::vector<cv::Point> >& c, int limit)
{
	size_t capacity1 = c.size();
	int count = 0;
	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = c[i].size();

		for (int j = 0; j < capacity2; j += 10) {
			if (capacity2 > (size_t)limit)
				;
			else {
				c.erase(c.begin() + i);
				i -= 1;
				break;
			}
		}

		count++;
		if (count == capacity1)
			break;
	}
}

void findCentroids_by_Contour(std::vector< std::vector<cv::Point> >& c, std::vector<cv::Point>& centroids)
{
	size_t capacity1 = c.size();

	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = c[i].size();

		for (int j = 0; j < capacity2; j++) {
			centroids[i].x += c[i][j].x;
			centroids[i].y += c[i][j].y;
		}

		centroids[i].x = (centroids[i].x) / capacity2;
		centroids[i].y = (centroids[i].y) / capacity2;
	}

	std::vector<cv::Point> temp = centroids;

	for (int i = 1; i < capacity1; i++) {
		temp[i].x += temp[i - 1].x;
		temp[i].y += temp[i - 1].y;
	}
}

void drawCentroids(cv::Mat& image, std::vector<cv::Point>& v)
{
	size_t capacity1 = v.size();

	for (int i = 0; i < capacity1; i++) {
		circle(image, v[i], 4, cv::Scalar(0, 0, 255), 2);
	}
}

void selectContours(std::vector< std::vector<cv::Point> >& c, std::vector<cv::Point>& centroids, cv::Point& landmark, int max_distance, int Nb_centroids)
{
	std::vector< double > D(centroids.size());
	std::vector< std::vector<cv::Point> > temp;
	std::vector< cv::Point > temp_cent;

	size_t capacity = centroids.size();

	for (int i = 0; i < capacity; i++) {
		D[i] = cv::norm(centroids[i] - landmark);
	}

	int IDmin = 0;
	int IDmax = std::distance(D.begin(), std::max_element(D.begin(), D.end()));

	if (max_distance != NULL) {
		for (int i = 0; i < capacity; i++) {
			if (D[i] < max_distance) {
				temp.push_back(c[i]);
				temp_cent.push_back(centroids[i]);
			}
		}
	}

	if (Nb_centroids != NULL) {
		for (int i = 0; i < Nb_centroids; i++) {
			IDmin = std::distance(D.begin(), std::min_element(D.begin(), D.end()));
			temp.push_back(c[IDmin]);
			temp_cent.push_back(centroids[IDmin]);
			D[IDmin] = D[IDmax];
		}
	}

	centroids = temp_cent;
	c = temp;
}

void mouse_callback(int  event, int  x, int  y, int  flag, void* param)
{
	if (flag == cv::EVENT_FLAG_LBUTTON)
	{
		Landmark = cv::Point(x, y);
		trigger = true;
	}
	if (flag == cv::EVENT_FLAG_CTRLKEY)
	{
		RADIUS -= 2;
	}
	if (flag == cv::EVENT_FLAG_SHIFTKEY)
	{
		RADIUS += 2;
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
		byCanny = !byCanny;

		cv::destroyWindow(name);

		if (byCanny == false) {
			name = "Using Adaptive Threshold";
			cv::namedWindow(name);
			cv::setMouseCallback(name, mouse_callback);
		}
		else {
			name = "Using Canny";
			cv::namedWindow(name);
			cv::setMouseCallback(name, mouse_callback);
			trackbar(empty_handle, name);
		}

		cv::waitKey(30); // give time to load the window
	}
}

void pauseProgram(cv::Mat& MASK, std::vector< std::vector<cv::Point> > c, std::vector<cv::Point> centroids, std::string name) {
	cv::Mat vazia = cv::Mat::zeros(MASK.rows, MASK.cols, CV_8UC3);
	cv::Mat vazia2 = vazia.clone();
	cv::Mat blah = MASK.clone();
	std::vector< std::vector<cv::Point> > c_temp;
	std::vector<cv::Point> centroids_temp(c.size());
	c_temp = c;
	centroids_temp = centroids;

	cv::imshow(name, MASK);

	while (cv::waitKey() != 27) {
		if (byRadius == false)
			selectContours(c, centroids, Landmark, NULL, 1);

		else
			selectContours(c, centroids, Landmark, RADIUS, NULL);

		findCentroids_by_Contour(c, centroids); // modifies centroids and centroid.
		drawContours(vazia, c, -1, cv::Scalar(255, 255, 255));
		drawCentroids(vazia, centroids);

		if (byRadius == true)
			circle(MASK, Landmark, RADIUS, cv::Scalar(0, 255, 0), 1);

		circle(MASK, Landmark, 5, cv::Scalar(255, 0, 255), 2);
		cv::imshow(name, MASK);
		cv::imshow("Selected TrackPoint", vazia);

		c = c_temp;
		centroids = centroids_temp;
		vazia = vazia2.clone();
		MASK = blah.clone();
	}
	pause = false;
	cv::destroyWindow("Selected TrackPoint");
}

void cannyPause(cv::Mat& MASK, std::vector< std::vector<cv::Point> > c, std::vector<cv::Point> centroids, std::string name) {
	cv::Mat vazia = cv::Mat::zeros(MASK.rows, MASK.cols, CV_8UC3);
	cv::Mat vazia2 = vazia.clone();
	cv::Mat blah = MASK.clone();
	std::vector< std::vector<cv::Point> > c_temp;
	std::vector<cv::Point> centroids_temp(c.size());
	c_temp = c;
	centroids_temp = centroids;
	cv::imshow(name, MASK);

	while (cv::waitKey() != 27) {
		canny_with_otsu(im.frame);
		findContours(im.imageCanny, c, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		eraseContours(c, 100);
		std::vector<cv::Point> centroids(c.size());

		if (byRadius == false)
			selectContours(c, centroids, Landmark, NULL, 1);
		else
			selectContours(c, centroids, Landmark, RADIUS, NULL);

		findCentroids_by_Contour(c, centroids);
		drawContours(MASK, c, -1, cv::Scalar(255, 255, 255));
		drawCentroids(MASK, centroids);
		selectContours(c, centroids, Landmark, RADIUS, NULL);
		findCentroids_by_Contour(c, centroids);
		drawContours(vazia, c, -1, cv::Scalar(255, 255, 255));
		drawCentroids(vazia, centroids);
		circle(MASK, Landmark, RADIUS, cv::Scalar(0, 255, 0), 1);
		circle(MASK, Landmark, 5, cv::Scalar(255, 0, 255), 2);
		cv::imshow("temp", vazia);
		cv::imshow(name, MASK);
		c = c_temp;
		centroids = centroids_temp;
		vazia = vazia2.clone();
		MASK = cv::Mat::zeros(MASK.rows, MASK.cols, CV_8UC3);
	}
	pause = false;
	cv::destroyWindow("temp");
}

void newPoint(std::vector <cv::Point>& p, cv::Point& point) {
	p.pop_back();
	p.insert(p.begin(), point);
}

void newSize(std::vector <int>& sizes, int size) {
	sizes.pop_back();
	sizes.insert(sizes.begin(), size);
}

void newDIST(std::vector <double>& DIST, double& distance) {
	DIST.pop_back();
	DIST.insert(DIST.begin(), distance);
}

void get_Points(std::vector< std::vector<cv::Point> >& c, std::vector< std::vector<cv::Point> >& p, int Nb_points)
{
	size_t capacity1 = c.size();
	std::vector < cv::Point > l;

	for (int i = 0; i < capacity1; i++) {
		size_t capacity2 = c[i].size();
		int step = cvRound(((int)capacity2 / Nb_points));

		for (int j = 0; j < capacity2; j = j + step) {
			l.push_back(c[i][j]);
		}
		p.push_back(l);
		l.clear();
	}
}

void calculateDistances(std::vector< std::vector<cv::Point> >& points, std::vector<cv::Point>& centroids, std::vector <double>& distances)
{
	size_t capacity = centroids.size();
	double sum = 0;

	for (int i = 0; i < capacity; i++) {
		size_t capacity2 = points[i].size();

		for (int j = 0; j < capacity2; j++) {
			sum += cv::norm(points[i][j] - centroids[i]);
		}

		sum = sum / capacity2;
		distances.push_back(sum);
		sum = 0;
	}
}

void drawResults(cv::Mat& image, cv::Mat& selectedContours, std::vector < std::vector<cv::Point> > c, std::vector< cv::Point > centroids) {
	if (byRadius == true)
		circle(image, Landmark, RADIUS, cv::Scalar(0, 255, 0), 1);

	circle(image, Landmark, 5, cv::Scalar(255, 0, 255), 2);
	circle(image, P[0], 6, cv::Scalar(255, 0, 0), 2);
	drawContours(selectedContours, c, -1, cv::Scalar(255, 255, 255));
	drawCentroids(selectedContours, centroids);
	cv::cvtColor(im.frame, im.frame, CV_GRAY2BGR);
	ballsContour(im.frame, c, cv::Scalar(0, 0, 255));
}

void validationOfCentroid(std::vector< std::vector<cv::Point> >& c, std::vector <cv::Point>& centroids, int Nb_cent, int param_px, double param_size, double param_distance, int param_crit, std::vector <cv::Point>& TrackedCentroid, std::vector <std::vector  < cv::Point >>& trackedContour) {
	cv::Point distance;
	double relative_size = 0;
	double relative_change = 0;
	int criteria = 0;
	std::vector < double > distances;
	double distanceC = 0;
	int sizeC = 0;
	std::vector <int> sizesC;
	int ID_cent = 0;
	std::vector < std::vector<cv::Point> > points;

	selectContours(c, centroids, Landmark, NULL, Nb_cent); // Modifies c, centroids.
	get_Points(c, points, 10); // points[i] are in same order as c[i] and size[i] etc..
	calculateDistances(points, centroids, distances);

	for (int i = 0; i < Nb_cent; i++) {
		TrackPoint = centroids[i];
		distanceC = distances[i];
		sizeC = (int)c[i].size();
		sizesC.push_back(sizeC);

		if (trigger)  // trigger  == true when we change TrackPoint
		{
			if (reset == true) {
				P[0] = TrackPoint;
				sizes[0] = sizeC;
				DIST[0] = distanceC;
				reset = false;
			}
			else
				trigger = false;
		}

		if ((norm(TrackPoint - P[0]) < param_px)) {
			criteria += 1;
		}

		relative_size = abs((double)(sizeC - sizes[0]) / sizes[0]);

		if ((relative_size < param_size)) {
			criteria += 1;
		}

		relative_change = abs((double)(distanceC - DIST[0]) / DIST[0]);

		if ((relative_change < param_distance)) {
			criteria += 1;
		}

		if (criteria >= param_crit) {
			ID_cent = i;
			newSize(sizes, sizeC);
			newDIST(distances, distances[i]);

			distance = TrackPoint - P[0];
			newPoint(P, TrackPoint);
			Landmark += distance;
			break;
		}
	}
	trackedContour.push_back(c[ID_cent]);
	TrackedCentroid.push_back(centroids[ID_cent]);
}