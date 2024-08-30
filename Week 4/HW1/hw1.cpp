#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define MAX_ITERATION 500

vector<Point> points;
vector<Point> previousPoints;
bool converged = false;
Mat original_img, imgCopy;

//  draw line between points
void drawPoints(Mat img) {
    for(int i=0; i<points.size(); i++) {
        int j = (i+1)%points.size();
        line(img, points[i], points[j], Scalar(255, 0, 0), 2);
    }
    cout << "draw"<< endl;
    imshow("Active Contour", img);
    waitKey(100);
}


//  active contour
void ACTIVE_CONTOUR(Mat &dx, Mat &dy) {
    //  set search region
    Size win(7, 7); 

    //  numbers of points in search region
    int neighbors = win.height * win.width;
    
    float alpha = 0.4;
    float beta = 0.6;
    float gamma = 0.8;

    //  the center of search region
    int centerx = win.width >> 1;
    int centery = win.height >> 1;

    vector<float> Econt(neighbors);
    vector<float> Ecurv(neighbors);
    vector<float> Eimg(neighbors);
    vector<float> E(neighbors);

    int movecount = 0;
    float averagedistance = 0;
    float tmp;
    int offsetx, offsety;

    //  calculate average distance between points
    for (size_t i = 0; i < points.size(); ++i) {
        Point diff = (i > 0) ? (points[i - 1] - points[i]) : (points.back() - points[i]);
        averagedistance += norm(diff);
    }
    averagedistance /= points.size();

    //  calculate energies
    for (size_t i = 0; i < points.size(); i++) {
        float maxEcont = 0.0f;
        float minEcont = 2.e+38f;
        float maxEcurv = 0.0f;
        float minEcurv = 2.e+38f;
        float maxEimg = 0.0f;
        float minEimg = 2.e+38f;
        
        //  adjust the search range to avoid exceeding the image range
        int left = min(points[i].x, centerx);
        int right = min(original_img.cols - 1 - points[i].x, centerx);
        int upper = min(points[i].y, centery);
        int bottom = min(original_img.rows - 1 - points[i].y, centery);

        //  evaluate energy for each pixel within the search region
        for (int j = -upper; j <= bottom; j++) {
            for (int k = -left; k <= right; k++) {
                int nx = points[i].x + k;
                int ny = points[i].y + j;

                //  calculate E cout
                Point prev_diff = (i > 0) ? (points[i - 1] - Point(nx, ny)) : (points.back() - Point(nx, ny));
                float E_cont = abs(averagedistance - norm(prev_diff));
                Econt[(j + centery) * win.width + k + centerx] = E_cont;
                maxEcont = max(maxEcont, E_cont);
                minEcont = min(minEcont, E_cont);

                //  calculate E curv
                Point prev_point = (i == 0) ? points[points.size() - 1] : points[i - 1];
                Point next_point = (i == points.size() - 1) ? points[0] : points[i + 1];
                Point current_point(nx, ny);
                Point curvature_vector = prev_point - 2 * current_point + next_point;
                float E_curv = norm(curvature_vector);
                Ecurv[(j + centery) * win.width + k + centerx] = E_curv;
                maxEcurv = max(maxEcurv, E_curv);
                minEcurv = min(minEcurv, E_curv);

                //  calculate E img
                float gradient_x = dx.at<float>(ny, nx);
                float gradient_y = dy.at<float>(ny, nx);
                float E_img = pow(gradient_x, 2) + pow(gradient_y, 2);
                Eimg[(j + centery) * win.width + k + centerx] = E_img;
                maxEimg = max(maxEimg, E_img);
                minEimg = min(minEimg, E_img);
            }
        }

        //  normalize E cout
        tmp = maxEcont - minEcont;
        tmp = (tmp == 0) ? 0.0f : (1.0/tmp);
        for (int j = 0; j < neighbors; j++) {
            Econt[j] = (Econt[j] - minEcont) * tmp;
        }

        //  normalize E curv
        tmp = maxEcurv - minEcurv;
        tmp = (tmp == 0) ? 0.0f : (1.0/tmp);
        for (int j = 0; j < neighbors; j++) {
            Ecurv[j] = (Ecurv[j] - minEcurv) * tmp;
        }

        //  normalize E img
        tmp = maxEimg - minEimg;
        tmp = (tmp == 0) ? 0.0f : (1.0/tmp);
        for (int j = 0; j < neighbors; j++) {
            Eimg[j] = (minEimg - Eimg[j]) * tmp;
        }

        //  calculate total energy
        for (int j = 0; j < neighbors; j++) {
            E[j] = alpha * Econt[j] + beta * Ecurv[j] + gamma * Eimg[j];
        }

        //  find best move
        float min_total_energy = 2.e+38f;
        Point best_move = points[i];
        for (int j = -upper; j <= bottom; j++)
			for (int k = -left; k <= right; k++)
					if (E[(j + centery) * win.width + k + centerx] < min_total_energy) {
						min_total_energy = E[(j + centery) * win.width + k + centerx];
						offsetx = k;
						offsety = j;
                    }

        //  update point position 
		if(offsetx || offsety) {
            points[i] += Point(offsetx, offsety);
            movecount++;
        }
    }

    converged = (movecount  == 0) ? true : false;

    //  draw new lines
    imgCopy = original_img.clone();
    drawPoints(imgCopy);
}


//  draw points when mouse click
static void onMouse(int event, int x, int y, int, void* imgptr){
    if (event == EVENT_LBUTTONDOWN) {
        Mat &img = *(Mat*)imgptr;
        Point pt = Point(x, y);
        points.push_back(pt);
        circle(img, pt, 3, Scalar(0, 255, 0), -1);
        imshow("Active Contour", img);
    }
}


int main() {
    //  read the image
    original_img = imread("img.jpg", IMREAD_COLOR);

    imgCopy = original_img.clone();

    if (original_img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    //  convert the image to Grayscale
    Mat gray_img;
    cvtColor(original_img, gray_img, COLOR_BGR2GRAY);
    
    //  denoising with Gaussian blur
    Mat blurred_img;
    GaussianBlur(gray_img, blurred_img, Size(5, 5), 2);
    
    //  calculate the gradient by using Sobel
    Mat gradX, gradY;
    Sobel(blurred_img, gradX, CV_32F, 1, 0, 3);
    Sobel(blurred_img, gradY, CV_32F, 0, 1, 3);

    //  innitialize position of points
    namedWindow("Active Contour", 0);
    resizeWindow("Active Contour", 700, 700);
    imshow("Active Contour", imgCopy);
    setMouseCallback("Active Contour", onMouse, &imgCopy);
    cout << "Enter '0' to continue..." << endl;
    while(waitKey(0)!='0');
    drawPoints(imgCopy);

    for (int i = 0; i < MAX_ITERATION; ++i) {
        ACTIVE_CONTOUR(gradX, gradY);
        cout << "Update points in turn " << i << endl;
        if(converged) break;
    }
    
    cout << "Finish !" << endl << "Enter '0' to exit" << endl;
    while(waitKey(0)!='0');
    return 0;
}
