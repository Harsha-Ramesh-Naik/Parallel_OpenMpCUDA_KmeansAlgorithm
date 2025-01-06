#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace cv;

/// Global variables
int nthreads = 0;
string output_file_name = "output.csv";

/// Structure to store RGB colors (instead of x, y for 2D points)
struct ColorPoint {
    int r, g, b;   // RGB values
    int cluster;    // Cluster ID
    double minDist; // Minimum distance to the nearest cluster
	
	//default constructor: initializes rgb values to zero, and cluster to -1, and minDist to the max distance
    ColorPoint() : r(0), g(0), b(0), cluster(-1), minDist(DBL_MAX) {}
	
	//constructor: initialise rgb to given values
    ColorPoint(int r, int g, int b) : r(r), g(g), b(b), cluster(-1), minDist(DBL_MAX) {}
	
	//compute distance using euclidean distance formula between 3 points
    double distance(const ColorPoint &p) const {
        return ((p.r - r) * (p.r - r) + (p.g - g) * (p.g - g) + (p.b - b) * (p.b - b));
    }
};

/// Perform k-means clustering serially on colors (RGB)
void kMeansClusteringColorQuantization(vector<ColorPoint>& colors, int numIterations, int nClusters) {
    int n = colors.size();

    // Randomly initialize nClusters many centroids (randomly select nClusters initial points)
    vector<ColorPoint> centroids;
    srand(time(0));
    for (int i = 0; i < nClusters; ++i) {
        centroids.push_back(colors[rand() % n]);
    }
    // Perform the K-means clustering process multiple times
    for (int i = 0; i < numIterations; ++i) {
        // -------------Assigning each ColorPoint to the nearest centroid--------------
		
		//for each centroid, compute the distance between each point and the centroid
        for (auto& point : colors) {
            double minDist = DBL_MAX;
			
			//for each point: if it's closest to this centroid then assign it to this centroid
            for (int j = 0; j < nClusters; ++j) {
                double dist = centroids[j].distance(point);
                if (dist < minDist) {
                    minDist = dist;
                    point.cluster = j; // Assign the point to the nearest centroid
                }
            }
            point.minDist = minDist;
        }

        // ------------Recomputing centroids (mean of all points in each cluster)-----------
        vector<int> nPointsInCluster(nClusters, 0);
        vector<int> sumR(nClusters, 0), sumG(nClusters, 0), sumB(nClusters, 0);
		
		//accumulate stats for each cluster 
        for (auto& point : colors) {
            int clusterId = point.cluster;
            nPointsInCluster[clusterId]++;
            sumR[clusterId] += point.r;
            sumG[clusterId] += point.g;
            sumB[clusterId] += point.b;
        }
		
		//for each centroid, compute its new centroid using the mean of all its points
        for (int j = 0; j < nClusters; ++j) {
            if (nPointsInCluster[j] > 0) {
                centroids[j].r = sumR[j] / nPointsInCluster[j];
                centroids[j].g = sumG[j] / nPointsInCluster[j];
                centroids[j].b = sumB[j] / nPointsInCluster[j];
            }
        }
    }

    // Assign the final cluster colors back to the image
    for (auto& point : colors) {
        int clusterId = point.cluster;
        point.r = centroids[clusterId].r;
        point.g = centroids[clusterId].g;
        point.b = centroids[clusterId].b;
    }
}

void colorQuantization(const Mat& image, int numIterations, int nClusters) {
    vector<ColorPoint> colors;

    // Convert the image to a vector of colors
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            int r = pixel[2];  // Red channel
            int g = pixel[1];  // Green channel
            int b = pixel[0];  // Blue channel
            colors.push_back(ColorPoint(r, g, b));
        }
    }

    // Apply K-means clustering on the colors
    kMeansClusteringColorQuantization(colors, numIterations, nClusters);

    // Convert the colors back to an image
    Mat quantizedImage(image.rows, image.cols, CV_8UC3);
    int idx = 0;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            quantizedImage.at<Vec3b>(i, j) = Vec3b(colors[idx].b, colors[idx].g, colors[idx].r);
            idx++;
        }
    }
    imwrite("quantized_image.png", quantizedImage); // Save the quantized image
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Invalid options." << endl;
        exit(1);
    }

    string inputImageFileName = argv[1];
    int numIterations = atoi(argv[2]);
    int nClusters = atoi(argv[3]);
	
	printf("from the cmdline args: inputImageFileName is %s, numIterations is %d, and nClusters is %d\n",
			inputImageFileName.c_str(), numIterations, nClusters);
	
    // Load the image using OpenCV
    Mat imageObj = imread(inputImageFileName);
    if (imageObj.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }
	
	double t1,t2;
	t1 = omp_get_wtime();
    // Perform color quantization
    colorQuantization(imageObj, numIterations, nClusters);
	t2 = omp_get_wtime();
	printf("serial algorithm for colour quantisation. Data file: %s, Time: %.3f\n", inputImageFileName.c_str(), t2-t1);

    return 0;
}