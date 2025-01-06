#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace cv;

/// Global variables
string output_file_name = "output.csv";

/// Structure to store RGB colors (instead of x, y for 2D points)
struct ColorPoint {
    int r, g, b;   // RGB values
    int cluster;    // Cluster ID
    double minDist; // Minimum distance to the nearest cluster
	
    __host__ __device__
    ColorPoint() : r(0), g(0), b(0), cluster(-1), minDist(DBL_MAX) {}
	
    __host__ __device__
    ColorPoint(int r, int g, int b) : r(r), g(g), b(b), cluster(-1), minDist(DBL_MAX) {}

    __host__ __device__
    double distance(const ColorPoint& p) const {
        return ((p.r - r) * (p.r - r) + (p.g - g) * (p.g - g) + (p.b - b) * (p.b - b));
    }
};

// Device function to compute Euclidean distance (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
__device__ double euclideanDistance(ColorPoint p1, ColorPoint p2) {
    return (p1.r - p2.r) * (p1.r - p2.r) + (p1.g - p2.g) * (p1.g - p2.g) + (p1.b - p2.b) * (p1.b - p2.b);
}

/// CUDA Kernels

// Kernel to assign points to the nearest centroid
__global__ void assignClusters(ColorPoint* points, ColorPoint* centroids, int numPoints, int nClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < numPoints){
		double minDist = DBL_MAX;
		int bestCluster = -1;
		
		ColorPoint& thePoint = points[idx];
		for (int j = 0; j < nClusters; ++j) {
			double dist = euclideanDistance(thePoint, centroids[j]);

			if (dist < minDist) {
				minDist = dist;
				bestCluster = j;
			}
		}
		thePoint.minDist = minDist;
		thePoint.cluster = bestCluster;
	}
}

// Kernel to recompute centroids: Accumulate RGB values for each cluster
__global__ void updateCentroids(ColorPoint* points, ColorPoint* centroids, int* clusterCount, int numPoints, int nClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(idx < numPoints){
		int clusterId = points[idx].cluster;
		atomicAdd(&centroids[clusterId].r, points[idx].r);
		atomicAdd(&centroids[clusterId].g, points[idx].g);
		atomicAdd(&centroids[clusterId].b, points[idx].b);
		atomicAdd(&clusterCount[clusterId], 1);
	}
}

//  Kernel to recompute centroids: Finalize centroids by averaging accumulated values
__global__ void finalizeCentroids(ColorPoint* centroids, int* clusterCount, int nClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	 if(idx < nClusters){
		 if (clusterCount[idx] > 0) {
			centroids[idx].r /= clusterCount[idx];
			centroids[idx].g /= clusterCount[idx];
			centroids[idx].b /= clusterCount[idx];
		}
	 }

}

/// Perform k-means clustering on colors (RGB) using CUDA
void kMeansClusteringColorQuantizationCUDA(vector<ColorPoint>& colors, int numIterations, int nClusters, int threadsPerBlockXdim) {
    int numPoints = colors.size();

    // Allocate memory on GPU
    ColorPoint* d_points;
    ColorPoint* d_centroids;
    int* d_clusterCount;
    cudaMalloc(&d_points, numPoints * sizeof(ColorPoint));
    cudaMalloc(&d_centroids, nClusters * sizeof(ColorPoint));
    cudaMalloc(&d_clusterCount, nClusters * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_points, colors.data(), numPoints * sizeof(ColorPoint), cudaMemcpyHostToDevice);

    // Randomly initialize centroids
    vector<ColorPoint> centroids(nClusters);
    srand(time(0));
    for (int i = 0; i < nClusters; ++i) {
        centroids[i] = colors[rand() % numPoints];
    }
    cudaMemcpy(d_centroids, centroids.data(), nClusters * sizeof(ColorPoint), cudaMemcpyHostToDevice);

    // Configure CUDA kernel dimensions
	dim3 threadsPerBlock(threadsPerBlockXdim, 1, 1);
	dim3 numBlocks((numPoints + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);

    // Perform iterations
    for (int i = 0; i < numIterations; ++i) {
        // Reset counts on GPU
        cudaMemset(d_clusterCount, 0, nClusters * sizeof(int));

        // Assign clusters
        assignClusters<<<numBlocks, threadsPerBlock>>>(d_points, d_centroids, numPoints, nClusters);
        cudaDeviceSynchronize();

        // Update centroids
        updateCentroids<<<numBlocks, threadsPerBlock>>>(d_points, d_centroids, d_clusterCount, numPoints, nClusters);
        cudaDeviceSynchronize();

        // Finalize centroids
        finalizeCentroids<<<numBlocks, threadsPerBlock>>>(d_centroids, d_clusterCount, nClusters);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(colors.data(), d_points, numPoints * sizeof(ColorPoint), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_clusterCount);
}

/// Perform color quantization on the image
void colorQuantization(const Mat& image, int numIterations, int nClusters, int threadsPerBlockXdim) {
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

    // Apply K-means clustering on the colors using CUDA
    kMeansClusteringColorQuantizationCUDA(colors, numIterations, nClusters, threadsPerBlockXdim);

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

    // Load the image using OpenCV
    Mat imageObj = imread(inputImageFileName);
    if (imageObj.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }
	//-------run kMeansClustering mutliple times with different block dimensions-----
	int loopIterator = 0;
	int threadsPerBlockXdim = 8;
	while(loopIterator < 5){
		auto t1 = std::chrono::high_resolution_clock::now();
		
		// Perform color quantization
		colorQuantization(imageObj, numIterations, nClusters, threadsPerBlockXdim);
		
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> runTime = t2 - t1;
		printf("Application: colour quantisation of image. Parallelisation: cuda. Data file: %s, Time: %.3f, using blockSize (%d, %d, %d) \n", 
				inputImageFileName.c_str(), runTime, threadsPerBlockXdim, 1, 1);
				
		threadsPerBlockXdim = threadsPerBlockXdim + 8;
		loopIterator++;
	}


    return 0;
}
