# Parallel_OpenMpCUDA_KmeansAlgorithm
Project: K-Means clustering Algorithm for Customer Segmentation (Parallelizing K-Means: Serial, OpenMP, and CUDA Approaches)
(Programs which uses Parallel computing Concepts in order to speedup the execution time)


a. Motivation of the project:
• Businesses need efficient clustering methods for large datasets to enhance decision-making.
• K-Means, with its simplicity and scalability, is a natural choice for customer segmentation but
demands high computational power for large datasets.
• Growing customer datasets lead to higher computational demands for clustering and running K- Means on large datasets can be computationally expensive. Hence there is a need to improve speed and efficiency using parallel computing techniques.

b. Objectives of the project:
• The objective of the project is to develop a parallelized version of the K-Means clustering algorithm to enhance its performance for large-scale datasets commonly encountered in business analytics.
• Accelerate customer segmentation by leveraging parallel computing techniques.
• Achieve faster and more efficient clustering without compromising the accuracy of results.


// Generate random data 

// compile 
g++ -o gen_data gen_data.cpp

// execute 
./gen_data <output_file_name> <rec_count> <low rang1> <high range1> <low rang2> <high range2>

example : ./gen_data input_1000.csv 1000 50 400 10 150


###############################

// Customer Segmentation

//Compile 

make kmeans_parallel

make kmeans_serial

//execute 

./kmeans_parallel <Input Data Filename> <Output Filename> <Number of Iterations> <Number of Clusters> [-t <num_threads>]

./kmeans_serial <Input Data Filename> <Output Filename> <Number of Iterations> <Number of Clusters> [-t <num_threads>]

example : 

./kmeans_parallel input_1000.csv out_1000 10000 5 -t8

./kmeans_serial input_1000.csv out_1000 10000 5 -t8


module load OpenCV/4.10.0

./kmeans_colour_quantisation inputImage.png 100 32

./kmeans_colour_quantisation_openMP inputImage.png 100 32

./kmeans_cuda input_1000.csv out_1000 10000 5 

./kmeans_colour_quantisation_waveHPC inputImage.png 100 32

./kmeans_colour_quantisation_waveHPC_CUDA inputImage.png 100 32

./kmeans_colour_quantisation_openMP inputImage.png 100 32 16

###################################

