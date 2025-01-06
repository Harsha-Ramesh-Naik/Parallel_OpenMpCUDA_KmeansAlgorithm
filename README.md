# Parallel_OpenMpCUDA_KmeansAlgorithm
Project: K-Means clustering Algorithm for Customer Segmentation (Parallelizing K-Means: Serial, OpenMP, and CUDA Approaches

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

