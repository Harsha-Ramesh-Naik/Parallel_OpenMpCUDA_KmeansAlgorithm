CC = g++

# Set this variable to 1 when running on HPC, set to 0 when running locally
#	Always compile kmeans_serial kmeans_parallel
#		- when on local machine, only compile: kmeans_colour_quantisation kmeans_colour_quantisation_openMP
#		- when on waveHPC, only compile: kmeans_cuda kmeans_color_quantization_waveHPC kmeans_color_quantization_waveHPC_CUDA
HPC = 0

default: kmeans_serial kmeans_parallel \
         $(if $(filter 1, $(HPC)), kmeans_cuda kmeans_colour_quantisation_waveHPC kmeans_colour_quantisation_waveHPC_CUDA, \
		 kmeans_colour_quantisation kmeans_colour_quantisation_openMP)

kmeans_serial: kmeans_serial.cpp
	${CC} -O0 -g  -std=c++11 -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ kmeans_serial.cpp

kmeans_parallel: kmeans_parallel.cpp
	${CC} -O0 -g  -std=c++11 -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ kmeans_parallel.cpp


kmeans_colour_quantisation: kmeans_colour_quantisation.cpp
	g++ -o kmeans_colour_quantisation kmeans_colour_quantisation.cpp `pkg-config --cflags --libs opencv4` -fopenmp	
	
kmeans_colour_quantisation_openMP: kmeans_colour_quantisation_openMP.cpp
	g++ -o kmeans_colour_quantisation_openMP kmeans_colour_quantisation_openMP.cpp `pkg-config --cflags --libs opencv4` -fopenmp		
	
#----------------------------------------------------------------------------------------

kmeans_cuda: kmeans_cuda.cu
	nvcc  -arch=sm_60 -DCUDA -o kmeans_cuda kmeans_cuda.cu 

kmeans_colour_quantisation_waveHPC: kmeans_colour_quantisation.cpp
	g++ -o kmeans_colour_quantisation_waveHPC kmeans_colour_quantisation.cpp \
	-I/WAVE/apps/el8/packages/OpenCV/4.10.0-CUDA/app/include -L/WAVE/apps/el8/packages/OpenCV/4.10.0-CUDA/app/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
	-fopenmp	

kmeans_colour_quantisation_waveHPC_CUDA: kmeans_cuda_colour_quantisation.cu
	nvcc -arch=sm_60 -DCUDA -o kmeans_colour_quantisation_waveHPC_CUDA kmeans_cuda_colour_quantisation.cu \
	-I/WAVE/apps/el8/packages/OpenCV/4.10.0-CUDA/app/include -L/WAVE/apps/el8/packages/OpenCV/4.10.0-CUDA/app/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs 

	
#------------------------------------------------------------------------------------------
clean:
	-rm -vf  kmeans_serial kmeans_parallel kmeans_cuda kmeans_colour_quantisation kmeans_colour_quantisation_openMP \
			kmeans_colour_quantisation_waveHPC kmeans_colour_quantisation_waveHPC_CUDA




