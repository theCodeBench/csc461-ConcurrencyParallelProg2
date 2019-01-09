/*************************************************************************//**
 * @file
 *
 * @mainpage Program 2 - Concurrency/Parallelism in C++
 *
 * @section CSC461 Programming Languages
 *
 * @author Ryan Hinrichs
 *
 * @date October 16, 2017
 *
 * @par Professor:
 *         Dr. Weiss
 *
 * @par Course:
 *         CSC461 - M001 - 11:00 a.m.
 *
 *
 * @section program_section Program Information
 *
 * @details
 *
 * Program Description:
 *
 * This program is designed to run the Sobel edge detection on an image
 * using three different methods to show the importance and usefulness of 
 * parallelism.  First, it runs the program on the regular CPU, then on 
 * multiple threads using OpenMP, and finally on the GPGPU using CUDA.
 *
 * @par Usage:
 * @verbatim
 * c:\> make
 * c:\> prog2 [png file to test]
 * c:\> make clean
 * @endverbatim
 *
 * @section todo_bugs_modification_section Todo, Bugs, and Modifications
 *
 * @todo Need to line up the speedup values by the decimal points.
 *
 *****************************************************************************/

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include "lodepng.h"
#include <cmath>
#include <chrono>
#include <ctime>
#include <thread>

using namespace std;

typedef unsigned char byte;

//Function prototypes
void cpuloop(byte* image, byte* cpuimage, int w, int h);
void openmploop(byte* &image, byte*& mpimage, int w, int h);
double cudacall(byte*& image, byte* cdimage, int w, int h);

/******************************************************************************
* Author: Ryan Hinrichs
*
* Function that runs on the NVIDIA GPU, performing the Sobel operation on 
* every pixel.
*
* Parameters:
*	byte* image     - original greyscale image
*	byte* cudaimage - image array that is filled with the changed pixels
*	int width       - width of the image
*	int npixels     - total number of pixels
******************************************************************************/
__global__ void init_gpu(byte* image, byte* cudaimage, int width, int npixels)
{
	//Calculates what index we are at based on thread placement
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    //Runs Sobel operation on pixel, if the pixel is not on the edge
    if( i > width && i < (npixels - width - 1)){
    
            //Calculates Gx
            int totalgx = -1*image[i-width-1];
            totalgx += -2*image[i-width];
            totalgx += -1*image[i-width+1];
            totalgx += image[i+width-1];
            totalgx += 2*image[i+width];
            totalgx += image[i+width-1];
    
            //Calculates Gy
            int totalgy = -1*image[i-width-1];
            totalgy += -2*image[i-1];
            totalgy += -1*image[i+width-1];
            totalgy += image[i-width+1];
            totalgy += 2*image[i+1];
            totalgy += image[i+width+1];
            
            float temp = sqrt((float)totalgx*totalgx+totalgy*totalgy);
            
        if(temp > 255) 
            temp = 255;
            
        cudaimage[i] = (byte) temp;
            

    }else if(i < npixels)
        cudaimage[i] = 0;
}

/******************************************************************************
* Author: Ryan Hinrichs
*
* Main function, performs initialization of CUDA properties, checks command
* arguments, and outputs correct information prior to running Sobel and the 
* results after running the Sobel operation three times.
*
* Parameters:
*	int argc    - number of command line arguments, including the program
*	char** argv - array of command line arguments
******************************************************************************/
int main(int argc, char** argv)
{
    //Variable initilization
    string filename = "";
    double cuda = 0;
    
    // CUDA device properties
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0);
    int cores = devProp.multiProcessorCount;
    switch( devProp.major )
    {
        case 2: //Fermi
            if (devProp.minor == 1) cores *= 48;
            else cores *= 32; break;
        case 3: //Kepler
            cores *= 192; break;
        case 5: //Maxwell
            cores *= 128; break;
        case 6: //Pascal
            if (devProp.minor == 1) cores *= 128;
            else if (devProp.minor == 0) cores *= 64;
            break;
    }


    // check usage
    if ( argc < 2 )
    {
        cout<<"Using: "<<argv[0]<<" image.png"<<endl;
        return -1;
    }
	
	
    // read input PNG file
    byte* pixels;
    unsigned int width, height;
    unsigned error = lodepng_decode_file( &pixels, &width, &height, argv[1], LCT_RGBA, 8 );
    if ( error )
    {
        cout<<"Error while reading file "<<argv[1]<<endl;
        cout<<"Error code "<<error<<": "<<lodepng_error_text( error)<<endl;

        return -2;
    }else
    {
        filename = argv[1];
        int pos = filename.find(".");
        filename = filename.substr(0,pos);
    }
    
    
    //Print initial information
    time_t currtime = time( 0 );
    cout<<"edge map benchmarks "<<ctime( &currtime);
    cout<<"CPU: "<<thread::hardware_concurrency()<<" hardware threads"<<endl;
    cout<<"GPGPU: "<<devProp.name<<", CUDA "<< devProp.major<<"."<<devProp.minor;
    cout<<", "<< devProp.totalGlobalMem/1048576 << " Mbytes global memory, ";
    cout<<cores<<" CUDA cores"<<endl<<endl; 
    cout<<"Processing "<<argv[1]<<": "<<height<< " rows x "<<width<<" columns"<<endl;

    // copy 24-bit RGB data into 8-bit grayscale intensity array
    int npixels = width * height;
    byte* image = new byte [ npixels ];
    byte* newimage = new byte [ npixels ] ;
    byte* mpimage = new byte [ npixels ] ;
    byte* cdimage = new byte [ npixels ] ;
    byte* img = pixels;
    for ( int i = 0; i < npixels; ++i )
    {
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;     // alpha channel is not used
        image[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }
    free( pixels );     // LodePNG uses malloc, not new


    //Runs and times CPU Sobel implementation
    auto prevtime = chrono::system_clock::now();
    cpuloop( image, newimage, width , height);
    chrono::duration<double> cpu = chrono::system_clock::now() - prevtime;
    
   
    //Runs and times OpenMP Sobel implementation
    prevtime = chrono::system_clock::now();
    openmploop(image, mpimage, width, height);
    chrono::duration<double> mp = chrono::system_clock::now() - prevtime;


    //Runs and times CUDA Sobel implementation, returning the calculated time
    cuda = cudacall(image, cdimage, width, height);
    
    
    //Outputs final times and speedup comparison
    cout<<setw(1);
    cout<<"CPU execution time = "<<setprecision(4)<<1000 * cpu.count()<<" msec"<<endl;
    cout<<"OpenMP execution time = "<<setprecision(3)<<1000 * mp.count()<<" msec"<<endl;
    cout<<"CUDA execution time = "<<setprecision(4)<<1000 * cuda<<" msec"<<endl<<endl;
    cout<<"CPU->OMP speedup:";
    cout<<setw(10);
    cout<<setprecision(5)<<cpu.count()/mp.count()<<" X"<<endl;
    cout<<"OMP->GPU speedup:";
    cout<<setw(10);
    cout<<setprecision(5)<<mp.count()/cuda<<" X"<<endl;
    cout<<"CPU->GPU speedup:";
    cout<<setw(10);
    cout<<setprecision(5)<<cpu.count()/cuda<<" X"<<endl;


    //Creates new file names based on method of Sobel operation
    string cpuname = filename + "_cpu.png";
    string mpname = filename + "_mp.png";
    string cudaname = filename + "_cuda.png";


    // write grayscale PNG files, error checking each one
    error =  lodepng_encode_file( cpuname.c_str(), newimage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        cout<<"Error while writing file " << cpuname << endl;
        cout<<"Error code "<< error<<": "<<lodepng_error_text( error );

        return -3;
    }
    error =  lodepng_encode_file( mpname.c_str(), mpimage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        cout<<"Error while writing file " << mpname << endl;
        cout<<"Error code "<< error<<": "<<lodepng_error_text( error );

        return -3;
    }
    error =  lodepng_encode_file( cudaname.c_str(), cdimage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        cout<<"Error while writing file " << cudaname << endl;
        cout<<"Error code "<< error<<": "<<lodepng_error_text( error );

        return -3;
    }
    

    //Free up memory
    delete [] image;
    delete [] mpimage;
    delete [] newimage;

    return 0;
}

/******************************************************************************
* Author: Ryan Hinrichs
*
* Function that runs on the CPU, performing the Sobel operation on every pixel.
*
* Parameters:
*	byte* image    - original greyscale image
*	byte* cpuimage - image array that is filled with the changed pixels
*	int w          - width of the image
*	int h          - height of the image
******************************************************************************/
void cpuloop(byte* image, byte* cpuimage, int w, int h)
{
    int width = w;
    int height = h;
    int npixels = width*height;
    
    //Runs Sobel operation on pixel, if the pixel is not on the edge
    for( int i = w; i<((w*h) - w - 1); i++)
    {
        if( i > width && i < (npixels - width)){
            //Calculates Gx
            int totalgx = -1*image[i-width-1];
            totalgx += -2*image[i-width];
            totalgx += -1*image[i-width+1];
            totalgx += image[i+width-1];
            totalgx += 2*image[i+width];
            totalgx += image[i+width-1];
    
            //Calculates Gy
            int totalgy = -1*image[i-width-1];
            totalgy += -2*image[i-1];
            totalgy += -1*image[i+width-1];
            totalgy += image[i-width+1];
            totalgy += 2*image[i+1];
            totalgy += image[i+width+1];

            cpuimage[i] = sqrt((pow(totalgx,2)+pow(totalgy,2)));
            
            if(cpuimage[i] > 255) cpuimage[i] = 255;
        }else
            cpuimage[i] = image[i];
    }
}

/******************************************************************************
* Author: Ryan Hinrichs
*
* Function that runs on multiple threads using OpenMP, performing the Sobel 
* operation on every pixel.
*
* Parameters:
*	byte* image   - original greyscale image
*	byte* mpimage - image array that is filled with the changed pixels
*	int w         - width of the image
*	int h         - height of the image
******************************************************************************/
void openmploop(byte*& image, byte*& mpimage, int w, int h)
{
    //Runs Sobel operation on pixel, if the pixel is not on the edge
    #pragma omp parallel for
    for( int i = w; i<((w*h) - w - 1); i++)
    {
        int width = w;          //Variables are initialized with the threads
        int height = h;
        int npixels = width*height;
        if( i > width && i < (npixels - width)){
            //Calculates Gx
            int totalgx = -1*image[i-width-1];
            totalgx += -2*image[i-width];
            totalgx += -1*image[i-width+1];
            totalgx += image[i+width-1];
            totalgx += 2*image[i+width];
            totalgx += image[i+width-1];
    
            //Calculates Gy
            int totalgy = -1*image[i-width-1];
            totalgy += -2*image[i-1];
            totalgy += -1*image[i+width-1];
            totalgy += image[i-width+1];
            totalgy += 2*image[i+1];
            totalgy += image[i+width+1];

            mpimage[i] = sqrt((pow(totalgx,2)+pow(totalgy,2)));
            
            if(mpimage[i] > 255) mpimage[i] = 255;
        }else
            mpimage[i] = image[i];
    }
}

/******************************************************************************
* Author: Ryan Hinrichs
*
* Function that prepares, times, and runs the Sobel process on the CUDA GPU.
*
* Parameters:
*	byte* imag    - original greyscale image
*	byte* cdimage - image array that is filled with the changed pixels
*	int w         - width of the image
*	int h         - height of the image
******************************************************************************/
double cudacall(byte*& image, byte* cdimage, int w, int h)
{
    byte* cudaimage;
    byte* d_image;
    cudaMalloc( ( void ** )&d_image, w*h);
    cudaMalloc( ( void ** )&cudaimage, w*h);

    cudaMemcpy( d_image, image, w*h, cudaMemcpyHostToDevice );
    
    int nThreads = 512;
    int nBlocks = (w*h + nThreads - 1) /nThreads;
    auto c = chrono::system_clock::now();
    init_gpu<<< nBlocks, nThreads >>>(d_image, cudaimage, w, w*h);
    cudaError_t cudaerror = cudaDeviceSynchronize();
    if( cudaerror != cudaSuccess ) 
        cout<<"Cuda failed to synchronize: "<< cudaGetErrorName( cudaerror )<<endl;

	chrono::duration<double> time_cuda = chrono::system_clock::now() - c;

    cudaMemcpy( cdimage, cudaimage, w*h, cudaMemcpyDeviceToHost );  

    cudaFree( cudaimage );
    cudaFree( d_image );
    
    return time_cuda.count();
}




