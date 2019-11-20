/***********************************************************************
*
* Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
* See LICENSE file in the project root for full license information.
*
************************************************************************/

#include <iostream>
#include <memory>
#include <fstream>
#include <cstring>
#include "GTILib.h"
//#include "GtiLog.h"
#define LOG(IO) std::cout

#ifdef LINUX
#include <unistd.h>
#include <sys/time.h>
unsigned long GetTickCount64_us()
{
    struct timeval tv;
    if(gettimeofday(&tv, NULL) != 0)
            return 0;
    return (tv.tv_sec * 1000000) + (tv.tv_usec);
}
#elif defined ANDROID
#include <time.h>
unsigned long GetTickCount64_us()
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts) != 0) {
        return 0;
    }
    return (ts.tv_sec * 1000000) + (ts.tv_nsec/1000);
}
#endif

#ifdef WIN32
#include <windows.h>
unsigned long GetTickCount64_us()
{
    return GetTickCount64() * 1000;
}
#endif

//#define CHECK_RESULTS
#define PRINT_LABEL

using namespace std;

static unique_ptr<char[]> file_to_buffer(char* filename, int *sizeptr)
{
    std::ifstream fin(filename, ios::in | ios::binary );
    if(!fin.is_open())
    {
        cout<<"Could not open file: "<< filename << endl; 
        exit(-1);
    }

    fin.seekg(0,std::ios::end);
    *sizeptr=fin.tellg();
    fin.seekg(0,std::ios::beg);
    unique_ptr<char[]> buffer(new char[*sizeptr]);

    fin.read((char *)buffer.get(), *sizeptr);
    fin.close();
    return move(buffer);
}

static void dump_buffer(const string &filename, char* buf, int size)
{
    std::ofstream fout(filename, ios::out | ios::binary );
    if(!fout.is_open())
    {
        cout<<"Could not open file: "<< filename << endl; 
        exit(-1);
    }
    fout.write(buf, size);
    fout.close();
}

template <class T> 
T hash(unsigned char *p, unsigned int size){
        T ret=0;
        while(size-->0) ret+=(T)(31*(*p++));
        return ret;
}

int main(int argc, char**argv)
{
    int loops=1;    
    if(argc<3)
    {
        cout<<"usage: "<<argv[0]<<" GTI_model_file image_file [loops]"<<endl; 
        cout<<"   Ex: "<<argv[0]<<" ../Models/.../gti_gnet3.model ../Data/Image_lite/bridge_c20.bin"<<endl;
        cout<<"       "<<argv[0]<<" ../Models/.../gti_mnet.model ../Data/Image_lite/swimming_c40.jpg 10"<<endl;

        exit(-1);
    }
    if(argc>=4) 
        loops=atoi(argv[3]);

    //Read model data from file
    int modelsize=0; 
    unique_ptr<char[]> modelfile = file_to_buffer(argv[1], &modelsize);

    //Read 224x224 BGR plannar format image data from file
    GtiTensor tensor;
    int datasize=0;
    unique_ptr<char[]> datafile = file_to_buffer(argv[2], &datasize);
    tensor.height=1;
    tensor.width=1;
    tensor.depth=datasize;
    tensor.size = datasize; // input size: height*width*depth; output size: size
    tensor.buffer=datafile.get();

    //Load GTI image classification model
    GtiModel *model=GtiCreateModelFromBuffer((void*)modelfile.get(), modelsize);
    //GtiModel *model=GtiCreateModel(argv[1]);
    if( model == nullptr){
        cout<<"Could not create model: "<< argv[1] << endl; 
        exit(-1);
    }

    char label[100];
    const char *p;
    long startTime;
    long endTime;

    for(int j=0;j<loops;j++)
    {
        //Get the inference results from chip
        startTime = GetTickCount64_us();
        GtiTensor *tensorOut=GtiEvaluate(model,&tensor);
        endTime = GetTickCount64_us();
        if(tensorOut==0)
        {
            LOG(ERROR)<<"evaluate error,exit";
            break;
        }
        #ifdef CHECK_RESULTS
        unsigned int v=::hash<unsigned int>((unsigned char *)tensorOut->buffer,tensorOut->size);
        if(v != 957714) {
            cout<<"output hash:"<<v <<endl;
            cout << "Checksum error, FAILED.\n";
           // exit(-1);
        }
        else {
            cout<<"output hash:"<<v <<endl;
            cout << "PASSED" <<endl;
        }
        #endif

        #ifdef PRINT_LABEL

        cout<<(char *)tensorOut->buffer<<endl;
        p = std::strstr((char *)tensorOut->buffer + 20, (const char *)"label");
        p = std::strchr((const char *)p+7, ' ');
        for(int k = 0; k < 100; k++)
        {
            if(*p == '\"')
            {
                label[k] = 0;
                break;
            }
            label[k] = *p++;
        }
        cout<<"\nRESULT:" << label <<endl;
        #else
        cout << "Output size: " << tensorOut->size << ", output format: " << tensorOut->format << endl;
        #endif
        
        float inferenceTime = (endTime - startTime)/1000.0;
        float fps = (1000.0)/inferenceTime;
        cout << endl << "Image inference time = " << inferenceTime  << " ms,  FPS = " << fps << endl <<endl;

        if (j>0)
	       cout << "loop " << j + 1<< endl;
    }
    GtiDestroyModel(model);
    return 0;
}

