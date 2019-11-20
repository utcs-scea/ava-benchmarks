////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "mergesort.h"
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096
#define DATA_SIZE (1024)
#define MAX_SOURCE_SIZE (0x100000)


cl_device_id device_id;             // compute device id
cl_context mergeContext;                 // compute context
cl_command_queue mergeCommands;
cl_program mergeProgram;                 // compute program
cl_kernel mergeFirstKernel;                   // compute kernel
cl_kernel mergePassKernel;
cl_kernel mergePackKernel;
cl_int err;
cl_mem d_resultList_first_buff;
cl_mem d_origList_first_buff;
cl_mem constStartAddr;
cl_mem finalStartAddr;
cl_mem nullElems;
cl_mem d_orig;
cl_mem d_res;
cl_float4 *d_resultList_first_altered = NULL;
double mergesum = 0;

extern int platform_id_inuse;
extern int device_id_inuse;

#ifdef TIMING
#include "timing.h"

extern struct timeval tv;
extern struct timeval tv_total_start, tv_total_end;
extern struct timeval tv_init_start, tv_init_end;
extern struct timeval tv_h2d_start, tv_h2d_end;
extern struct timeval tv_d2h_start, tv_d2h_end;
extern struct timeval tv_kernel_start, tv_kernel_end;
extern struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
extern struct timeval tv_close_start, tv_close_end;
extern float init_time, mem_alloc_time, h2d_time, kernel_time,
      d2h_time, close_time, total_time;
#endif

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
void init_mergesort(int listsize){
    cl_uint num = 0;
    clGetPlatformIDs(0,NULL,&num);
    cl_platform_id platformID[num];
    clGetPlatformIDs(num,platformID,NULL);
    clGetDeviceIDs(platformID[platform_id_inuse],CL_DEVICE_TYPE_ALL,0,NULL,&num);
    cl_device_id devices[num];
    err = clGetDeviceIDs(platformID[platform_id_inuse],CL_DEVICE_TYPE_ALL,num,devices,NULL);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group! (Init_mergesort)\n");
        exit(1);
    }
    char name[128];
    
    
    clGetDeviceInfo(devices[device_id_inuse],CL_DEVICE_NAME,128,name,NULL);
    
    mergeContext = clCreateContext(0, 1, &devices[device_id_inuse], NULL, NULL, &err);
    
    mergeCommands = clCreateCommandQueue(mergeContext, devices[device_id_inuse], CL_QUEUE_PROFILING_ENABLE, &err);
    
    d_resultList_first_altered = (cl_float4 *)malloc(listsize*sizeof(float));
    d_resultList_first_buff = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, listsize * sizeof(float),NULL,NULL);
    d_origList_first_buff = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, listsize * sizeof(float),NULL,NULL);
    
    d_orig = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, listsize * sizeof(float),NULL,NULL);
    d_res = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, listsize * sizeof(float),NULL,NULL);
    
    FILE *fp;
    const char fileName[]="./mergesort.cl";
    size_t source_size;
    char *source_str;
    
    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load mergesort kernel.\n");
		exit(1);
	}
    
    
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

	fclose(fp);
    mergeProgram = clCreateProgramWithSource(mergeContext, 1, (const char **) &source_str, (const size_t *)&source_size, &err);
    if (!mergeProgram)
    {
        printf("Error: Failed to create merge compute program!\n");
        exit(1);
    }
    
    err = clBuildProgram(mergeProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build merge program executable!\n");
        clGetProgramBuildInfo(mergeProgram, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
}

void finish_mergesort() {
    clReleaseMemObject(constStartAddr);
    clReleaseMemObject(nullElems);
    clReleaseMemObject(finalStartAddr);
    clReleaseMemObject(d_orig);
    clReleaseMemObject(d_res);
    clReleaseMemObject(d_origList_first_buff);
    clReleaseMemObject(d_resultList_first_buff);
    clReleaseKernel(mergeFirstKernel);
    clReleaseProgram(mergeProgram);
    clReleaseCommandQueue(mergeCommands);
    clReleaseContext(mergeContext);
}
cl_float4* runMergeSort(int listsize, int divisions,
                        cl_float4 *d_origList, cl_float4 *d_resultList,
                        int *sizes, int *nullElements,
                        unsigned int *origOffsets){
    
    int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
	int largestSize = -1;
	startaddr[0] = 0;
	for(int i=1; i<=divisions; i++)
	{
		startaddr[i] = startaddr[i-1] + sizes[i-1];
		if(sizes[i-1] > largestSize) largestSize = sizes[i-1];
	}
	largestSize *= 4;
    
    mergeFirstKernel = clCreateKernel(mergeProgram, "mergeSortFirst", &err);
    if (!mergeFirstKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create merge sort first compute kernel!\n");
        exit(1);
    }
    
    err = clEnqueueWriteBuffer(mergeCommands, d_resultList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_resultList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_resultList_first_buff source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(mergeCommands, d_origList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_origList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_origList_first_buff source array!\n");
        exit(1);
    }
    
    err = 0;
    err  = clSetKernelArg(mergeFirstKernel, 0, sizeof(cl_mem), &d_origList_first_buff);
    err  = clSetKernelArg(mergeFirstKernel, 1, sizeof(cl_mem), &d_resultList_first_buff);
    err  = clSetKernelArg(mergeFirstKernel, 2, sizeof(cl_int), &listsize);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set merge first kernel arguments! %d\n", err);
        exit(1);
    }
#ifdef MERGE_WG_SIZE_0
    const int THREADS = MERGE_WG_SIZE_0;
#else
    const int THREADS = 256;
#endif
    size_t local[] = {THREADS,1,1};
	int blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
	size_t global[] = {blocks*THREADS,1,1};
    size_t grid[] = {blocks,1,1,1};
    
    err = clEnqueueNDRangeKernel(mergeCommands, mergeFirstKernel, 3, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute mergeFirst kernel!\n");
        exit(1);
    }
    clFinish(mergeCommands);

    err = clEnqueueReadBuffer( mergeCommands, d_resultList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_resultList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read prefix output array! %d\n", err);
        exit(1);
    }

//    for(int i = 0; i < listsize/4;i++) {
//        printf("RESULT %f \n", d_resultList[i].s[0]);
//        printf("RESULT %f \n", d_resultList[i].s[1]);
//        printf("RESULT %f \n", d_resultList[i].s[2]);
//        printf("RESULT %f \n", d_resultList[i].s[3]);
//    }

//    for(int i =0; i < listsize/4; i++) {
//        printf("TEST %f \n", d_resultList[i].s[0]);
//        printf("TEST %f \n", d_resultList[i].s[1]);
//        printf("TEST %f \n", d_resultList[i].s[2]);
//        printf("TEST %f \n", d_resultList[i].s[3]);
//    }
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
    constStartAddr = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, (divisions+1)*sizeof(int),NULL,NULL);
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    err = clEnqueueWriteBuffer(mergeCommands, constStartAddr, CL_TRUE, 0, (divisions+1)*sizeof(int), startaddr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to constStartAddr source array!\n");
        exit(1);
    }

    mergePassKernel = clCreateKernel(mergeProgram, "mergeSortPass", &err);
    if (!mergePassKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create merge sort pass compute kernel!\n");
        exit(1);
    }

    double mergePassTime = 0;
    int nrElems = 2;
	while(1==1){
		int floatsperthread = (nrElems*4);
//        printf("FPT %d \n", floatsperthread);
		int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread);
//        printf("TPD %d \n",threadsPerDiv);
		int threadsNeeded = threadsPerDiv * divisions;
//        printf("TN %d \n", threadsNeeded);
        
#ifdef MERGE_WG_SIZE_1
        local[0] = MERGE_WG_SIZE_1;
#else
		local[0] = 208;
#endif
		grid[0] = ((threadsNeeded%local[0]) == 0) ?
        threadsNeeded/local[0] :
        (threadsNeeded/local[0]) + 1;
		if(grid[0] < 8){
			grid[0] = 8;
			local[0] = ((threadsNeeded%grid[0]) == 0) ?
            threadsNeeded / grid[0] :
            (threadsNeeded / grid[0]) + 1;
		}
		// Swap orig/result list
		cl_float4 *tempList = d_origList;
		d_origList = d_resultList;
		d_resultList = tempList;
      
        err = clEnqueueWriteBuffer(mergeCommands, d_resultList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_resultList, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to d_resultList_first_buff source array!\n");
            exit(1);
        }

        err = clEnqueueWriteBuffer(mergeCommands, d_origList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_origList, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to d_origList_first_buff source array!\n");
            exit(1);
        }
        err = 0;
        err  = clSetKernelArg(mergePassKernel, 0, sizeof(cl_mem), &d_origList_first_buff);
        err  = clSetKernelArg(mergePassKernel, 1, sizeof(cl_mem), &d_resultList_first_buff);
        err  = clSetKernelArg(mergePassKernel, 2, sizeof(cl_int), &nrElems);
        err  = clSetKernelArg(mergePassKernel, 3, sizeof(int), &threadsPerDiv);
        err  = clSetKernelArg(mergePassKernel, 4, sizeof(cl_mem), &constStartAddr);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set merge pass kernel arguments! %d\n", err);
            exit(1);
        }
		
        global[0] = grid[0]*local[0];
        err = clEnqueueNDRangeKernel(mergeCommands, mergePassKernel, 3, NULL, global, local, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute mergePass kernel!\n");
            exit(1);
        }
        
        clFinish(mergeCommands);
        err = clEnqueueReadBuffer( mergeCommands, d_resultList_first_buff, CL_TRUE, 0, listsize*sizeof(float), d_resultList, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read prefix output array! %d\n", err);
            exit(1);
        }
        clFinish(mergeCommands);
    	nrElems *= 2;
		floatsperthread = (nrElems*4);
        
		if(threadsPerDiv == 1) break;
	}
    printf("Merge Pass Kernel Time: %0.3f \n", mergePassTime);
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
    finalStartAddr = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, (divisions+1)*sizeof(int),NULL,NULL);
    nullElems = clCreateBuffer(mergeContext,CL_MEM_READ_WRITE, (divisions)*sizeof(int),NULL,NULL);
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    err = clEnqueueWriteBuffer(mergeCommands, finalStartAddr, CL_TRUE, 0, (divisions+1)*sizeof(int), origOffsets, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to finalStartAddr source array!\n");
        exit(1);
    }
    
    err = clEnqueueWriteBuffer(mergeCommands, nullElems, CL_TRUE, 0, (divisions)*sizeof(int), nullElements, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to nullElements source array!\n");
        exit(1);
    }
#ifdef MERGE_WG_SIZE_0
    local[0] = MERGE_WG_SIZE_0;
#else
    local[0] = 256;
#endif
	grid[0] = ((largestSize%local[0]) == 0) ?
    largestSize/local[0] :
    (largestSize/local[0]) + 1;
	grid[1] = divisions;
//    grid[0] = 17;
    global[0] = grid[0]*local[0];
    global[1] = grid[1]*local[1];
    
    mergePackKernel = clCreateKernel(mergeProgram, "mergepack", &err);
    if (!mergePackKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create merge sort pack compute kernel!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(mergeCommands, d_res, CL_TRUE, 0, listsize*sizeof(float), d_resultList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_resultList_first_buff source array!\n");
        exit(1);
    }
    
    err = clEnqueueWriteBuffer(mergeCommands, d_orig, CL_TRUE, 0, listsize*sizeof(float), d_origList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_origList_first_buff source array!\n");
        exit(1);
    }
    err = 0;
    err  = clSetKernelArg(mergePackKernel, 0, sizeof(cl_mem), &d_res);
    err  = clSetKernelArg(mergePackKernel, 1, sizeof(cl_mem), &d_orig);
    err  = clSetKernelArg(mergePackKernel, 2, sizeof(cl_mem), &constStartAddr);
    err  = clSetKernelArg(mergePackKernel, 3, sizeof(cl_mem), &nullElems);
    err  = clSetKernelArg(mergePackKernel, 4, sizeof(cl_mem), &finalStartAddr);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set merge pack kernel arguments! %d\n", err);
        exit(1);
    }
    err = clEnqueueNDRangeKernel(mergeCommands, mergePackKernel, 3, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute merge pack kernel!\n");
        exit(1);
    }

    clFinish(mergeCommands);
    err = clEnqueueReadBuffer( mergeCommands, d_orig, CL_TRUE, 0, listsize*sizeof(float), d_origList, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read origList array! %d\n", err);
        exit(1);
    }
    
    
	free(startaddr);
	return d_origList;
    
}

double getMergeTime() {
  return mergesum;
}
