/*
 * demo.c
 * To compile: gcc nanojpeg.c demo.c -ltensorflow
 * To run: ./a.out data/inception_v3_2016_08_28_frozen.pb data/imagenet_slim_labels.txt data/grace_hopper.jpg
 *
 * author: Miguel Jiménez
 * date: May 9, 2017
 */
#define _NJ_INCLUDE_HEADER_ONLY

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "nanojpeg.c"
#include <mvnc.h>

#include <sys/time.h>

#define NUM_ITR 1

struct timestamp {
    struct timeval start;
    struct timeval end;
};

static inline void tvsub(struct timeval *x,
						 struct timeval *y,
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

void probe_time_start(struct timestamp *ts)
{
    gettimeofday(&ts->start, NULL);
}

float probe_time_end(struct timestamp *ts)
{
    struct timeval tv;
    gettimeofday(&ts->end, NULL);
	tvsub(&ts->end, &ts->start, &tv);
	return (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
}

/*
 * check_status_ok
 * description:
 * Verifies status OK after each TensorFlow operation.
 * parameters:
 *     input status - the TensorFlow status
 *     input step   - a description of the last operation performed
 */
void check_status_ok(ncStatus_t status, char* step) {
    if (status != NC_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %d\n", step, status);
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

/*
 * check_result_ok
 * description:
 * Verifies result OK after decoding an image.
 * parameters:
 *     input result - the nanojpeg result
 *     input step   - a description of operation performed
 */
void check_result_ok(enum _nj_result result, char* step) {
    if (result != NJ_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, result);
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

/*
 * file_length
 * description:
 * Returns the length of the given file.
 * parameters:
 *     input file - any file
 */
unsigned long file_length(FILE* file) {
    fseek(file, 0, SEEK_END);
    unsigned long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    return length;
}

/*
 * load_file
 * description:
 * Loads a binary buffer from the given file
 * parameters:
 *     input file  - a binary file
 *     inut length - the length of the file
 */
char* load_file(FILE* file, unsigned long length) {
    char* buffer;
    buffer = (char *) malloc(length + 1);
    if (!buffer) {
        fprintf(stderr, "Memory error while reading buffer");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(buffer, length, 1, file);
    return buffer;
}

/*
 * 1. Initialize TensorFlow session
 * 2. Read in the previosly exported graph
 * 3. Read tensor from image
 * 4. Run the image through the model
 * 5. Print top labels
 * 6. Close session to release resources
 *
 * Note: the input image is not automatically resized. A jpg image is expected,
 * with the same dimensions of the images in the trained model.
 *
 * arguments:
 *     graph  - a file containing the TensorFlow graph
 *     labels - a file containing the list of labels
 *     image  - an image to test the model
 */
int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "3 arguments expected, %d received\n", argc - 1);
        exit(EXIT_FAILURE);
    }

    struct timestamp ts_gpu;
    float total_time = 0;

    char* input_graph = argv[1];
    char* input_labels = argv[2];
    char* input_image = argv[3];

    ncStatus_t retCode;
    struct ncDeviceHandle_t *deviceHandle;

    // 1. Initialize. Find and open device.
    //int loglevel = 2;
    //probe_time_start(&ts_gpu);
    //retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));
    //total_time += probe_time_end(&ts_gpu);
    //check_status_ok(retCode, "Setting Log Level");

    probe_time_start(&ts_gpu);
    retCode = ncDeviceCreate(0, &deviceHandle);
    int attemptCounter = 10000;
    while( retCode != NC_OK) {
        if (attemptCounter == 0)
            break;
        usleep(100);
        retCode = ncDeviceCreate(0, &deviceHandle);
        attemptCounter--;
    }
    total_time += probe_time_end(&ts_gpu);
    check_status_ok(retCode, "ncDeviceCreate.");
    
    probe_time_start(&ts_gpu);
    retCode = ncDeviceOpen(deviceHandle);
    attemptCounter = 10000;
    while( retCode != NC_OK) {
        if (attemptCounter == 0)
            break;
        usleep(100);
        retCode = ncDeviceOpen(deviceHandle);
        attemptCounter--;
    }
    total_time += probe_time_end(&ts_gpu);
    check_status_ok(retCode, "ncDeviceOpen");
    if (NULL == deviceHandle) {
        fprintf(stderr, "deviceHandle NULL");
        exit(EXIT_FAILURE);
    }

    // 2. Read in the previosly exported graph
    FILE* graphFile = fopen(input_graph, "rb");
    if (!graphFile) {
        fprintf(stderr, "Could not read graph file \"%s\"\n", input_graph);
        exit(EXIT_FAILURE);
    }

    unsigned long graphFileLength = file_length(graphFile);
    char* graphFileBuffer = load_file(graphFile, graphFileLength);
    struct ncGraphHandle_t *graphHandle;

    probe_time_start(&ts_gpu);
    retCode = ncGraphCreate("Inceptionv3-reshape", &graphHandle);
    total_time += probe_time_end(&ts_gpu);
    check_status_ok(retCode, "Create Graph Handle");
 
    struct ncFifoHandle_t* inFifoHandle;
    struct ncFifoHandle_t* outFifoHandle;
    probe_time_start(&ts_gpu);
    retCode = ncGraphAllocateWithFifos(deviceHandle, graphHandle, graphFileBuffer, graphFileLength, &inFifoHandle, &outFifoHandle);
    total_time += probe_time_end(&ts_gpu);
    check_status_ok(retCode, "Allocate Graph to Device with FIFOs.");

    // 3. Read tensor from image
    njInit(); 
    FILE* image_file = fopen(input_image, "rb");
    if (!image_file) {
        fprintf(stderr, "Could not read image file \"%s\"\n", input_image);
        exit(EXIT_FAILURE);
    }
    unsigned long image_file_length = file_length(image_file);
    char* image_file_buffer = load_file(image_file, image_file_length);
    nj_result_t result = njDecode(image_file_buffer, image_file_length);
    check_result_ok(result, "Loading of test image");
    unsigned char* old_image_data = njGetImage();
    int old_image_data_length = njGetImageSize();

    // convert uint8 to float
    unsigned int imageBufferLength = old_image_data_length * sizeof(float);
    float* imageBuffer = malloc(imageBufferLength);
    int i;
    for (i = 0; i < old_image_data_length; ++i) {
        imageBuffer[i] = (float)(old_image_data[i])/255.0;
    }

    // 4. Run the image through the model
    probe_time_start(&ts_gpu);
    retCode = ncGraphQueueInferenceWithFifoElem(graphHandle, inFifoHandle, outFifoHandle, imageBuffer, &imageBufferLength, NULL);
    check_status_ok(retCode, "Enqueue the image for processing.");

    unsigned int outFifoElemSize = 0; 
    unsigned int optionSize = sizeof(outFifoElemSize);
    retCode = ncFifoGetOption(outFifoHandle,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);
    check_status_ok(retCode, "Get Output Tensor Size.");

    // Get the output tensor
    float* resultData = (float*) malloc(outFifoElemSize);
    // We don't support userParam...
    //void* userParam;  // this will be set to point to the user-defined data that you passed into ncGraphQueueInferenceWithFifoElem() with this tensor
    retCode = ncFifoReadElem(outFifoHandle, (void*)resultData, &outFifoElemSize, NULL);
    check_status_ok(retCode, "Get Output from FIFO.");

    // Clean up the FIFOs
    ncFifoDestroy(&inFifoHandle);
    ncFifoDestroy(&outFifoHandle);
    
    // Clean up the graph
    ncGraphDestroy(&graphHandle);

    // Close and clean up the device
    ncDeviceClose(deviceHandle);
    ncDeviceDestroy(&deviceHandle);

    total_time += probe_time_end(&ts_gpu);
    fprintf(stdout, "Total time = %f\n", total_time);

    // 6. Close session to release resources
    fclose(graphFile);
    free(graphFileBuffer);
    free(imageBuffer);
    njDone(); // resets NanoJPEG's internal state and frees memory
    return EXIT_SUCCESS;
}
