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
#include <tensorflow/c/c_api.h>
#include "nanojpeg.c"

#include <sys/time.h>

#ifndef NUM_ITR
#define NUM_ITR 100
#endif

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

void deallocator(void* ptr, size_t len, void* arg) {
  free((void*) ptr);
}

/*
 * check_status_ok
 * description:
 * Verifies status OK after each TensorFlow operation.
 * parameters:
 *     input status - the TensorFlow status
 *     input step   - a description of the last operation performed
 */
void check_status_ok(TF_Status* status, char* step) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(status));
        fprintf(stderr, "Error message: %s\n", TF_Message(status));
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

    // int input_width = 299;
    // int input_height = 299;
    // int input_mean = 0;
    // int input_std = 255;

    // 1. Initialize TensorFlow session
    probe_time_start(&ts_gpu);
    TF_Graph* graph = TF_NewGraph();
//    printf("graph=0x%lx\n", (uintptr_t)graph);
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Status* status = TF_NewStatus();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    total_time += probe_time_end(&ts_gpu);

//    printf("session=0x%lx\n", (uintptr_t)session);
    if (session == NULL) return 0;

    check_status_ok(status, "Initialization of TensorFlow session");

    // 2. Read in the previosly exported graph
    FILE* pb_file = fopen(input_graph, "rb");
    if (!pb_file) {
        fprintf(stderr, "Could not read graph file \"%s\"\n", input_graph);
        exit(EXIT_FAILURE);
    }
    unsigned long pb_file_length = file_length(pb_file);
    char* pb_file_buffer = load_file(pb_file, pb_file_length);
    probe_time_start(&ts_gpu);
    TF_Buffer* graph_def = TF_NewBufferFromString(pb_file_buffer, pb_file_length);
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    total_time += probe_time_end(&ts_gpu);
    check_status_ok(status, "Loading of .pb graph");

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
    int image_data_length = old_image_data_length * sizeof(float);
    float* image_data = malloc(image_data_length);
    int i;
    for (i = 0; i < old_image_data_length; ++i) {
        image_data[i] = (float)(old_image_data[i])/255.0;
    }

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel].
    int ndims = 4, channels_in_image = 3;
    int64_t dims[] = {1, (long) njGetHeight(), (long) njGetWidth(), channels_in_image};
    fprintf(stdout, "image height %ld, width %ld, channels %ld\n", dims[1], dims[2], dims[3]);
    probe_time_start(&ts_gpu);
    TF_Tensor* tensor = TF_NewTensor(TF_FLOAT, dims, ndims, image_data, image_data_length, deallocator, NULL);

    fprintf(stdout, "pb file length %lu, image file length %ld, data length %d\n",
            pb_file_length, image_file_length, image_data_length);

    // 4. Run the image through the model
    TF_Output output1;
    output1.oper = TF_GraphOperationByName(graph, "input");
    output1.index = 0;
    TF_Output* inputs = {&output1};
    //TF_Tensor* const* input_values = {&tensor};
    TF_Tensor* const* input_values = &tensor;

    const TF_Operation* target_op = TF_GraphOperationByName(graph, "InceptionV3/Predictions/Reshape_1");
    TF_Output output2;
    output2.oper = (void *) target_op;
    output2.index = 0;
    TF_Output* outputs = {&output2};
    TF_Tensor* output_values;

    const TF_Operation* const* target_opers = {&target_op};

//    printf("session=0x%lx, inputs=[0x%lx], outputs=[0x%lx], op=[0x%lx], input_op=0x%lx, output_op=0x%lx\n",
//            (uintptr_t)session, (uintptr_t)input_values[0], (uintptr_t)output_values, (uintptr_t)target_opers[0],
//            (uintptr_t)inputs[0].oper, (uintptr_t)outputs[0].oper);

    int itr;
    for (itr = 0; itr < NUM_ITR; ++itr) {
        TF_SessionRun(
            session,
            NULL,
            inputs, input_values, 1,
            outputs, &output_values, 1,
            target_opers, 1,
            NULL,
            status
        );
    }
    check_status_ok(status, "Running of the image through the model");

    TF_DeleteSessionOptions(session_opts);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
    TF_DeleteTensor(output_values);
    TF_DeleteGraph(graph);
    total_time += probe_time_end(&ts_gpu);
    fprintf(stdout, "Total time = %f\n", total_time);

    // 6. Close session to release resources
    fclose(pb_file);
    free(pb_file_buffer);
//    free(image_data);
    njDone(); // resets NanoJPEG's internal state and frees memory
    return EXIT_SUCCESS;
}
