#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

size_t min_size_t(size_t a, size_t b) {
  return a < b ? a : b;
}

void opencl_merge_sort(const size_t array_size, const int* array, int* array_out) {
  size_t source_max_size = 20000;
  char* source_code = malloc(sizeof(char) * source_max_size);
  FILE* file = fopen("./merge.cl", "r");
  assert(file != NULL);

  size_t source_size = fread(source_code, sizeof(char), 20000, file);
  if (feof == 0) {
    printf("Too large of a kernel\n");
    exit(1);
  }

  fclose(file);

  cl_int ret;
  /* For easier hardware watchpoints */
  #ifndef NDEBUG
  ret = CL_SUCCESS;
  #endif

  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint platform_count = 0;
  cl_uint device_count = 0;

  ret = clGetPlatformIDs(1, &platform_id, &platform_count);
  if (platform_count > 1) {
    printf("There is more than one platform, but this code can only handle one.\n");
  }

  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &device_count);
  if (platform_count > 1) {
    printf("There is more than one device, but this code can only handle one.\n");
  }

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  /* TODO check if int is the same in openCL as the host device */
  /* Allocates both the input, and output arrays in one chunk as they will be
     swapping anyway */
  cl_mem mem_in_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(int) * array_size, NULL, &ret);

  /* input as the beginning */
  ret = clEnqueueWriteBuffer(command_queue, mem_in_out, CL_TRUE, 0, sizeof(int) * array_size, array, 0, NULL, NULL);

  cl_program program = clCreateProgramWithSource(context, 1, (const char**) &source_code,
     (const size_t*) &source_size, &ret);

  ret = clBuildProgram(program, 1, &device_id, "-D num_type=int", NULL, NULL);

  if (ret != CL_SUCCESS) {
    char buffer[2048];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
    printf("Build failed with the following message:\n %s\n", buffer);
    exit(1);
  }

  cl_kernel kernel_one_sort = clCreateKernel(program, "merge_sort_one", &ret);
  cl_kernel kernel_two_sort = clCreateKernel(program, "merge_sort_two", &ret);

  ret = clSetKernelArg(kernel_one_sort, 0, sizeof(cl_mem), (void*) &mem_in_out);
  ret = clSetKernelArg(kernel_two_sort, 0, sizeof(cl_mem), (void*) &mem_in_out);

  size_t work_size_one = array_size / 2;
  size_t work_size_two = array_size / 4;
  cl_event event_one;
  cl_event event;
  ret = clEnqueueNDRangeKernel(command_queue, kernel_one_sort, 1, NULL, &work_size_one, NULL, 0, NULL, &event_one);
  ret = clEnqueueNDRangeKernel(command_queue, kernel_two_sort, 1, NULL, &work_size_two, NULL, 1, &event_one, &event);

  bool mem_flipped = false;
  unsigned int work_piece = 4;
  size_t work_size = array_size / 2 / work_piece;
  size_t local_work_size = min_size_t(4, work_size);
  cl_mem mem_in;
  cl_mem mem_out;
  while (work_size > 0) {
    cl_event tmp_event;

    cl_kernel kernel_sort = clCreateKernel(program, "merge_sort_simple_wrapper", &ret);
    ret = clSetKernelArg(kernel_sort, 0, sizeof(cl_mem), (void*) &mem_in_out);
    ret = clSetKernelArg(kernel_sort, 1, sizeof(unsigned int), &work_piece);
    cl_char mem_flipped_char = mem_flipped ? 1 : 0;
    ret = clSetKernelArg(kernel_sort, 2, sizeof(cl_char), (void*) &mem_flipped_char);

    ret = clEnqueueNDRangeKernel(command_queue, kernel_sort, 1, NULL, &work_size, &local_work_size, 1, &event, &tmp_event);
    event = tmp_event;

    clReleaseKernel(kernel_sort);

    printf("Work piece size %u queued\n", work_piece);
    work_piece *= 2;
    work_size = array_size / 2 / work_piece;
    local_work_size = min_size_t(4, work_size);
    mem_flipped = !mem_flipped;
  }

  size_t offset;
  if (mem_flipped) {
    offset = sizeof(int) * array_size;
  } else {
    offset = 0;
  }
  clEnqueueReadBuffer(command_queue, mem_in_out, CL_TRUE, offset, sizeof(int) * array_size, array_out, 0, NULL, NULL);

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel_one_sort);
  ret = clReleaseKernel(kernel_two_sort);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(mem_in_out);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(source_code);
}

int main(int argc, char** argv) {
  printf("Calling openCL magic :D\n");
  const size_t numbers_size = 8 * 1024 * 1024;
  int* numbers = malloc(numbers_size * sizeof(int));
  int* numbers_out = malloc(numbers_size * sizeof(int));
  assert(numbers != NULL);
  assert(numbers_out != NULL);

  for (int i = 0; i < numbers_size; ++i) {
    numbers[i] = random();
  }

  printf("Starting openCL section\n");
  struct timespec start;
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  opencl_merge_sort(numbers_size, numbers, numbers_out);

  clock_gettime(CLOCK_MONOTONIC, &end);
  long long diff_exact = 1000000000 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  double diff = diff_exact / 1000000000.0;
  printf("Time to run %f seconds\n", diff);

  /*printf("Input\n");
  for (int i = 0; i < numbers_size / 4; ++i) {
    printf("%d %d %d %d\n",
      numbers[4*i + 0],
      numbers[4*i + 1],
      numbers[4*i + 2],
      numbers[4*i + 3]);
  }

  printf("Output\n");
  for (int i = 0; i < numbers_size / 4; ++i) {
    printf("%d %d %d %d\n",
      numbers_out[4*i + 0],
      numbers_out[4*i + 1],
      numbers_out[4*i + 2],
      numbers_out[4*i + 3]);
  }*/
  /*for (int i = 0; i < numbers_size - 1; ++i) {
    if (numbers_out[i] > numbers_out[i + 1]) {
      printf("The order is incorrect :( %d %d\n", numbers_out[i], numbers_out[i + 1]);
      return 1;
    }
  }*/

  return 0;
}
