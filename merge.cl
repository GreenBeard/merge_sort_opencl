#ifndef num_type
#error num_type must be defined
#endif

/* Both inputs have size of 1 */
kernel void merge_sort_one(global num_type* array) {
  size_t offset = 2 * get_global_id(0);
  if (array[offset] > array[offset + 1]) {
    num_type tmp = array[offset];
    array[offset] = array[offset + 1];
    array[offset + 1] = tmp;
  }
}


/* both inputs have size of 2
   output is size 4 */
kernel void merge_sort_two(global num_type* array) {
  local num_type remaining[2];
  size_t offset = 4 * get_global_id(0);
  global num_type* input_a = array + offset;
  global num_type* input_b = array + offset + 2;
  global num_type* output = array + offset;
  if (input_a[0] < input_b[0]) {
    remaining[0] = input_b[0];
    output[0] = input_a[0];
  } else {
    remaining[0] = input_a[0];
    output[0] = input_b[0];
  }
  if (input_a[1] > input_b[1]) {
    remaining[1] = input_b[1];
    output[3] = input_a[1];
  } else {
    remaining[1] = input_a[1];
    output[3] = input_b[1];
  }
  if (remaining[0] < remaining[1]) {
    output[1] = remaining[0];
    output[2] = remaining[1];
  } else {
    output[1] = remaining[1];
    output[2] = remaining[0];
  }
}

/* A feeble attempt of taking two merge sort inputs and sort both the lefts,
 * and both the rights so that we know the bottom fourth and top fourth then to
 * sort the remaining middle. This is a similar algorithm to some of the
 * division algorithms which take two steps and divide them into 3 which are
 * easier (in this case more parallelizable). Only likely useful for the latter
 * steps in sorting when the GPU isn't fully utilized.
*/
/*kernel void merge_sort_even(global const int* input_a, global const int* input_b,
    global const int* output, size_t size_in_each) {
  void (^merge_child)(void) = ^{input_a, input_b, output, size_in_each};
  enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(size_in_each, size_in_each/2), merge_child);
  enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(size_in_each, size_in_each/2), merge_child);
}*/

/* size_t is not allowed as a parameter which is why unsigned int is used */
void merge_sort_simple(global const num_type* array_in, global num_type* array_out, unsigned int size_in_each) {
  size_t offset = 2 * size_in_each * get_global_id(0);
  const global num_type* input_a = array_in + offset;
  const global num_type* input_b = array_in + offset + size_in_each;
  global num_type* output = array_out + offset;

  unsigned int index_a = 0;
  unsigned int index_b = 0;
  unsigned int index_out = 0;
  while (index_a < size_in_each && index_b < size_in_each) {
    if (input_a[index_a] < input_b[index_b]) {
      output[index_out] = input_a[index_a];
      ++index_a;
    } else {
      output[index_out] = input_b[index_b];
      ++index_b;
    }
    ++index_out;
  }

  /* Only one of the two following will be true */
  if (index_a == size_in_each) {
    while (index_b < size_in_each) {
      output[index_out] = input_b[index_b];
      ++index_b;
      ++index_out;
    }
  } else if (index_b == size_in_each) {
    while (index_a < size_in_each) {
      output[index_out] = input_a[index_a];
      ++index_a;
      ++index_out;
    }
  }
}

kernel void merge_sort_simple_wrapper(global const num_type* array, unsigned int size_in_each, char out_first) {
  global const num_type* array_in;
  global const num_type* array_out;
  if (out_first) {
    array_out = array;
    array_in  = array + 2 * size_in_each * get_global_size(0);
  } else {
    array_in  = array;
    array_out = array + 2 * size_in_each * get_global_size(0);
  }
  merge_sort_simple(array_in, array_out, size_in_each);
}
