// NAME = VIJUL
// ROLL NUMBER = 2210994860

#include <stdio.h> // STANDARD INPUT-OUTPUT HEADER
#include <stdlib.h> // STANDARD LIBRARY HEADER
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h> // OPENCL HEADER
#include <chrono> // HEADER FOR TIMING UTILITIES

#define PRINT 1 // MACRO DEFINITION FOR PRINT CONTROL

int SZ = 100000000; // DEFAULT SIZE FOR ARRAYS

int *v1, *v2, *v_out; // POINTERS FOR INPUT AND OUTPUT ARRAYS

cl_mem bufV1, bufV2, bufV_out; // OPENCL MEMORY OBJECTS

cl_device_id device_id; // OPENCL DEVICE ID

cl_context context; // OPENCL CONTEXT

cl_program program; // OPENCL PROGRAM

cl_kernel kernel; // OPENCL KERNEL

cl_command_queue queue; // OPENCL COMMAND QUEUE

cl_event event = NULL; // OPENCL EVENT

int err; // ERROR VARIABLE FOR OPENCL FUNCTIONS

cl_device_id create_device(); // FUNCTION PROTOTYPE FOR CREATING OPENCL DEVICE

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname); // FUNCTION PROTOTYPE FOR SETTING UP OPENCL DEVICE, CONTEXT, COMMAND QUEUE, AND KERNEL

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename); // FUNCTION PROTOTYPE FOR BUILDING AN OPENCL PROGRAM

void setup_kernel_memory(); // FUNCTION PROTOTYPE FOR SETTING UP MEMORY BUFFERS FOR OPENCL KERNEL ARGUMENTS

void copy_kernel_args(); // FUNCTION PROTOTYPE FOR COPYING ARGUMENTS TO THE OPENCL KERNEL

void free_memory(); // FUNCTION PROTOTYPE FOR FREEING MEMORY ALLOCATED FOR OPENCL OBJECTS

void init(int *&A, int size); // FUNCTION PROTOTYPE FOR INITIALIZING AN ARRAY WITH RANDOM VALUES

void print(int *A, int size); // FUNCTION PROTOTYPE FOR PRINTING AN ARRAY

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        SZ = atoi(argv[1]); // SET ARRAY SIZE FROM COMMAND LINE ARGUMENT
    }

    init(v1, SZ); // INITIALIZE INPUT ARRAYS
    init(v2, SZ);
    init(v_out, SZ);

    size_t global[1] = {(size_t)SZ}; // DEFINE GLOBAL WORK SIZE FOR THE KERNEL

    print(v1, SZ); // PRINT INPUT ARRAYS
    print(v2, SZ);

    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl"); // SETUP OPENCL DEVICE, CONTEXT, COMMAND QUEUE, AND KERNEL
    setup_kernel_memory(); // SETUP MEMORY BUFFERS FOR KERNEL ARGUMENTS
    copy_kernel_args(); // COPY ARGUMENTS TO THE KERNEL
    auto start = std::chrono::high_resolution_clock::now(); // START TIMING KERNEL EXECUTION
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event); // ENQUEUE KERNEL FOR EXECUTION
    clWaitForEvents(1, &event); // WAIT FOR KERNEL EXECUTION TO FINISH

    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL); // READ RESULT BACK FROM DEVICE
    print(v_out, SZ); // PRINT OUTPUT ARRAY
    auto stop = std::chrono::high_resolution_clock::now(); // STOP TIMING KERNEL EXECUTION
    std::chrono::duration<double, std::milli> elapsed_time = stop - start; // CALCULATE ELAPSED TIME
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count()); // PRINT KERNEL EXECUTION TIME
    free_memory(); // FREE ALLOCATED MEMORY FOR OPENCL OBJECTS
}

// FUNCTION DEFINITION FOR INITIALIZING AN ARRAY WITH RANDOM VALUES
void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size); // ALLOCATE MEMORY FOR ARRAY

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // ASSIGN RANDOM VALUES TO ARRAY ELEMENTS
    }
}

// FUNCTION DEFINITION FOR PRINTING AN ARRAY
void print(int *A, int size)
{
    if (PRINT == 0) // CHECK IF PRINT CONTROL IS DISABLED
    {
        return;
    }

    if (PRINT == 1 && size > 15) // CHECK IF ARRAY SIZE IS LARGER THAN 15
    {
        for (long i = 0; i < 5; i++) // PRINT FIRST 5 ELEMENTS
        {
            printf("%d ", A[i]);
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++) // PRINT LAST 5 ELEMENTS
        {
            printf("%d ", A[i]);
        }
    }
    else
    {
        for (long i = 0; i < size; i++) // PRINT ALL ELEMENTS
        {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n"); // PRINT SEPARATOR
}

// FUNCTION DEFINITION FOR FREEING MEMORY ALLOCATED FOR OPENCL OBJECTS
void free_memory()
{
    clReleaseMemObject(bufV1); // RELEASE MEMORY OBJECTS
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    clReleaseKernel(kernel); // RELEASE KERNEL
    clReleaseCommandQueue(queue); // RELEASE COMMAND QUEUE
    clReleaseProgram(program); // RELEASE PROGRAM
    clReleaseContext(context); // RELEASE CONTEXT

    free(v1); // FREE INPUT ARRAYS
    free(v2);
    free(v_out);
}

// FUNCTION DEFINITION FOR COPYING ARGUMENTS TO THE OPENCL KERNEL
void copy_kernel_args()
{
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ); // SET KERNEL ARGUMENTS
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0) // CHECK FOR ERRORS
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

// FUNCTION DEFINITION FOR SETTING UP MEMORY BUFFERS FOR OPENCL KERNEL ARGUMENTS
void setup_kernel_memory()
{
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL); // CREATE MEMORY BUFFERS
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL); // WRITE INPUT ARRAYS TO MEMORY BUFFERS
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// FUNCTION DEFINITION FOR SETTING UP OPENCL DEVICE, CONTEXT, COMMAND QUEUE, AND KERNEL
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    // CREATE COMMAND QUEUE USING clCreateCommandQueue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    }

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    }
}

// FUNCTION DEFINITION FOR BUILDING AN OPENCL PROGRAM
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r"); // OPEN OPENCL PROGRAM FILE
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1); // ALLOCATE MEMORY FOR PROGRAM BUFFER
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle); // READ PROGRAM CONTENT INTO BUFFER
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err); // CREATE PROGRAM FROM SOURCE
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); // BUILD PROGRAM
    if (err < 0)
    {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1); // ALLOCATE MEMORY FOR PROGRAM LOG
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL); // GET PROGRAM BUILD LOG
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

// FUNCTION DEFINITION FOR CREATING AN OPENCL DEVICE
cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL); // GET PLATFORM ID
    if (err < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL); // GET GPU DEVICE ID
    if (err == CL_DEVICE_NOT_FOUND) // CHECK IF GPU NOT FOUND
    {
        printf("GPU not found\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL); // GET CPU DEVICE ID
    }
    if (err < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}

