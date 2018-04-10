#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <ctime>
#include<fstream>

using namespace std;
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"__kernel void prefixSum(__global int *output, __global int *input, const int length) { \n" \
"    int tid = get_global_id(0);                              \n" \
"   for(int i=0;i<=tid;i++) output[tid]+=input[i];           \n" \
"}\n";

const char *kernelSource1 =                                     "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"__kernel void findMax(__global int *max_array, __global int *rj, const int length,const int block) {    \n" \
"int global_index = get_global_id(0) * block;\n" \
"int max = -32767;\n" \
"int upper_bound = (get_global_id(0) + 1) * block;\n" \
"if (upper_bound > length) upper_bound = length;\n" \
"while (global_index < upper_bound) {\n" \
"    int element = rj[global_index];\n" \
"    max = (max > element) ? max : element;\n" \
"    global_index++;\n" \
"}\n" \
"max_array[0] = max;\n" \
"}\n" ;


class IntervalGraph {
    private:
    int *component;
    int *cliquesize;
    int componentSize;
public:
    IntervalGraph(char *filename) {
        int maxNode, maxRow;
        ifstream inFile(filename);
        //cout << "filename " << filename << endl;
        
        if (!inFile) {
            cout << endl << "Failed to open file " << filename;
            exit(0);
        }
        inFile >> maxNode;
        inFile >> maxRow;
        
        setcomponentSize(maxNode);
        component = (int*)malloc(maxNode*sizeof(int));
        //cout<<"\nmax node "<<componentSize;
        int n = 0;
        while (!inFile.eof()) {
            inFile >> n;
            component[n-1] = 1;
            inFile >> n;
            component[n-1] = -1;
        }
        //printComponent();
    }
    void setcomponentSize(int maxNode){
        componentSize = maxNode;
    }
    
    void printComponent(){
        cout << "Component \n";
        for (int i = 0; i < componentSize; i++)
        cout << component[i] << "  ";
        
    }
    int findMaxCliqueSize() {
        return calculate_rjSum();
    }
    int calculate_rjSum() {                             /*Calculate Prefix Sum*/
        int *host_rj;
        cl::Buffer device_components;
        cl::Buffer device_rj;
        cl::Buffer device_block;
        host_rj = new int[componentSize];
        size_t bytes = componentSize*sizeof(int);
        cl_int err = CL_SUCCESS;
        try {
            
            // Query platforms
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.size() == 0) {
                std::cout << "Platform size 0\n";
                return -1;
            }
            
            // Get list of devices on default platform and create context
            cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
            cl::Context context(CL_DEVICE_TYPE_GPU, properties);
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            
            // Create command queue for first device
            cl::CommandQueue queue(context, devices[0], 0, &err);
            
            // Create device memory buffers
            device_components = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
            device_rj = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);
            device_block = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);
            
            // Bind memory buffers
            queue.enqueueWriteBuffer(device_components, CL_TRUE, 0, bytes, component);
            //queue.enqueueWriteBuffer(device_block, CL_TRUE, 0, bytes, component);
            
            
            //Build kernel from source string
            cl::Program::Sources source(1,
                                        std::make_pair(kernelSource,strlen(kernelSource)));
            
            cl::Program program_ = cl::Program(context, source);
            
            program_.build(devices);
            //program_.build(devices);
            
            // Create kernel object
            cl::Kernel kernel(program_, "prefixSum", &err);
            
            // Bind kernel arguments to kernel
            kernel.setArg(0, device_rj);
            kernel.setArg(1, device_components);
            //kernel.setArg(2, device_block);
            kernel.setArg(2, componentSize);
            
            // Number of work items in each local work group
            cl::NDRange localSize(componentSize);
            // Number of total work items - localSize must be devisor
            cl::NDRange globalSize(componentSize);
            
            // Enqueue kernel
            cl::Event event;
            queue.enqueueNDRangeKernel(
                                       kernel,
                                       cl::NullRange,
                                       globalSize,
                                       localSize,
                                       NULL,
                                       &event);
            
            // Block until kernel completion
            event.wait();
            
            // Read back d_c
            queue.enqueueReadBuffer(device_rj, CL_TRUE, 0, bytes, host_rj);
            
        }
        catch (cl::Error err) {
            std::cerr
            << "ERROR: "<<err.what()<<"("<<err.err()<<")"<<std::endl;
        }
        cout<<"\n";
        for(int i=0;i<componentSize;i++) {
            cout<< host_rj[i]<<" ";
        }
        int w = calculateMaxCliqueSize(host_rj);
        return w;
        
    }
    int calculateMaxCliqueSize(int *rj) {           /*Calculate Max Clique Size*/
        int *w;
        int block=64;                           //Size of the block
        
        cl::Buffer device_max;
        cl::Buffer device_rj;
        cl::Buffer device_block;
        w = new int[componentSize];
        size_t bytes = componentSize*sizeof(int);
        cl_int err = CL_SUCCESS;
        try {
            
            // Query platforms
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.size() == 0) {
                std::cout << "Platform size 0\n";
                return -1;
            }
            
            // Get list of devices on default platform and create context
            cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
            cl::Context context(CL_DEVICE_TYPE_GPU, properties);
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            
            // Create command queue for first device
            cl::CommandQueue queue(context, devices[0], 0, &err);
            
            // Create device memory buffers
            device_rj = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
            device_max = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);
            //device_block = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);
            
            // Bind memory buffers
            queue.enqueueWriteBuffer(device_rj, CL_TRUE, 0, bytes, rj);
            //queue.enqueueWriteBuffer(device_block, CL_TRUE, 0, bytes, component);
            
            
            //Build kernel from source string
            cl::Program::Sources source(1,
                                        std::make_pair(kernelSource1,strlen(kernelSource1)));
            
            cl::Program program_ = cl::Program(context, source);
            
            program_.build(devices);
            //program_.build(devices);
            
            // Create kernel object
            cl::Kernel kernel(program_, "findMax", &err);
            
            // Bind kernel arguments to kernel
            kernel.setArg(0, device_max);
            kernel.setArg(1, device_rj);
            //kernel.setArg(2, device_block);
            kernel.setArg(2, componentSize);
            kernel.setArg(3, block);
            // Number of work items in each local work group
            cl::NDRange localSize(1);
            // Number of total work items - localSize must be devisor
            cl::NDRange globalSize(componentSize);
            
            // Enqueue kernel
            cl::Event event;
            queue.enqueueNDRangeKernel(
                                       kernel,
                                       cl::NullRange,
                                       globalSize,
                                       localSize,
                                       NULL,
                                       &event);
            
            // Block until kernel completion
            event.wait();
            
            // Read back d_c
            queue.enqueueReadBuffer(device_max, CL_TRUE, 0, bytes, w);
            
        }
        catch (cl::Error err) {
            std::cerr
            << "ERROR: "<<err.what()<<"("<<err.err()<<")"<<std::endl;
        }
        //cout<<"\n\n\nhello "<<w[0]<<"\n";
        return w[0];
    }
    int calculate_maxclique_seq() {
        int *partial_sum;
        partial_sum = new int[componentSize];
        for(int i=0;i<componentSize;i++) {
            partial_sum[i]=0;
            for(int j=0;j<=i;j++) {
                partial_sum[i]+=component[j];
            }
            cout<<" "<<partial_sum[i];
        }
        return calculate_max_seq(partial_sum);
    }
    int calculate_max_seq(int * partial_sum) {
        int max=-32767;
        for(int i=0;i<componentSize;i++) {
            if(max<partial_sum[i]) {
                max=partial_sum[i];
            }
        }
        return max;
    }
    ~IntervalGraph(){
        
    }
};
int main(int argc, char *argv[])
{
    /*Parallel max clique*/
    int start_s=clock();
    IntervalGraph ig("/Users/Arthi/Downloads/intervaldata.txt");
    int maxCliqueSize = ig.calculate_rjSum();
    cout<<"\nMax Clique Size "<<maxCliqueSize<<"\n";
    int stop_s=clock();
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " milliseconds"<<endl;
    
    /*Sequential max clique*/
    start_s = clock();
    int maxCliqueSeq = ig.calculate_maxclique_seq();
    cout<<"\n Max Clique sequential "<<maxCliqueSeq;
    stop_s = clock();
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " milliseconds"<<endl;
    return 0;
}