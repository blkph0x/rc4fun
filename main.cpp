#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <stdexcept>
#include <chrono>

// OpenCL kernel for RC4 decryption
const char* kernel_code = R"(
__kernel void rc4_decrypt(__global const uchar *encrypted_data,
                          __global uchar *decrypted_data,
                          __global const uchar *keys,
                          const int key_length,
                          const int data_length) {
    int gid = get_global_id(0);
    if (gid < data_length) {
        int i = 0, j = 0;
        uchar S[256];
        for (int k = 0; k < 256; k++) {
            S[k] = k;
        }
        for (int k = 0; k < 256; k++) {
            j = (j + S[k] + keys[k % key_length]) % 256;
            uchar temp = S[k];
            S[k] = S[j];
            S[j] = temp;
        }
        i = j = 0;
        for (int n = 0; n < gid; n++) {
            i = (i + 1) % 256;
            j = (j + S[i]) % 256;
            uchar temp = S[i];
            S[i] = S[j];
            S[j] = temp;
        }
        decrypted_data[gid] = encrypted_data[gid] ^ S[(S[i] + S[j]) % 256];
    }
}
)";

bool is_valid_plaintext(const std::vector<unsigned char>& data) {
    // Enhanced validation: check for printable characters, spaces, and common punctuation
    return std::all_of(data.begin(), data.end(), [](unsigned char c) {
        return isprint(c) || isspace(c);
        });
}

std::vector<unsigned char> brute_force_rc4_gpu(const std::vector<unsigned char>& encrypted_data, const std::string& charset, int max_key_length) {
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platform_id;
    cl_uint num_devices;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform IDs. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL initialization error");
    }

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL device IDs. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL initialization error");
    }

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL context creation error");
    }

    queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL command queue creation error");
    }

    program = clCreateProgramWithSource(context, 1, &kernel_code, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL program creation error");
    }

    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        throw std::runtime_error("OpenCL program build error");
    }

    kernel = clCreateKernel(program, "rc4_decrypt", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL kernel creation error");
    }

    size_t data_length = encrypted_data.size();
    std::vector<unsigned char> decrypted_data(data_length);

    cl_mem encrypted_data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_length, (void*)encrypted_data.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create encrypted data buffer. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL buffer creation error");
    }

    cl_mem decrypted_data_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_length, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create decrypted data buffer. Error code: " << err << std::endl;
        throw std::runtime_error("OpenCL buffer creation error");
    }

    // Measure performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int key_length = 1; key_length <= max_key_length; ++key_length) {
        std::vector<unsigned char> keys(key_length);
        std::string key_str(key_length, charset[0]);

        do {
            std::copy(key_str.begin(), key_str.end(), keys.begin());

            cl_mem keys_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, keys.size(), keys.data(), &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create keys buffer. Error code: " << err << std::endl;
                throw std::runtime_error("OpenCL buffer creation error");
            }

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &encrypted_data_buffer);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &decrypted_data_buffer);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &keys_buffer);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &key_length);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &data_length);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to set OpenCL kernel arguments. Error code: " << err << std::endl;
                throw std::runtime_error("OpenCL kernel argument setting error");
            }

            size_t global_work_size = data_length;
            err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to enqueue OpenCL kernel. Error code: " << err << std::endl;
                throw std::runtime_error("OpenCL kernel enqueue error");
            }

            err = clEnqueueReadBuffer(queue, decrypted_data_buffer, CL_TRUE, 0, data_length, decrypted_data.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to read buffer from OpenCL kernel. Error code: " << err << std::endl;
                throw std::runtime_error("OpenCL buffer read error");
            }

            if (is_valid_plaintext(decrypted_data)) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;
                std::cout << "Decryption successful, key found: " << key_str << std::endl;
                std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
                return decrypted_data;
            }

            clReleaseMemObject(keys_buffer);
        } while (std::next_permutation(key_str.begin(), key_str.end()));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "No valid key found" << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    return {};
}

int main() {
    try {
        std::ifstream input_file("encrypted_file.bin", std::ios::binary);
        if (!input_file) {
            std::cerr << "Failed to open input file" << std::endl;
            throw std::runtime_error("File open error");
        }

        std::vector<unsigned char> encrypted_data((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
        input_file.close();

        // Adjust charset and max_key_length based on your specific requirements for P1 and DMR
        std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        int max_key_length = 5;

        std::vector<unsigned char> decrypted_data = brute_force_rc4_gpu(encrypted_data, charset, max_key_length);

        if (!decrypted_data.empty()) {
            std::ofstream output_file("decrypted_file.bin", std::ios::binary);
            if (!output_file) {
                std::cerr << "Failed to open output file" << std::endl;
                throw std::runtime_error("File open error");
            }

            output_file.write(reinterpret_cast<char*>(decrypted_data.data()), decrypted_data.size());
            output_file.close();
            std::cout << "Decryption successful, output written to decrypted_file.bin" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

