#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>  // For reading and writing images
#include <vector>
#include <string>
#include <thread>
#include <dirent.h> // For directory handling on Unix-like systems
#include <sys/types.h> // For DIR, dirent types
#include <regex>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }

const int MAX_CONCURRENT_STREAMS = 4; // Set based on your GPU capabilities

// CUDA kernel for RGB to Grayscale conversion
__global__ void rgbToGrayKernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgbIndex = (y * width + x) * 3;
        int grayIndex = y * width + x;
        unsigned char r = rgb[rgbIndex];
        unsigned char g = rgb[rgbIndex + 1];
        unsigned char b = rgb[rgbIndex + 2];
        gray[grayIndex] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height, const float* filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int halfFilterWidth = filterWidth / 2;

        for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
            for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sum += input[iy * width + ix] * filter[(ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth)];
            }
        }

        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// CUDA kernel for Sobel edge detection
__global__ void sobelKernel(const unsigned char* input, float* gradient, float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float Gx = 0.0f;
        float Gy = 0.0f;

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            Gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
                 + input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

            Gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                 + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
        }

        gradient[y * width + x] = sqrtf(Gx * Gx + Gy * Gy);
        direction[y * width + x] = atan2f(Gy, Gx);
    }
}

// CUDA kernel for non-maximum suppression
__global__ void nonMaximumSuppressionKernel(const float* gradient, const float* direction, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float angle = direction[y * width + x] * 180.0f / M_PI;
        angle = angle < 0 ? angle + 180 : angle;

        float g = gradient[y * width + x];
        float g1 = 0.0f, g2 = 0.0f;

        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
            g1 = gradient[y * width + (x - 1)];
            g2 = gradient[y * width + (x + 1)];
        } else if (angle >= 22.5 && angle < 67.5) {
            g1 = gradient[(y - 1) * width + (x + 1)];
            g2 = gradient[(y + 1) * width + (x - 1)];
        } else if (angle >= 67.5 && angle < 112.5) {
            g1 = gradient[(y - 1) * width + x];
            g2 = gradient[(y + 1) * width + x];
        } else if (angle >= 112.5 && angle < 157.5) {
            g1 = gradient[(y - 1) * width + (x - 1)];
            g2 = gradient[(y + 1) * width + (x + 1)];
        }

        if (g >= g1 && g >= g2) {
            output[y * width + x] = static_cast<unsigned char>(g);
        } else {
            output[y * width + x] = 0;
        }
    }
}

// CUDA kernel for double threshold and edge tracking by hysteresis
__global__ void doubleThresholdKernel(unsigned char* image, int width, int height, unsigned char lowThreshold, unsigned char highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char pixel = image[y * width + x];
        if (pixel >= highThreshold) {
            image[y * width + x] = 255;
        } else if (pixel < lowThreshold) {
            image[y * width + x] = 0;
        } else {
            bool connectedToStrongEdge = false;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    if (kx == 0 && ky == 0) continue;
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        if (image[iy * width + ix] == 255) {
                            connectedToStrongEdge = true;
                            break;
                        }
                    }
                }
                if (connectedToStrongEdge) break;
            }
            image[y * width + x] = connectedToStrongEdge ? 255 : 0;
        }
    }
}

// Function to process each image in its own CUDA stream
void processImage(const cv::Mat& inputImage, const std::string& outputFileName, cudaStream_t stream) {
    if (inputImage.channels() != 3) {
        std::cerr << "Image must be RGB." << std::endl;
        return;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int stepRGB = width * 3 * sizeof(unsigned char);
    int stepGray = width * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_rgb, *d_gray, *d_blurred, *d_output;
    float *d_gradient, *d_direction;
    CUDA_CHECK(cudaMalloc(&d_rgb, stepRGB * height));
    CUDA_CHECK(cudaMalloc(&d_gray, stepGray * height));
    CUDA_CHECK(cudaMalloc(&d_blurred, stepGray * height));
    CUDA_CHECK(cudaMalloc(&d_output, stepGray * height));
    CUDA_CHECK(cudaMalloc(&d_gradient, stepGray * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_direction, stepGray * height * sizeof(float)));

    // Copy the input image to the GPU (non-blocking, with a stream)
    CUDA_CHECK(cudaMemcpyAsync(d_rgb, inputImage.data, stepRGB * height, cudaMemcpyHostToDevice, stream));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Convert RGB to Grayscale
    rgbToGrayKernel<<<gridSize, blockSize, 0, stream>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Apply Gaussian blur
    float h_filter[3 * 3] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    float *d_filter;
    CUDA_CHECK(cudaMalloc(&d_filter, 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

    gaussianBlurKernel<<<gridSize, blockSize, 0, stream>>>(d_gray, d_blurred, width, height, d_filter, 3);
    CUDA_CHECK(cudaGetLastError());

    // Perform Sobel edge detection
    sobelKernel<<<gridSize, blockSize, 0, stream>>>(d_blurred, d_gradient, d_direction, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Non-maximum suppression
    nonMaximumSuppressionKernel<<<gridSize, blockSize, 0, stream>>>(d_gradient, d_direction, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Double threshold
    unsigned char lowThreshold = 50;
    unsigned char highThreshold = 150;
    doubleThresholdKernel<<<gridSize, blockSize, 0, stream>>>(d_output, width, height, lowThreshold, highThreshold);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    cv::Mat outputImage(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpyAsync(outputImage.data, d_output, stepGray * height, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Write output image
    cv::imwrite(outputFileName, outputImage);

    // Clean up
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_output);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    cudaFree(d_filter);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>" << std::endl;
        return 1;
    }

    std::string inputDirectory = argv[1];
    std::string outputDirectory = argv[2];

    // Open the input directory
    DIR* dir = opendir(inputDirectory.c_str());
    if (!dir) {
        std::cerr << "Error opening directory: " << inputDirectory << std::endl;
        return 1;
    }

    struct dirent* entry;
    std::vector<std::string> imageFiles;

    const std::regex pattern("[^\\s]+(.*?)\\.(jpg|jpeg|JPG|JPEG)$");
    // Read all files in the directory
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // Regular file
            std::string fileName = entry->d_name;
            // Assuming images are .jpg files, you might want to add more conditions to check for valid images
            if (regex_match(fileName, pattern)) {
                imageFiles.push_back(inputDirectory + "/" + fileName);
            }
        }
    }
    closedir(dir);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(MAX_CONCURRENT_STREAMS);
    for (int i = 0; i < MAX_CONCURRENT_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Process images in parallel using CUDA streams
    size_t imageCount = imageFiles.size();
    for (size_t i = 0; i < imageCount; ++i) {
        std::string outputFileName = outputDirectory + "/" + imageFiles[i].substr(imageFiles[i].find_last_of('/') + 1);
        processImage(cv::imread(imageFiles[i]), outputFileName, streams[i % MAX_CONCURRENT_STREAMS]);
    }

    // Cleanup
    for (int i = 0; i < MAX_CONCURRENT_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
