#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Utility function to preprocess image
cv::Mat preprocess(const cv::Mat& img) {
    cv::Mat processed;
    cv::resize(img, processed, cv::Size(640, 640));  // Resize to model input size
    processed.convertTo(processed, CV_32FC3, 1.0 / 255);  // Normalize to [0, 1]
    return processed;
}

// Utility function to postprocess and display the result
void postprocess(const cv::Mat& img, const std::vector<float>& output) {
    // This function would include your logic to convert the raw model output to a meaningful format
    // For now, we'll just display the input image for simplicity
    cv::imshow("Output", img);
    cv::waitKey(0);
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Create session and load model
    const char* model_path = "path_to_model.onnx";
    Ort::Session session(env, model_path, session_options);

    // Load an image
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    // Preprocess the image
    cv::Mat input_tensor = preprocess(img);

    // Prepare input container
    std::vector<int64_t> input_tensor_shape = {1, 3, 640, 640};
    std::vector<float> input_tensor_values(input_tensor.reshape(1, 640 * 640 * 3).begin<float>(), input_tensor.reshape(1, 640 * 640 * 3).end<float>());
    std::vector<const char*> input_node_names = {"input_node_name"};

    // Prepare output container
    std::vector<int64_t> output_tensor_shape = {1, 1, 640, 640};  // Adjust according to model output
    std::vector<float> output_tensor_values(640 * 640);
    std::vector<const char*> output_node_names = {"output_node_name"};

    // Run inference
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_values.data(), 1, output_node_names.data(), &output_tensor_values.data(), 1);

    // Postprocess and display the result
    postprocess(img, output_tensor_values);

    return 0;
}
