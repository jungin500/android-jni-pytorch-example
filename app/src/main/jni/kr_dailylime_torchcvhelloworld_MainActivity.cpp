#include <iostream>
#include <string>
#include <memory>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <jni.h>
#include "kr_dailylime_torchcvhelloworld_MainActivity.h"

// Log into Android
#include <android/log.h>
#define ALOGI(...)                                                             \
  __android_log_print(ANDROID_LOG_INFO, "NativeModule", __VA_ARGS__)
#define ALOGE(...)                                                             \
  __android_log_print(ANDROID_LOG_ERROR, "NativeModule", __VA_ARGS__)

void logi(std::string body) {
    const char* m = body.c_str();
    ALOGI("%s", m);
}

template <typename T> void logi(std::string body, T t) {
    const char* m = body.c_str();
    std::ostringstream os;
    os << t << std::endl;
    ALOGI("%s %s", m, os.str().c_str());
}

void loge(std::string body) {
    const char* m = body.c_str();
    ALOGE("%s", m);
}

template <typename T> void loge(std::string body, T t) {
    const char* m = body.c_str();
    std::ostringstream os;
    os << t << std::endl;
    ALOGE("%s %s", m, os.str().c_str());
}

// PyTorch model-related stuffs
class TorchModelHandler {
private:
    torch::jit::script::Module module;
    std::vector<std::string> labels;

public:
    std::vector<float> fpses;

    TorchModelHandler(std::string model_path, std::string label_path) : module(), labels() {
        // Load JIT module
        try {
            this->module = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            loge("Error loading model");
        }

        // Load label
        std::ifstream label_file;
        label_file.open(label_path);
        if (label_file) {
            // Only write on file exists
            while (!label_file.eof()) {
                std::string line;
                std::getline(label_file, line);
                this->labels.push_back(line);
            }
        }
        label_file.close();

        logi("Loaded torchscript model " + model_path);
        logi("Loaded " + std::to_string(this->labels.size()) + " labels");
    }

    // Infer given OpenCV mat image and return Top-1 class name
    std::string infer(const cv::Mat &input_image) {
        std::vector<torch::jit::IValue> inputs;

        // Resize input image for inference
        cv::Mat tensor_input_mat;
        cv::resize(input_image, tensor_input_mat, cv::Size(224, 224));

        // Convert color scheme
        // cv::cvtColor(tensor_input_mat, tensor_input_mat, cv::COLOR_BGR2RGB);
        // -> already a RGB!
        
        // Convert to Tensor
        torch::Tensor input_tensor = torch::from_blob(tensor_input_mat.data, {tensor_input_mat.rows, tensor_input_mat.cols, 3}, torch::kByte);

        // Transpose tensor (HWC to CHW) and change dtype
        input_tensor = input_tensor.permute({2, 0, 1});
        input_tensor = input_tensor.toType(torch::kFloat);
        input_tensor.div_(255.);

        // Normalize with given ImageNet mean and std
        const float mean[3] = {0.485, 0.456, 0.406};
        const float std[3] = {0.229, 0.224, 0.225};
        for(int ch=0; ch < 3; ch++)
            input_tensor[ch].sub_(mean[ch]).div_(std[ch]);

        // Unsqueeze first(batch) dimension
        // [3, 224, 224] -> [1, 3, 224, 224]
        input_tensor.unsqueeze_(0);

        // Add input to inputs list
        inputs.push_back(input_tensor);

        // Forward inputs and get Tensor
        torch::Tensor output = this->module.forward(inputs).toTensor();

        // Return outputs with classes with maximum value regarding second dimension
        output.squeeze_(0);
        // torch::Tensor result = output.softmax(0);
        torch::Tensor result = output.sigmoid();
        result = output.argmax(0);
        // int result_class_id = (int)(result[0].item<float>());
        int result_class_id = result.item<int>();

        // Get class name with given result class id
        std::string class_name = "UNKNOWN";
        if(this->labels.size() > 0)
            class_name = this->labels[result_class_id];

        return "Class id: " + std::to_string(result_class_id) + " (" + class_name + ")";
    }
};

TorchModelHandler* TorchModelHandlerInstance = NULL;

// Export to JNI interface
extern "C"
JNIEXPORT void JNICALL
Java_kr_dailylime_torchcvhelloworld_MainActivity_ConvertRGBtoGray(
        JNIEnv *env,
        jobject  instance,
        jlong matAddrInput,
        jlong matAddrResult) {

    // OpenCV convert color format
    cv::Mat &matInput = *(cv::Mat *)matAddrInput;
    cv::Mat &matResult = *(cv::Mat *)matAddrResult;
    cv::cvtColor(matInput, matResult, cv::COLOR_RGBA2GRAY);

    // Load torchscript module
    torch::jit::script::Module module;

}

extern "C"
JNIEXPORT void JNICALL
Java_kr_dailylime_torchcvhelloworld_MainActivity_InferMobileNetClassifier(
        JNIEnv *env,
        jobject instance,
        jlong matAddrInput,
        jlong matAddrResult) {

    // OpenCV convert color format
    cv::Mat &matInput = *(cv::Mat *)matAddrInput;
    cv::Mat &matResult = *(cv::Mat *)matAddrResult;

    cv::Mat bgrMat;
    cv::cvtColor(matInput, bgrMat, cv::COLOR_RGBA2RGB);

    if (TorchModelHandlerInstance == NULL) {
        cv::Mat(cv::Mat::zeros(matInput.size(), CV_8UC1)).copyTo(matResult);
        std::cerr << "Model not initialized!" << std::endl;
        return;
    }

    // Infer using torchscript module
    TorchModelHandler* handler = TorchModelHandlerInstance;

    // During torchscript inference, measure inference time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const std::string infer_result = handler->infer(bgrMat);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    int infer_time_ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count());

    // Get average fps
    handler->fpses.push_back((1000 / infer_time_ms));

    if (handler->fpses.size() == 10) {
        // Get average fps

        float fpses = 0;
        while (!handler->fpses.empty()) {
            fpses += handler->fpses.back();
            handler->fpses.pop_back();
        }

        fpses /= 10;

        loge("Average FPS: " + std::to_string(fpses));
    }

    const std::string infer_time = "Infer time: " + std::to_string(infer_time_ms) + " ms (" + std::to_string(1000 / infer_time_ms) + " FPS)";

    // Draw result on mat
    cv::putText(bgrMat, infer_result, cv::Point2i(16, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0));
    cv::putText(bgrMat, infer_time, cv::Point2i(16, 128), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0));

    // Copy result mat to output mat
    bgrMat.copyTo(matResult);
}

extern "C"
JNIEXPORT void JNICALL
Java_kr_dailylime_torchcvhelloworld_MainActivity_InitializeClassifier(
        JNIEnv *env,
        jobject instance,
        jstring jModelPath,
        jstring jLabelPath) {

    // Initialize TorchModelHandler
    const char *modelPath = env->GetStringUTFChars(jModelPath, 0);
    const char *labelPath = env->GetStringUTFChars(jLabelPath, 0);

    TorchModelHandlerInstance = new TorchModelHandler(modelPath, labelPath);
}