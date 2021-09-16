Android JNI (Java Native Interface) example of PyTorch+OpenCV realtime image classification using MobileNetV3 classifier

## Summary
- Sample real-time android image classifier application using MobileNetV3
- Used PyTorch with OpenCV (both native C++)
- Detailed configurations and tutorials are written on my blog: []()

## Model stats (from [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch))
|      Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
|:----------------------:|:------------:|:------:|:--------------------------:|
|  MobileNetV3-Large 1.0 |    5.483M    | 216.60 |       74.280 / 91.928      |
| MobileNetV3-Large 0.75 |    3.994M    | 154.57 |       72.842 / 90.846      |
|  MobileNetV3-Small 1.0 |    2.543M    |  56.52 |       67.214 / 87.304      |
| MobileNetV3-Small 0.75 |    2.042M    |  43.40 |       64.876 / 85.498      |

## Performance
|          Model         | Average ms | Average FPS |
|:----------------------:|:----------:|:-----------:|
|  MobileNetV3-Large-1.0 |     34     |      29     |
| MobileNetV3-Large-0.75 |     32     |      31     |
|  MobileNetV3-Small-1.0 |     22     |      45     |
| MobileNetV3-Small-0.75 |     20     |      30     |

- Performances are measured in C++, extra overheads may occur.
    ```cpp
    // During torchscript inference, measure inference time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const std::string infer_result = handler->infer(bgrMat);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    int infer_time_ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count());
    ```

## References
- [MAKING NATIVE ANDROID APPLICATION THAT USES PYTORCH PREBUILT LIBRARIES](https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html)
- [Android NDK + OpenCV 카메라 예제 및 프로젝트 생성방법(ndk-build 사용)](https://webnautes.tistory.com/923)
- MobileNetV3 PyTorch implementation - [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)