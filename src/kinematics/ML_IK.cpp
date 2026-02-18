#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cstring>

// ======= REAL SCALERS FROM TRAINING =======

static float x_min[3] = {
    -1.66917063f,
     0.00156376333f,
    -1.40007092f
};

static float x_scale[3] = {
    6.69163434f,
    2.6103421f,
    4.00012255f
};

static float y_min[3] = {
    0.000773813202f,
    0.357333827f,
   -1.00726712f
};

static float y_scale[3] = {
    0.76267821f,
    0.7851248f,
    0.66299091f
};

class ML_IK
{
public:
    ML_IK(const char* model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "ik"),
          session(nullptr)
    {
        Ort::SessionOptions opts;
        std::wstring wpath(model_path, model_path + strlen(model_path));
        session = Ort::Session(env, wpath.c_str(), opts);
    }

    std::vector<float> predict(float x, float y, float z)
    {
        // -------- Correct Normalization --------
        float nx = x * x_scale[0] + x_min[0];
        float ny = y * x_scale[1] + x_min[1];
        float nz = z * x_scale[2] + x_min[2];

        std::vector<float> input = {nx, ny, nz};
        std::vector<int64_t> shape = {1,3};

        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                mem, input.data(), 3,
                shape.data(), 2);

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* out = output_tensors[0]
                         .GetTensorMutableData<float>();

        // -------- Correct Denormalization --------
        float t1 = (out[0] - y_min[0]) / y_scale[0];
        float t2 = (out[1] - y_min[1]) / y_scale[1];
        float t3 = (out[2] - y_min[2]) / y_scale[2];

        return {t1, t2, t3};
    }

private:
    Ort::Env env;
    Ort::Session session;
};



