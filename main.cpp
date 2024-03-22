#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include "yolov8cpp/yolov8.cpp"

#include <fstream>

// ifstream if you just want to read 
// ofstream to write
// fstream for both).

void yamlParse(std::string var)
{
    std::string path;
    path = "../coco128.yaml";

    std::ifstream yaml_file;
    yaml_file.open(path);

    // std::string line;
    // while (getline(yaml_file,line))
    // {
    //     auto res = line.find(var);
    //     if(res!=std::string::npos){
    //         std::cout<<line<<std::endl;
    //     };
    // }

    std::istream_iterator<std::string> fileIterator(yaml_file);
    std::istream_iterator<std::string> fileIteratorEnd;

    std::cout << "\n yaml file : \n";
    
    std::string readPart;
    // Note that the std::istream_iterator reads space-separated words
    while (fileIterator!=fileIteratorEnd)
    {
        readPart = *fileIterator++;
        if (readPart==var)
        {
            break;
        }     
    }

    if (fileIterator==fileIteratorEnd)
    {
        std::cout<<std::endl<<"can't find the variable in the yaml file!" <<std::endl;
        return;

    } else {
        fileIterator++;
        std::vector<std::string> result;

            while (fileIterator!=fileIteratorEnd)
            {
                if (*result.end()==":")
                {

                }
            }
    }
   

} // end of yamlParse()

int main(int argc, const char* argv[])
{
    // if (argc != 2) {
    // std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    // return -1;
    // }

    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU);


    // Note that in this example the classes are hard-coded
    std::vector<std::string> classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                                      "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                      "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


   
    torch::jit::script::Module model;
    try {
        // Load the model (e.g. yolov8s.torchscript)
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load(argv[1]);
        
        model = torch::jit::load("/home/ws/data/yolov8l.torchscript");
        std::cout<<"model loaded successfully! \n";
        model.eval();
        model.to(device, torch::kFloat32);        
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Load image and preprocess
    cv::Mat image = cv::imread("/home/ws/data/im2.jpg");
    cv::Mat input_image;
    letterbox(image, input_image, {640, 640});

    torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte).to(device);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({2, 0, 1});
    image_tensor = image_tensor.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs {image_tensor};

    // Inference
    torch::Tensor output = model.forward(inputs).toTensor().cpu();
    
    // NMS
    auto keep = non_max_supperession(output)[0];
    auto boxes = keep.index({Slice(), Slice(None, 4)});
    keep.index_put_({Slice(), Slice(None, 4)}, scale_boxes({input_image.rows, input_image.cols}, boxes, {image.rows, image.cols}));

    // Show the results
    for (int i = 0; i < keep.size(0); i++) {
        int x1 = keep[i][0].item().toFloat();
        int y1 = keep[i][1].item().toFloat();
        int x2 = keep[i][2].item().toFloat();
        int y2 = keep[i][3].item().toFloat();
        float conf = keep[i][4].item().toFloat();
        int cls = keep[i][5].item().toInt();
        std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << classes[cls] << std::endl;
    }

    std::cout<<"adet "<<keep.size(0)<<std::endl;





    std::cout<<"bitti \n";

    yamlParse("names");

    return 0;
    
}