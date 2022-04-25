#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
    }
    
    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    return tensor_image;
}

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 3, 2, 1, 0})
{
    tensor = tensor.permute(dims);
    return tensor;
}

class DataLoader
{
public:

	DataLoader()
	{
    firstFrame = 0;
    currentFrame = 0;
	}

  void initImagePair(cv::Mat frame)
  {
    firstFrame = frame.clone(); // Saves only the first frame
    currentFrame = frame; // Keeps up with video feed
  }

  auto getImagePair()
  {
    
    ////////////////////////////////////////////////////////////////////////////
    
    // Process image pair
    imagePair = torch::cat({ToTensor(firstFrame), ToTensor(currentFrame)}, 2);

    // convert the tensor into float and scale it 
    imagePair = imagePair.toType(c10::kFloat).div(255);
    // swap axis 
    imagePair = transpose(imagePair, { (2),(1),(0) });
    //add batch dim (an inplace operation just like in pytorch)
    imagePair.unsqueeze_(0);

    return ToInput(imagePair);

    ///////////////////////////////////////////////////////////////////////////
  }

private:	
  cv::Mat firstFrame;
  cv::Mat currentFrame;
  at::Tensor imagePair;
};

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module DR_TANet;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    DR_TANet = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "Model loaded successfully\n";

  cv::Mat frame;
  //--- INITIALIZE VIDEOCAPTURE
  cv::VideoCapture cap;
  // open the default camera using default API
  // cap.open(0);
  // OR advance usage: select any API backend
  int deviceID = 0;             // 0 = open default camera
  int apiID = cv::CAP_ANY;      // 0 = autodetect default API
  // open selected camera using selected API
  cap.open(deviceID, apiID);
  // check if we succeeded
  if (!cap.isOpened()) {
      std::cerr << "ERROR! Unable to open camera\n";
      return -1;
  }
  //--- GRAB AND WRITE LOOP
  std::cout << "Start grabbing" << std::endl
      << "Press any key to terminate" << std::endl;

  DataLoader *dataLoader = new DataLoader();

  bool onFirstFrame = true;
  for (;;)
  {
    // wait for a new frame from camera and store it into 'frame'
    cap.read(frame);
    // check if we succeeded
    if (frame.empty()) {
        std::cerr << "ERROR! blank frame grabbed\n";
        break;
    }

    if (onFirstFrame)
    {
      dataLoader->initImagePair(frame);
      onFirstFrame = false;
    }

    // Compute prediction
    at::Tensor prediction = DR_TANet.forward(dataLoader->getImagePair()).toTensor();  
    // Activation
    prediction = torch::sigmoid(prediction);
    prediction = torch::where(prediction > 0.5, 255, 0);
    prediction = prediction.squeeze(0).to(torch::kInt);

    //std::cout << "max:" << std::endl;
    //std::cout << torch::max(prediction) << std::endl;
    //std::cout << "min:" << std::endl;
    //std::cout << torch::min(prediction) << std::endl;

    cv::Mat predMat = cv::Mat(prediction.sizes()[2], prediction.sizes()[1], CV_8U);

    std::memcpy(prediction.data_ptr(), predMat.data, sizeof(torch::kInt)*prediction.numel());

    //cv::Mat binaryMat;

    //cv::threshold(predMat, binaryMat, 0.5, 255, 0);

    //std::cout << "predMat:" << std::endl;
    //std::cout << binaryMat << std::endl;

    //Show the results
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", predMat);

    // show live and wait for a key with timeout long enough to show images
    // Show prediction
    //cv::imshow("Live", predImg);
    if (cv::waitKey(5) >= 0)
        break;
  }
}