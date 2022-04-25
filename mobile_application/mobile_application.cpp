#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// This is the Circle2D class.
class DataLoading
{
public:

	DataLoading()
	{

	}

	// Return the number of circle objects
	static uint8_t*** getImagePair()
	{
		return imagePair;
	}

private:	
  static uint8_t *** imagePair; // 1024 x 224, 2 * 224 = 448, RGB
};

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "Model loaded successfully\n";

  Mat frame;
  //--- INITIALIZE VIDEOCAPTURE
  VideoCapture cap;
  // open the default camera using default API
  // cap.open(0);
  // OR advance usage: select any API backend
  int deviceID = 0;             // 0 = open default camera
  int apiID = cv::CAP_ANY;      // 0 = autodetect default API
  // open selected camera using selected API
  cap.open(deviceID, apiID);
  // check if we succeeded
  if (!cap.isOpened()) {
      cerr << "ERROR! Unable to open camera\n";
      return -1;
  }
  //--- GRAB AND WRITE LOOP
  cout << "Start grabbing" << endl
      << "Press any key to terminate" << endl;
  for (;;)
  {
      // wait for a new frame from camera and store it into 'frame'
      cap.read(frame);
      // check if we succeeded
      if (frame.empty()) {
          cerr << "ERROR! blank frame grabbed\n";
          break;
      }
      // show live and wait for a key with timeout long enough to show images
      imshow("Live", frame);
      if (waitKey(5) >= 0)
          break;
  }
}