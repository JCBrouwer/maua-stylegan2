#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor rotate(torch::Tensor image, double angle) {
  image = image.to(torch::kCPU);
  cv::Mat image_mat(image.size(0),
                    image.size(1),
                    CV_32FC1,
                    image.data_ptr<float>());

  cv::Mat output_mat;
  cv::Point2f centre(((float)image.size(0))/2.0,((float)image.size(1))/2.0);
  cv::Mat rotationMat = cv::getRotationMatrix2D(centre,angle,1);
  cv::warpAffine(image_mat, output_mat, rotationMat, image_mat.size());

  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(),{image.size(0), image.size(1)});
  return output.clone().to(torch::kCUDA);
}

static auto registry =
  torch::RegisterOperators("my_ops::rotate", &rotate);