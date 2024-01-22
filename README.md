
# **LANE DETECTION USING ENET**
Lane detection is a crucial computer vision task that involves identifying the boundaries of driving lanes in an image or video of a road scene. It plays a vital role in various applications, particularly in the realm of autonomous vehicles and advanced driver-assistance systems (ADAS).  Convolutional neural networks (CNNs) are now the dominant approach for lane detection. Trained on large datasets of road images, these models can achieve high accuracy even in challenging situations.  In this we implemented ENET architecture which is a deep learning algorithm widely used for image segmentation.
## ENET(EFFICIENT NET)
ENet (Efficient Neural Network) is a deep neural network architecture designed for real-time semantic segmentation tasks.    ENet is significantly faster than other popular semantic segmentation architectures, such as VGG and ResNet. This is due to its use of several techniques to reduce the number of parameters and floating-point operations (FLOPs) required. For example, ENet uses a smaller number of convolutional filters and smaller kernel sizes than other architectures. Additionally, ENet uses a technique called "bottleneck residual connections" to reduce the number of FLOPs required for each layer.

Overall, ENet is a powerful and efficient architecture for real-time semantic segmentation tasks. It is a good choice for applications where low latency and memory footprint are important considerations.


## TuSimple Dataset
The TuSimple dataset is a large-scale dataset for autonomous driving research, focusing on lane detection and perception tasks. It's widely used in computer vision and autonomous driving communities for benchmarking and developing algorithms.

The TuSimple dataset consists of 6,408 road images on US highways. The resolution of image is 1280×720. The dataset is composed of 3,626 for training, 358 for validation, and 2,782 for testing called the TuSimple test set of which the images are under different weather conditions.



## ENET Architecuture 

![ENET](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/d90a3ebc-f794-4e3c-b0f7-c983808818a2)

## Downloads :    
Download the Dataset Here: [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple)



Checkout the Kaggle Link for this project : [Kaggle](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-enet)
## Getting Started 

To run this project you can download the ENET.ipynb file provided in the repository and the dataset from the download section and can implement the whole process by following the instructions in the [Kaggle Link](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-enet).  Below are the basic Requirements to run the code 
  - ![](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  - GPU
  - CUDA

I choose Kaggle to implement this because it provides inbuilt GPU accelerator which accelerate the training process, I used  GPU P100 to implement this.  You can also choose google colab to run this, google colab also provides inbuilt GPU accelerator which fast up the training process much faster that using CPU.
## Training the Model

To train this model I used  GPU P100 accelerator which accelerated my trained process much more faster than using CPU.  In my model training process the training Epochs are 32, batch size is 8 with Adam Optimizer with learning rate of 0.00005 and the process went well with higher accuracy and low loss. 

I used  [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) to implement this project and the training process went very well with an overall good accuracy and low loss. 



### Test 

You can download the weights file ENET.pth file and directly test it for predictions.  

Also find the inference_ent.ipynb file which contained the testing or inference code.

To test the code
```bash
  - Download the "inference_enet.ipynb" file (Testing Code) and the "ENET.pth" file (model weights) and "lane_detector.py" file.
  - Open the "inference_enet.ipynb" file and specify the "ENET.pth" file in the "model_path" variable. Next, provide the file path for the testing image in the 
     "input_image_path" variable.
  - By executing the "inference_enet.ipynb" file, you can visualize the predictions.
  - Feel free to test it on any image of your choice.
Note:- Make sure that all the three files "inference_enet.ipynb","ENET.pth" and "lane_detector.py" in the same directory so that the code run without any errors.
```

## METRICS VISUALIZATION

![ENET_V](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/c406694c-4266-48ff-9175-b655083ad14d)


The Above graph visualize the metrics during the training process, it shows the graph showing Training & Validation Loss and Training & Validation Accuracy with the staring value and ending value.  The graphs shows the gradual decrease in the loss function and gradual increase accuracy as shown in the visualization.

You can also check the TensorBoard logs to visualize the metrics and the layers in the Architecture.  Here are the TensorBoard visualization of graphs:

![enet_BA](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/16ba871a-c0a7-4a5b-9575-1e4b788224ac)

The Above graph visualize the Binary Accuracy of ENet model visualized using TensorBoard.

![enet_BL](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/2846344c-7f13-4fe9-950b-a3b351534e14)

The Above graph visualize the Binary Loss of ENet model visualized using TensorBoard.

![enet_IL](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/35c9a78a-90e7-4b33-b6c1-105dc63bc530)

The Above graph visualize the Instance Loss of ENet model visualized using TensorBoard.

To run the TensorBoard logs follow the command in your Terminal:
```bash
tensorboard --logdir=path/to/your/logs/directory
```
After running the command, open your web browser and go to http://localhost:6006 to access the TensorBoard interface. You'll be able to navigate through the different tabs to explore the data recorded in the tensorboard v2 file.
## Predictions 

![1_enet](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/20e88d9d-f9fc-4497-b461-1e91daa0a6cb)

![2_enet](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/a0d293a1-7595-4c10-93b1-d6453d7cc2eb)

![3_enet](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/d6565822-dd32-4383-95fe-4bb38d7d9f72)

![4_enet](https://github.com/maheshmm7/ENET_Lane_Detection/assets/121345928/36133c9c-19bd-438d-be06-21869f42c00f)



I tested the Predictions on the inference code by loading the saved .pth weights file and testing it on the new images.  The model predictions came out to be good as shown in the figures.

## 🔗 Connect with me
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/maheshmm7)  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rangala-mahesh-455163233/)  [![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/MAHESHRANGALA13)


[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://www.instagram.com/mahesh_mm7/)  [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/rangalamahesh)  [![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Rangala-Mahesh)



## 🚀 About Me

I am Rangala Mahesh, an enthusiastic and versatile individual deeply passionate about the realm of technology and its endless possibilities.

- 🔭 I’m currently Pursing **BTech in Computer Science and Engineering in Centurion University of Technology and Management** 

- 🌱 I’m currently learning about **Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, other Programming Languages**

- 👨‍💻 All of my projects are available at [https://github.com/maheshmm7](https://github.com/maheshmm7)

- 📫 How to reach me **maheshrangala7@gmail.com**


