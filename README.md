
A lane detection integrated Real-time Instance Segmentation based on YOLACT (You Only Look At CoefficienTs), which is a simple, fully convolutional model developed by Daniel Bolya, Chong Zhou, Fanyi Xiao and Yong Jae Lee in 2019 (see repository https://github.com/dbolya/yolact). Here are the codes for their papers:
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)
 
In order to use YOLACT++, make sure you compile the DCNv2 code. (See [Installation](https://github.com/dbolya/yolact#installation))

# Sample running 

![image](https://github.com/jkd2021/YOLACT-with-lane-detection/blob/main/data/123.gif)


# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/jkd2021/YOLACT-with-lane-detection.git
   cd YOLACT-with-lane-detection
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YOLACT on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```
 - If you want to use YOLACT++, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
   ```Shell
   cd external/DCNv2
   python setup.py build develop
   ```


# Evaluation
see Evaluation in original YOLACT models https://github.com/dbolya/yolact#evaluation (released on April 5th, 2019)

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4

# Process a video with higher frame rate and save it to another file.
python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.3 --top_k=20 --video_multiframe=16 --display_fps --video=../video/Hangzhou_Street_01.mp4:output_HZ_Street_01.mp4

# Process a video with higher frame rate
python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.3 --top_k=20 --video_multiframe=16 --display_fps --video=../video/Hangzhou_Street_01.mp4

```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python eval.py --help
```


# Training
see Training in original repository https://github.com/dbolya/yolact#training

# Citation
If you use any code from here base in your work, please cite
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

For YOLACT++, please cite
```
@article{yolact-plus-tpami2020,
  author  = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title   = {YOLACT++: Better Real-time Instance Segmentation}, 
  year    = {2020},
}
```

