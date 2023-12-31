# Invoice Information Extractor

<div align="center">
  <img src="https://i.postimg.cc/R0Fk2T95/machine-Learning.jpg" width="40%">
</div>

## Description
The fundamental purpose of this project is the extraction of essential invoice data from images. To achieve this goal, we have implemented an automated process and employed a number of specialised tools. 
In the following sections, we will explore in detail the key steps that make up this information extraction process.

## Environment 

This project is interpreted/tested on Ubuntu 22.04 LTS/ macOS Sonoma Versión 14.0 using python3 (version 3.9.5)

## Invoice detection with Grauding DINO
[GraudingDINO](https://github.com/IDEA-Research/GroundingDINO) 
is a computer vision based invoice detection tool. It detects invoices in the provided images and generates tags for their location. You can find more details in its linked documentation.

## Installation and Configuration of groundingDino

Be sure to follow these steps to set up and use groundingDino.

### Pre-requisites
- Python 3.9.5 installed
- Virtual env installed

### Virtual Environment Configuration

1. Create and activate your virtual environment. You can do this by executing the following commands:

```bash
python3 -m venv venv
```
Linux/Mac
```bash
source venv/bin/activate
```
Windows
```bash
.\venv\Scripts\activate 
```

2. Install the necessary dependencies. You can do this by running the following commands:
```bash
pip install opencv-python
```
```bash
pip install Pillow
```
```bash
pip install numpy
```
```bash
pip install torch
```
```bash
pip install torchvision
```
```bash
pip install supervision==0.6.0
```
3. Clone the groundingDino repository
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
```
4. Once the GroundingDINO repository has been cloned, we go inside it to install groundingDINO in silent and editable mode. You can do this by running the following commands:
```bash
pip install -q -e .

```

5. From the main groundingDINO folder, create the 'weights' folder and download the pre-trained weights. You can do this by executing the following commands:
```bash
mkdir weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/

```

## Invoice Labelling with LabelMe
[LabelMe](https://github.com/wkentaro/labelme) It is used to label the invoices detected by GraudingDINO. If an invoice is not correctly labelled during detection, this tool allows you to correct and improve the labels. See the LabelMe documentation for detailed instructions on how to use it.


### Installation of labelme and labelme2yolo

1. Make sure you have labelme and labelme2yolo installed in your environment. You can do this by running the following commands:

```bash
pip install labelme
```
```bash
pip install labelme2yolo
```
2. To convert the labels generated by labelme to YOLO format, use the following command:
```bash
labelme2yolo --json_dir path/to/your/images --output_dir path/del/dataset/YOLODataset
```


## Training a Model YOLOv8l (YOLOv8l)
After labelling the areas of interest in the invoices, a model will be trained YOLOv8l.

[YOLOv8](https://github.com/ultralytics/ultralytics) is an object detection model based on a convolutional neural network (CNN). The CNN will be trained to identify and locate the required information in invoices. See the YOLOv8 documentation for details on its configuration and training.

### Installation of YOLOv8 with Ultralytics

Below are the steps to install Ultralytics YOLOv8 and configure your environment for object detection.

1. We have to download YoloV8l, which is the one we work with in this case, you can use the one you like the most, in our case we use YoloV8l, because it has an excellent performance in this detection process.

- **`YOLOv8l`(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt**

2. create a folder in the root directory called model, to store the download of yolov8l
```bash
mkdir model
```

3. Install the necessary dependencies. You can do this by running the following commands:
```bash
pip install ultralytics
```
```bash
pip install pyyaml
```


## Invoice Segmentation with SAM
Invoice segmentation is carried out with [SAM (Semantic Area Mapper)](https://github.com/facebookresearch/segment-anything.git). SAM allows you to divide invoices into specific areas, which makes it easier to extract information. Be sure to consult the SAM documentation to learn about its capabilities and how to use it in this project.


### Installation of Segmentation Anything Models (SAM)

The following are the steps to install SAM and its dependencies.

1. Cloning the SAM repositoryCloning the SAM repository

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

```
2. Download the pre-trained weights. You can do this by executing the following commands:


- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

3. Once downloaded, we create a folder called sam, and then store the weights in it.

```bash
mkdir sam
```


## Information Abstraction with Tesseract
[Tesseract](V) is an Optical Character Recognition (OCR) library used to extract text from labelled areas on invoices. See the Tesseract documentation to learn how to set it up and use it effectively.

## Export of information
The extracted information is exported to a readout file, which may be a text file or another format suitable for further use.

# Authors

![GitHub Contributors Image](https://contrib.rocks/image?repo=anacardona0220/holbertonschool-higher_level_programming) Ana Maria Cardona Botero - <a href="https://github.com/anacardona0220" target="_blank"> @anacardona0220</a> :genie_woman:![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=anacardona0220&show_icons=true)

![GitHub Contributors Image](https://contrib.rocks/image?repo=juespega/holbertonschool-higher_level_programming) Julian Esteban Perez - <a href="https://github.com/juespega" target="_blank"> @juespega</a> :genie_woman:![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=juespega&show_icons=true)

![GitHub Contributors Image](https://contrib.rocks/image?repo=JDaniel26-Bory/holbertonschool-higher_level_programming) Juan Daniel Restrepo - <a href="https://github.com/JDaniel26-Bory" target="_blank"> @JDaniel26-Bory</a> :genie_woman:![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=JDaniel26-Bory&show_icons=true)