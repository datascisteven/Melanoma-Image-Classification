# Web or App Implementation of Melanoma Image Classifier

## Overview and Business Understanding

Skin cancer is the most prevalent type of cancer with melanoma being responsible for 75% of skin cancer deaths despite being the least common type. According to the American Cancer Society, in 2021, about 106,000 new melanomas will be diagnosed with a 60/40 split between men and women, and just over 7000 people are expected to die of melanoma.  Melanoma is the deadliest form of skin cancer and is responsible for an overwhelming majority of skin cancer deaths.  When melanoma is detected early, the survival rate exceeds 95% and can be cured with minor surgery. which is the very reason why having access to some sort of screening process as essential to prevent unnecessary death.

Doctors can detect melanomas through visual inspection supplemented by their years of clinical experience, but recent studies have shown that machine learning can detect such lesions through image analysis can be as reliable as if not more than a visit with the dermatologist.  Deploying an app for public consumption that can screen for potential melanoma would be essential to prevent unnecessary death and equalize access to individuals without easy access to advice from an expert clinician especially those from lower socio-economic status.  

Those very populations inherently have constructed a certain skepticism and reticence to relate to and seek advice from a doctor of a different ethnic background.  Some of these barriers are due to past offenses and some purely cultural.  Creating such an app would allow for access by such individuals and break down some of these initial barriers so that they may receive the care they truly need.


## Data Understanding

As the leading healthcare organization for informatics in medical imaging, the Society for Imaging Informatics in Medicine (SIIM)'s mission is to advance medical imaging informatics through education, research, and innovation in a multi-disciplinary community. SIIM is joined by the International Skin Imaging Collaboration (ISIC), an international effort to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermatoscopic images of skin lesions.

There are three public repositories curated by SIIM-ISCC that we relied on for the project:

2020 Training Dataset:
- 33,126 DICOM images with embedded metadata and JPEG images with 425 duplicates
- 33,126 metadata entries of patient ID, lesion ID, sex, age, and general anatomic site.
- Training Set: 467 melanoma vs. 26033 non-melanoma images
- Validation Set: 117 melanoma vs. 6509 non-melanoma images

2020 Testing Dataset:
- 10,982 DICOM images with embedded metadata and JPEG images
- 10,982 metadata entries of patient ID, sex, age, and general anatomic site.
- Holdout Set: 10982 images

The 2020 dataset contains 33,126 dermatoscopic training images of skin lesions from over 2000 patients, and the images are either in DICOM format, which is a common medical imaging data format or in JPEG and TYFRecord format.  DICOM files contain both the image and metadata, but the metadata is also provided outside of the DICOM format in CSV format, which includes the following features:

1. patient ID
2. gender
3. approximate age
4. location of imaged site,
5. diagnosis information
6. indicator of malignancy
7. binarized version of target variable (melanoma or not melanoma)





## Data Preparation:

**File and folder management**
- Challenges with unstructured data
- Keras requires the data to be organized into training, validation, and testing folders with the classes organized as subfolders to create the testing sets
- Time consuming process of moving folders 
- Challenge of incorporating folder

**Class Imbalance**
- Employ a variety of methods to address severe class imbalance
- Additional Datasets for minority class augmentation:
	- 4522 additional melanoma images from the 2019 Training Dataset
	- 1114 additional melanoma images from the 2018 Training Dataset
- ImageDataGenerator() transformations
- Albumentation() transformations

Some of the challenges we faced in the initial phase of the project include learning to manage such a large set of unstructured data and whether to incorporate the use of DICOM files into the project.  DICOM is a specific file format that is specific to the medical industry.  The file contains not only the image file but also the metadata, which consists of patient information, both personal and clinical.  Learning how to use 

## Modeling and Evaluation:

**Baseline Model:**
- `Sequential()`
- 2 convolutional layers with input shape (224, 244, 3) with filters applied to extract different features:
	- Filters: number of filters that convolutional layer will learn
	- `kernel_size`: specifies width and height of 2D convolutional window
	- Padding:  same ensure that spatial dimensions are the same after convolution
	- Activation:  activation function that will be applied for convolutional layers
	- `layers.Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="same", activation="relu"))`
- `BatchNormalization()`
	- acts like standardization or normalization for regression models
- `MaxPool2D()` 
	- To reduce dimensionality of images by reducing number of pixels in output
- `layers.MaxPool2D(pool_size=(2,2),strides=(2,2))`
- `Flatten()`
	- To be able to generate a prediction, flatten output of convolutional base
- `layers.Flatten()`
- Dense layers feeds output of convolutional base to neurons
- `layers.Dense(units=4096, activation="relu"))`
- Loss function:  `loss= ‘binary_crossentropy’`
- Optimizer:  `Adam(learning_rate=0.01)`

**Metrics:**
- Accuracy
- Precision (Positive Predictive Value)
- Recall (True Positive Rate)
- ROC-AUC Score
- PR-AUC Score


## Folder Structure:

├── README.md                   <- the top-level README for reviewers of this project
├── _notebooks			<- folder containing all the project notebooks
│   ├── albumentation.ipynb		<- notebook for displaying augmentations
│   ├── EDA.ipynb			<- notebook for dataset understanding and EDA
│   ├── folders.ipynb			<- notebook for image folder management
│   ├── modeling.ipynb			<- notebook for models with imbalanced dataset
│   ├── modeling2.ipynb			<- notebook for models with dataset with augmentations
│   ├── pretrained.ipynb		<- notebook for pretrained models
│   └── utils.py  			<- py file with self-defined functions
├── final_notebook.ipynb        <- final notebook for capstone project
├── _data                       <- folder of csv files (csv)
├── final_presentation.pdf    	<- pdf of the final project presentation
├── MVP Presentation.pdf		<- pdf of the MVP presentation
├── _images                     <- folder containing visualizations
├── _split			<- folder substructure of image folder (not on Github)
│   ├──	_train				<- folder containing training JPEG files
│   │	├── _mel					
│   │	│   ├── _2020
│   │	│   ├── _2019
│   │	│   ├── _2018
│   │	│   └── _aug
│   │	└──	_not_mel				
│   ├── _train_dcm			<- folder containing training DICOM files
│   ├── _val				<- folder containing validation JPEG files
│   │	├── _mel
│   │	└── _not_mel					
│   ├── _test2				<- folder containing test JPEG files
│   │	├── _mel
│   │	└── _not_mel	
│   └── _train_imb			<- folder containing original JPEG files
│	├── _mel
│	└── _not_mel	
├── _sample			<- folder containing sample dataset of images (not on Github)
├── _models			<- folder containing saved models (not on Github)
└── utils.py			<- py file with self-defined functions


## Contact Information:

**Steven Yan**

Email:  [stevenyan@uchicago.edu][1]

LinkedIn:   [https://www.linkedin.com/in/examsherpa][2]

Github:  [https://www.github.com/examsherpa][3]



## References:

International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration [https://doi.org/10.34970/2020-ds01][4] (2020).

Rotemberg, V. _et al_. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. _Sci. Data_ 8: 34 (2021). [https://doi.org/10.1038/s41597-021-00815-z]()

ISIC 2019 data is provided courtesy of the following sources:

BCN20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [https://doi.org/10.1038/sdata.2018.161][6]
MSK Dataset: (c) Anonymous; [https://arxiv.org/abs/1710.05006][7] ; [https://arxiv.org/abs/1902.03368][8]

Tschandl, P. _et al_. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. _Sci. Data_ 5: 180161 doi: 10.1038/sdata.2018.161 (2018)

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.

Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: “BCN20000: Dermoscopic Lesions in the Wild”, 2019; arXiv:1908.02288.

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; [https://arxiv.org/abs/1902.03368][9]


[1]:	mailto:stevenyan@uchicago.edu
[2]:	https://www.linkedin.com/in/examsherpa
[3]:	https://www.github.com/examsherpa
[4]:	https://doi.org/10.34970/2020-ds01
[6]:	https://doi.org/10.1038/sdata.2018.161
[7]:	https://arxiv.org/abs/1710.05006
[8]:	https://arxiv.org/abs/1902.03368
[9]:	https://arxiv.org/abs/1902.03368
