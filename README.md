# Melanoma Image Classification with Flask Application

## Project Page

Please navigate to this page for the original final report for the project:

https://datascisteven.github.io/Melanoma-Image-Classification

**Presentation Link:** https://prezi.com/view/JoLKnGuw0ZFBYFZra0c3/


## Flask application

While I had been exploring implementation through Flutter for app deployment, Flask seemed much more feasible given my time constraints and level of expertise.

## Home page:

<img src="images/Homepage.png">

The homepage asks for the user to upload a JPEG of any size into the application and to press SUBMIT once done.


## Results page:

<img src="images/Results.png">

Upon pressing SUBMIT, you automatically get transferred to the Results page, and you are given a message to get the mole checked out or that it is just another beauty mark.  The confidence level of that prediction is also given.
   

## Folder Structure:

	├── README.md                   <- the top-level README for reviewers of this project
	├── _notebooks					<- folder containing all the project notebooks
	│   ├── albumentation.ipynb		<- notebook for displaying augmentations
	│   ├── EDA.ipynb				<- notebook for dataset understanding and EDA
	│   ├── folders.ipynb			<- notebook for image folder management
	│   ├── holdout.ipynb			<- notebook for predicting on holdout sets
	│   ├── preaugmentation.ipynb	<- notebook for models with imbalanced dataset
	│   ├── postaugmentation.ipynb	<- notebook for models with dataset post-augmentations
	│   ├── pretrained.ipynb		<- notebook for pretrained models
	│   └── utils.py  				<- py file with self-defined functions
	├── final_notebook.ipynb        <- final notebook for capstone project
	├── _data                       <- folder of csv files (csv)
	├── MVP Presentation.pdf		<- pdf of the MVP presentation
	├── _Melanoma-Flask				<- folder with Flask application
	└── utils.py					<- py file with self-defined functions


## Contact Information:

**Steven Yan**

<img src="images/mail_icon.png"> **Email:**  [stevenyan@uchicago.edu][1]

<img src="images/linkedin_icon.png"> **LinkedIn:**   [https://www.linkedin.com/in/datascisteven][2]

<img src="images/github_icon.png"> **Github:** [https://www.github.com/datascisteven][3]


## References:

International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration [https://doi.org/10.34970/2020-ds01][4] (2020).

Rotemberg, V. _et al_. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. _Sci. Data_ 8: 34 (2021). [https://doi.org/10.1038/s41597-021-00815-z][5]

ISIC 2019 data is provided courtesy of the following sources:

- BCN20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
- HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [https://doi.org/10.1038/sdata.2018.161][6]
- MSK Dataset: (c) Anonymous; [https://arxiv.org/abs/1710.05006][7] ; [https://arxiv.org/abs/1902.03368][8]

Tschandl, P. _et al_. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. _Sci. Data_ 5: 180161 doi: 10.1038/sdata.2018.161 (2018)

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.

Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: “BCN20000: Dermoscopic Lesions in the Wild”, 2019; arXiv:1908.02288.

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; [https://arxiv.org/abs/1902.03368][9]


[1]:	mailto:stevenyan@uchicago.edu
[2]:	https://www.linkedin.com/in/datascisteven
[3]:	https://www.github.com/datascisteven
[4]:	https://doi.org/10.34970/2020-ds01
[5]:    https://doi.org/10.1038/s41597-021-00815-z
[6]:	https://doi.org/10.1038/sdata.2018.161
[7]:	https://arxiv.org/abs/1710.05006
[8]:	https://arxiv.org/abs/1902.03368
[9]:	https://arxiv.org/abs/1902.03368
