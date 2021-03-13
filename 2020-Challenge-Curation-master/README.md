# 2020 Challenge Dataset Curation Tools

This repository provides the scripts written (and pointers to the applications developed) for use in preparing the multicenter contextual dermoscopic image database for the SIIM-ISIC Melanoma Classification competition. The competition was hosted on Kaggle during the Summer of 2020 https://www.kaggle.com/c/siim-isic-melanoma-classification). The dataset is available for download at the DOI https://doi.org/10.34970/2020-ds01.

## Standardize Identification Metadata

Each image in this dataset is attributed to one patient who of which is associated to multiple other images. Because the images that make up this data come from multiple centers, and because the associated identification metadata in each centers' internal image databases often are considered PHI, it was necessary to assign new patient and lesion identification codes in standard format, free of PHI, for the dataset.

[2020 Challenge assign Id_200408.py](https://github.com/ISIC-Research/2020-Challenge-Curation/blob/master/2020%20Challenge%20assign%20Id_200408.py) reads a csv containing filenames associated with pre-existing lesion IDs and patient IDs, and outputs 3 mapping tables (patient, lesion, and image) between original identification codes and a standardized format absent of PHI. Additional items can be added to the dataset without redundancy or double coding by specifying the existence of any of the three output tables. Duplicate patients, lesions, and images already present in the existing mapping tables are not assigned extra identification codes.

## Image Selector

It is regular practice in some clinics to take multiple images per lesion using various dermoscopy types. We developed a tool to display all images that meet certain characteristics in a UI where a user then selects any number of those which are displayed. We used this to display all images of a certain lesion from a certain timepoint, and asked medical research fellows to select the image or images which were taken using contact dermoscopy. The application is available at [https://github.com/ISIC-Research/lesionimagepicker](https://github.com/ISIC-Research/lesionimagepicker).

## Quality Assurance - "Tagger"

An image organization application for semantic-level annotations was developed to QA diagnosis labels and to rule out those which failed to meet image quality standards. The application code is provided through [https://github.com/dgutman/webix_image_organizer](https://github.com/dgutman/webix_image_organizer). It uses the Webix Toolkit to interact with a Girder server, and the organization method and feature annotation list is configurable and not limited to the specific uses in preparing this dataset. 

## Timepoint Selection

Every lesion in the dataset is represented by just a single image. Generally, all of a patient's lesions were not photographed on a single visit. Thus, some of the within patient variability is attributed to environmental lighting conditions and the evolution of dermoscopic imaging technology. Imaging timpoints were carefully selected in order to minimize bias between patient charts containing a melanoma and patient charts which do not contain a melanoma.

[SelectLesionTimepoint_200423.R](https://github.com/ISIC-Research/2020-Challenge-Curation/blob/master/SelectLesionTimepoint_200423.R) reads a csv containing all potential images for the dataset and each of their associated patient IDs and lesion IDs, and produces a table of choice images according to a list of conditions.
1. There is exactly one image per lesion.
2. Each patient must meet a minimum number of associated lesions.
3. Biopsied lesions are represented by that nearest preceding imaging date to the biopsy.
4. Non-biopsied lesion images in the **melanoma class** of patients are selected to minimize the time difference between the benign and malignant imaging date.
5. Non-biopsied lesion images in the **benign class** of patients are selected to reflect the within-patient imaging date variation of the **melanoma class**.
6. If multiple images are available for a lesion at the selected timepoint, one of them was chosen at random.

