"""
Author: Nicholas Kurtansky
Date Initiated: 11/06/2019
Title: Assign 7-digit image, lesion, and patient IDs
"""


import os
import pandas as pd
import numpy as np


def main():

    # user will input these in the finished product. For efficiency in development, let them already be specified """
    time_stamp = input("Today's date (yymmdd): ")
    contributor_id = input("Enter the name of this contributor or dataset: ")
    #   Establish working directory
    wd = input("Enter the local working directory: ")
    os.chdir(str(wd))
    print(os.listdir())

        #   Read csv table into a dataframe using pandas
    # must include a column each for filename, lesion id, patient id
    new_fn = input("Enter the csv file containing filenames, lesion IDs, and patient IDs of interest: ")
    newTable = pd.read_csv(str(new_fn), delimiter = ",")
        #   User input the column indexes
    image_col = input("Enter the FILENAME column index: ")
    lesion_col = input("Enter the LESION ID column index: ")
    patient_col = input("Enter the PATIENT ID column index: ")

    # 3 python dictionaries to assign random numeric ids to:
    # Patient
    try:
        patId = pd.read_csv(input("Enter the current PATIENT mapping filename if it exists, or hit Enter: "), delimiter = ',')
        # patient mapping
        patmap_id_col = int(input("New 7-digit patient ID column index: "))
        patmap_id = list(patId.iloc[:,patmap_id_col])
        
        patmap_old_col = int(input("Old patient ID column index: "))
        patmap_old = list(patId.iloc[:,patmap_old_col])
     
        patmap_contributor_col = int(input("Contributor column index: "))
        patmap_contributor = list(patId.iloc[:,patmap_contributor_col])

    except FileNotFoundError:
        patmap_id = []
        patmap_old = []
        patmap_contributor= []
        
    # Lesion
    try:
        lesId = pd.read_csv(input("Enter the current LESION mapping filename if it exists, or hit Enter: "), delimiter = ',')
        # lesion mapping
        lesmap_id_col = int(input("New 7-digit lesion ID column index: "))
        lesmap_id = list(lesId.iloc[:,lesmap_id_col])
        
        lesmap_old_col = int(input("Old lesion ID column index: "))
        lesmap_old = list(lesId.iloc[:,lesmap_old_col])
 
        lesmap_contributor_col = int(input("Contributor column index: "))
        lesmap_contributor = list(lesId.iloc[:,lesmap_contributor_col])
        
    except FileNotFoundError:
        lesmap_id = []
        lesmap_old = []
        lesmap_contributor = []

    # Image
    try:
        imgId = pd.read_csv(input("Enter the current IMAGE mapping filename if it exists, or hit Enter: "), delimiter = ',')
        # image mapping
        imgmap_id_col = int(input("7-digit image ID column index: "))
        imgmap_id = list(imgId.iloc[:,imgmap_id_col])

        imgmap_old_col = int(input("Old image ID column index: "))
        imgmap_old = list(imgId.iloc[:,imgmap_old_col])

        imgmap_contributor_col = int(input("Contributor column index: "))
        imgmap_contributor = list(imgId.iloc[:,imgmap_contributor_col])

    except FileNotFoundError:
        imgmap_id = []
        imgmap_old = []
        imgmap_contributor = []
    


    # dictionary of unique patient_id
    unique_pat = newTable.iloc[:,int(patient_col)].unique()
    unique_les = newTable.iloc[:,int(lesion_col)].unique()
    unique_img = newTable.iloc[:,int(image_col)].unique()           

    # track progress
    toGo = len(unique_pat)
    for pat in unique_pat: 
        
        # check if this patient is already assinged to a unique id
        if str(pat) not in list(patmap_old):  
            # assign a unique ID
            exists = True
            while exists:
                samp = np.random.randint(0,9999999)
                if 'IP_{:07d}'.format(samp) not in list(patmap_id):
                    exists = False
            
            # append information to existing lists
            patmap_id.append('IP_{:07d}'.format(samp))
            patmap_old.append(str(pat))
            patmap_contributor.append(contributor_id)
          
        # track progress
        toGo += -1
        print('{}_more patients to go'.format(toGo))

    # track progress   
    toGo = len(unique_les)
    for les in unique_les:
        
        # check if this lesion is already assigned a unique id
        if str(les) not in list(lesmap_old):
            # assign a unique ID
            exists = True
            while exists:
                samp = np.random.randint(0,9999999)
                if 'IL_{:07d}'.format(samp) not in list(lesmap_id):
                    exists = False
                    
            # append information to existing lists            
            lesmap_id.append('IL_{:07d}'.format(samp))
            lesmap_old.append(str(les))
            lesmap_contributor.append(contributor_id)

        # track progress
        toGo += -1
        print('{}_more lesions to go'.format(toGo))
        
    # track progress
    toGo = len(unique_img)
    for img in unique_img:
        
        # check if this lesion is already assigned a unique id
        if str(img) not in list(imgmap_old):
            # assign a unique ID
            exists = True
            while exists:
                samp = np.random.randint(0,9999999)
                if 'II_{:07d}'.format(samp) not in list(imgmap_id):
                    exists = False
            
            # append information to existing lists
            imgmap_id.append('II_{:07d}'.format(samp))
            imgmap_old.append(str(img))
            imgmap_contributor.append(contributor_id)
        
        # track progress                     
        toGo += -1
        print('{}_more images to go'.format(toGo))

 
    # zip lists into pandas dataframes    
    out_patDf = pd.DataFrame(list(zip(patmap_id, patmap_old, patmap_contributor)), 
               columns =['newID', 'oldID', 'subset']) 
    out_lesDf = pd.DataFrame(list(zip(lesmap_id, lesmap_old, lesmap_contributor)), 
               columns =['newID', 'oldID', 'subset']) 
    out_imgDf = pd.DataFrame(list(zip(imgmap_id, imgmap_old, imgmap_contributor)), 
               columns =['newID', 'oldID', 'subset']) 

    
    # write csv filess
    out_patDf.to_csv('Patient ID Mapping_'+time_stamp+'.csv', index = False)
    out_lesDf.to_csv('Lesion ID Mapping_'+time_stamp+'.csv', index = False)
    out_imgDf.to_csv('Image ID Mapping_'+time_stamp+'.csv', index = False)
            
    

# run program
if __name__ == "__main__":
    main()
    
