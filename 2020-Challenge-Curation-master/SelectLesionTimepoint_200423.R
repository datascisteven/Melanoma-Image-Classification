################################################################
# Author: Nicholas Kurtansky
# Date Initiated: 04/23/2020
# Title: Timepoint and Image Selection - 1 Image per Lesion
################################################################

# LOAD PACKAGES

library(dplyr)
library(stringr)
library(ggplot2)
library(moments)


#################### BEGIN ENTERING PARAMETERS

# MINIMUM NUMBER OF PATIENT LESIONS
min_les <- 3

# DATE STAMP
todaysDate <- 'YYMMDD'

# DATASET NAME
thisSite <- 'SUBSET'

# LOCAL DIRECTORY
directory <- "LOCAL WORKING DIRECTORY"
setwd(directory)

# DATA (CSV)
data_filename <- "IMAGE LEVEL DATA FOR TIMEPOINT SELECTION.CSV"
#
# Table must include the following fields:
#   1. diagnosis_confirm_type:
#       {
#         diagnosis determined via histopathological assessment after biopsy: x = "histopathology"
#         else: x = "serial imaging showing no change", "single image expert consensus", "confocal microscopy with consensus dermoscopy", etc.
#       }
#   2. benign_malignant:
#       {
#         malignant diagnosis (ex. melanoma): x = "malignant"
#         benign diagnosis (ex. nevi): x = "benign"
#       }
#   3. patient_id:
#       # unique patient identifier
#   4. lesion_id:
#       # unique lesion identifier
#   5. imaging_day:
#       date (%m/%d/%y) of photagrphy capture
#   6. filename
#
data <- read.csv(data_filename, header=T, stringsAsFactors = F)
str(data)

# COLUMN NUMBERS FOR THE ABOVE VARIABLES
#   PLEASE ADJUST TO FIT 'DATA'
diagnosis_confirm_type_col <- 10
benign_malignant_col <- 11
patient_id_col <- 5
lesion_id_col <- 4
imaging_day_col <- 14
filename_col <- 1

#################### ENTERING PARAMETERS IS COMPLETE








#################### PREPARE DATAFRAME

data <- data %>%
  select(filename_col, diagnosis_confirm_type_col, benign_malignant_col, patient_id_col, lesion_id_col, imaging_day_col)
colnames(data) <- c("filename",
                    "diagnosis_confirm_type",
                    "benign_malignant",
                    "patient_id",
                    "lesion_id",
                    "imaging_day")

### INCLUSION REQUIREMENTS
# filter accordingly
data <- data %>% 
  mutate(biopsied = ifelse(diagnosis_confirm_type == "histopathology", 1, 0)) %>%
  mutate(dx_chal = ifelse(benign_malignant == 'malignant', 'melanoma', 'benign'))
# format dates
data$imaging_day <- as.Date(data$imaging_day, tryFormats = "%m/%d/%y")


# list of patients in melanoma-patient class
melpatient_id <- unique(data$patient_id[data$dx_chal == 'melanoma'])
data <- data %>% 
  mutate(patClass = ifelse(patient_id %in% melpatient_id, 'MelPat', 'CtrlPat'))


## FILTER OUT PATIENT'S WITHOUT MINUMUM NUMBER OF LESIONS
data_perpat <- data %>% group_by(patient_id) %>% summarise(n = length(unique(lesion_id))) %>% filter(n >= min_les)
data <- data %>% filter(patient_id %in% data_perpat$patient_id)


##################################################################################################
##################################################################################################
# PAUSE
# ... SUMMARY OF PATIENT COUNT - BEGIN
##################################################################################################
##################################################################################################

# Count of melanoma patients in the dataset under varying requirements
n <- c()
inclusion <- c()
for(i in 1:5){
  filt <- data
  melPat <- filt %>% filter(benign_malignant == "malignant") %>% pull(patient_id)
  count <- filt %>% filter(patient_id %in% melPat) %>% group_by(patient_id) %>% summarise(n = n())
  count <- count %>% filter(n > i+1)
  n <- c(n, nrow(count))
  inclusion <- c(inclusion, paste("melanocytic only. context >", i))
}  
data.frame(inclusion, n)

###################################################################################################
##################################################################################################
# SUMMARY OF PATIENT COUNT - END...
# RESUME
##################################################################################################
##################################################################################################


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#
#
# BIOPSIED LESIONS
#
# FIRST SELECT THE TIMEPOINTS FOR THE BIOPSIED LESIONS IN THE DATASET. THE TIMEPOINT WILL ALWAYS BE THE 
# CLOSEST PRECEDING IAMGE TO THE LESION'S PATHOLOGY REPORT DATE.
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# Initiate output table
out <- data.frame(filename = character(), patient_id = character(), imaging_day = character(), lesion_id = character(), biopsied = numeric(), benign_malignant = character(), stringsAsFactors = F)
#for(p in 1:nrow(raw_pat)){
for(p in unique(data$patient_id)){
  
  ###############
  # track progress
  print(p)
  ###############
  
  temp_p <- data %>% filter(patient_id == p & biopsied == 1)
  
  if(nrow(temp_p) < 1){
    next
  }

  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
    patient_id_l <- temp_l$patient_id[1]
    lesion_id_l <- temp_l$lesion_id[1]
    
    # closest image date/exam id to every biopsy
    # image date (top of sorted table)
    imaging_day_l <- temp_l$imaging_day[1]

    biopsied_l <- 1
    dx_chal <- temp_l$benign_malignant[1]

    fn <- temp_l$filename[1]

    out <- rbind(out, data.frame(filename = fn, patient_id = patient_id_l, imaging_day = imaging_day_l, lesion_id = lesion_id_l, biopsied = biopsied_l, benign_malignant=dx_chal, stringsAsFactors = F))
  }
}
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#
#
# NON-BIOPSIED LESIONS - MELANOMA PATIENT CLASS
#
# IF MULTIPLE IMAGES ARE AVAILABLE, SELECT THE IMAGE CLOSEST TO A PATIENT'S
# MELANOMA IMAGE.
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


melanomasAll <- out %>% filter(benign_malignant == 'malignant')

# NEED VECTOR OF MELANOMA CLASS PATIENTS
melPat_patient_id <- unique(melanomasAll$patient_id)
melPat_lesion_id <- out$lesion_id
# UNBIOPSIED LESIONS OF THE MELANOMA PATIENT CLASS
temp_data <- data %>% filter(patient_id %in% melPat_patient_id & !(lesion_id %in% melPat_lesion_id))

# number of timepoints per lesion
nExams <- temp_data %>% group_by(lesion_id) %>% summarise(exams = length(unique(imaging_day)))
table(nExams$exams)
# Most of the lesions have a single timepoint... select them first for efficiency
temp_1exam <- temp_data %>% filter(lesion_id %in% nExams$lesion_id[nExams$exams ==1])

#########################################
# LESIONS WITH SINGLE TIMEPOINT
#########################################

for(p in unique(temp_1exam$patient_id)){
  
  ###############
  # track progress
  print(p)
  ###############
  
  temp_p <- temp_1exam %>% filter(patient_id == p)
  
  if(nrow(temp_p) < 1){
    next
  }
  
  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
    
    if(temp_l$lesion_id[1] %in% out$lesion_id){
      print(c('SKIPPING THIS ONE', l))
      next
    }
    
    patient_id_l <- temp_l$patient_id[1]
    lesion_id_l <- temp_l$les[1]
    
    biopsied_l <- 0
    dx_chal <- "benign"

    # only 1 timepoint to choose
    # image date (top of sorted table)
    imaging_day_l <- temp_l$imaging_day[1]

    fn <- temp_l$filename[1]
    
    out <- rbind(out, data.frame(filename = fn, patient_id = patient_id_l, imaging_day = imaging_day_l, lesion_id = lesion_id_l, biopsied = biopsied_l, benign_malignant=dx_chal, stringsAsFactors = F))
  }
}
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))

# Most of the lesions have a single timepoint... select them first for efficiency
temp_2exam <- temp_data %>% filter(lesion_id %in% nExams$lesion_id[nExams$exams >1])

#########################################
# LESIONS WITH MULTIPLE TIMEPOINTS
#########################################

for(p in unique(temp_2exam$patient_id)){
  
  ###############
  # track progress
  print(p)
  ###############
  
  temp_p <- temp_2exam %>% filter(patient_id == p)
  
  if(nrow(temp_p) < 1){
    next
  }
  
  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
    
    if(temp_l$lesion_id[1] %in% out$lesion_id){
      print(c('SKIPPING THIS ONE', l))
      next
    }
    
    patient_id_l <- temp_l$patient_id[1]
    lesion_id_l <- temp_l$les[1]
    
    biopsied_l <- 0
    dx_chal <- "benign"

    # melanoma image date
    p_melDates <- melanomasAll %>% filter(patient_id == patient_id_l) %>% pull(imaging_day)
    
    # if possible, only use dates after 2014
    after2014 <- temp_l %>% filter(as.Date(imaging_day) >= as.Date('2014-01-01'))
    if(nrow(after2014) > 0){
      temp_l <- after2014
    }
      
    
    date_ls <- unique(temp_l$imaging_day)
    
    date_l = 1
    differenceFromMel = abs(min(difftime(date_ls[date_l], p_melDates)))
    for(d in 1:length(date_ls)){
      date_lsd = date_ls[d]
      if(abs(min(difftime(date_ls[d], p_melDates))) < differenceFromMel){
        date_l = d
        differenceFromMel = min(difftime(date_ls[date_l], p_melDates))
      }
    }
    imaging_day_l <- date_ls[date_l]
    
    fn <- temp_l$filename[1]
    
    out <- rbind(out, data.frame(filename = fn, patient_id = patient_id_l, imaging_day = imaging_day_l, lesion_id = lesion_id_l, biopsied = biopsied_l, benign_malignant=dx_chal, stringsAsFactors = F))
  }
}
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#
#
# NON-BIOPSIED LESIONS - NON-MELANOMA PATIENT CLASS
#
# FOR EACH PATIENT WITH MULTIPLE TIMEPOINTS FOR AT LEAST ONE IMAGE, USE RANDOM SAMPLING TO 
# DECIDE ON A SET OF TIMEPOINTS THAT RESULTS IN A STANDARD DEVIATION CLOSEST TO THE
# MEAN STANDARD DEVAITON OF TIMEPOINTS IN THE MELANOMA PATIENT CLASS.
#
# WE HAVE SELECTED DATES CLOSEST TO THE PATIENTS' NEAREST MELANOMA DATE.
# UNFORTUNATELY, THE DISTRIBUTION OF MELNAOMA FROM A PATIENT'S AVERAGE DATE HAS THICKER TAILS THAN THE DISTRIBUTION
# OF A RANDOM SELECTED CONTEXTUAL LESION FROM THE PATIENT'S AVERAGE DATE.
#
# SELECT TIMEPOINTS IN THE NON-MELANOMA CLASS SO THAT THE DISTANCE FROM A
# RANDOM SELECTED IMAGE TO THE AVERAGE IS AS CLOSE TO THE SAME DISTRIBUTION IN BOTH PATIENT CLASSES.
#
# NEXT, ESTIMATE THE MEAN STANDARD DEVIATION OF A MEL PATIENT'S IMAGE DATES AND SET THAT AS THE TARGET VARIATION
# FOR THE BENIGN PATIENT CLASS
#
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


melSdDate = out %>% filter(patient_id %in% melPat_patient_id) %>% group_by(patient_id) %>% summarise(sdDate = sd(as.Date(imaging_day))) %>% filter(!(is.na(sdDate)))

meanSd <- mean(melSdDate$sdDate)
gammaMean <- mean(melSdDate$sdDate)
gammaVar <- var(melSdDate$sdDate)

gammaShape = gammaMean^2 / gammaVar
gammaRate = (gammaMean^2 / gammaVar) / gammaMean

# NEED VECTOR OF MELANOMA CLASS PATIENTS
used_lesion_id <- out$lesion_id
# UNBIOPSIED LESIONS OF THE BENIGN PATIENT CLASS
temp_data <- data %>% filter(!(patient_id %in% melPat_patient_id) & !(lesion_id %in% used_lesion_id))

# number of timepoints per lesion
nExams <- temp_data %>% group_by(lesion_id) %>% summarise(exams = length(unique(imaging_day)))
table(nExams$exams)
# Most of the lesions have a single timepoint... select them first for efficiency
temp_1exam <- temp_data %>% filter(lesion_id %in% nExams$lesion_id[nExams$exams ==1])

#########################################
# LESIONS WITH SINGLE TIMEPOINT
#########################################

for(p in unique(temp_1exam$patient_id)){
  
  ###############
  # track progress
  print(p)
  ###############
  
  temp_p <- temp_1exam %>% filter(patient_id == p)
  
  if(nrow(temp_p) < 1){
    next
  }
  
  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
    
    if(temp_l$lesion_id[1] %in% out$lesion_id){
      print(c('SKIPPING THIS ONE', l))
      next
    }
    
    patient_id_l <- temp_l$patient_id[1]
    lesion_id_l <- temp_l$les[1]
    
    biopsied_l <- 0
    dx_chal <- "benign"

    # only 1 timepoint to choose
    # image date (top of sorted table)
    imaging_day_l <- temp_l$imaging_day[1]

    
    fn <- temp_l$filename[1]
    
    out <- rbind(out, data.frame(filename = fn, patient_id = patient_id_l, imaging_day = imaging_day_l, lesion_id = lesion_id_l, biopsied = biopsied_l, benign_malignant=dx_chal, stringsAsFactors = F))
  }
}
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))

# Most of the lesions have a single timepoint... select them first for efficiency
temp_2exam <- temp_data %>% filter(lesion_id %in% nExams$lesion_id[nExams$exams >1])

#########################################
# LESIONS WITH MULTIPLE TIMEPOINTS
#########################################

for(p in unique(temp_2exam$patient_id)){
  
  ###############
  # track progress
  print(p)
  ###############
  
  temp_p <- temp_2exam %>% filter(patient_id == p)
  
  if(!(p %in% temp_2exam$patient_id)){
    print("this does not have any flexibility")
    next
  }
  
  if(nrow(temp_p) < 1){
    print('there is the mistake 2')
    next
  }

  # list of all possible dates to select from!!!
  # date list
  listLesDate <- vector(mode = "list", length = length(unique(temp_p$lesion_id)))
  names(listLesDate) <- unique(temp_p$lesion_id)
  # length of items in date list
  listLesDateItems <- c()
  
  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
    
    listLesDate[[l]] <- unique(temp_l$imaging_day)
    listLesDateItems[l] <- length(unique(temp_l$imaging_day))
  }
  
  # 10 randomly sampled vectors
  samples = 10
  sampDate <- vector(mode = 'list', length = samples)
  length(listLesDate)
  
  # random sample
  i = 1
  bestOne <- NA
  
  # variation goal = average variability of imageing timepoints in the melanoma class of patientss
  goalSd = meanSd
  
  while(i <= 10){
    i = i + 1
    
    # sample timepoints by random uniform
    undecidedDates <- c()
    
    for(l in names(listLesDateItems)){
      undecidedDates[l] <- as.character(listLesDate[[l]][sample(x = c(1:listLesDateItems[l]), size = 1)])
    }
    decidedDates <- out %>% filter(patient_id == p) %>% pull(imaging_day)
    
    # calculate patient's variability in imageing date
    sdDates <- sd(as.Date(c(undecidedDates, decidedDates)))
    if(is.na(bestOne)){
      bestOne <- abs(sdDates - goalSd)
      bestDates <- undecidedDates
    } else{
      thisOne <- abs(sdDates - goalSd)
      if(thisOne < bestOne){
        bestOne <- thisOne
        bestDates <- undecidedDates
      }
    }
  }
  
  
  for(l in unique(temp_p$lesion_id)){
    temp_l <- temp_p %>% filter(lesion_id == l) %>% arrange(-as.numeric(imaging_day), filename)
      
        
    if(l %in% out$lesion_id){
      print(c('SKIPPING THIS ONE', l))
      next
    }
    
    
    patient_id_l <- temp_l$patient_id[1]
    lesion_id_l <- temp_l$les[1]
    
    biopsied_l <- 0
    dx_chal <- "benign"

    imaging_day_l <- bestDates[l]
    
    fn <- temp_l$filename[1]
    
    out <- rbind(out, data.frame(filename = fn, patient_id = patient_id_l, imaging_day = imaging_day_l, lesion_id = lesion_id_l, biopsied = biopsied_l, benign_malignant=dx_chal, stringsAsFactors = F))
  }
}
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))


# FILTER OUT PATIENTS WHO DON'T MEET THE LESION COUNT REQUIREMENT
nLes <- out %>% group_by(patient_id) %>% summarise(nLes = length(unique(lesion_id))) %>% filter(nLes >= min_les)
out <- out %>% filter(patient_id %in% nLes$patient_id)
print(paste('Number of rows = ', nrow(out), ', number of unique lesions = ', length(unique(out$lesion_id)), sep = ''))


###### WRITE THE TABLES
colnames(out)[6,7] <- c("biopsied", "")
write.csv(out, paste("Timepoint_perLesion_", thisSite, "_", todaysDate,".csv", sep = ''), row.names = F)






##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#
#
# PLOTS TO REPORT NUMBER OF LESIONS AND VARIATION IN TIMEPOINTS BETWEEN CLASSES
#
#
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

MASTER_mel <- out
theme <- theme_bw()

# Plot distribution of # lesions per patient
temp <- MASTER_mel %>% mutate(MELPAT = ifelse(patient_id %in% melPat_patient_id, 1, 0)) %>% group_by(patient_id, MELPAT) %>% summarise(lesions = length(unique(lesion_id)))
temp_mel <- temp %>% filter(MELPAT == 1) %>% mutate(class = 'mel')
temp_ctrl <- temp %>% filter(MELPAT == 0) %>% mutate(class = 'ctrl')
temp_merge <- rbind(x = temp_mel, y = temp_ctrl)
ggplot(data = temp_merge, aes(color = class)) + stat_ecdf(aes(lesions), size = .9) + 
  theme_bw(base_size = 12) + 
  labs(title = 'CUMULATIVE DISTRIBUTION OF NUMBER OF LESIONS PER PATIENT', x = '# of Lesions', y = 'P(X <= x)')

# Plot histograms for patient level SD of dates
melSdDate = out %>% filter(patient_id %in% melPat_patient_id) %>% group_by(patient_id) %>% summarise(sdDate = sd(as.Date(imaging_day))) %>% mutate(class = 'MelPat')
ctrlSdDate = out %>% filter(!(patient_id %in% melPat_patient_id)) %>% group_by(patient_id) %>% summarise(sdDate = sd(as.Date(imaging_day))) %>% mutate(class = 'CtrlPat')
sdDate = rbind(melSdDate, ctrlSdDate)
ggplot(sdDate) + facet_wrap(~class) + geom_density(aes(sdDate), fill = 'grey') +
  labs(title = "VARIATION IN TIMEPOINTS WITHIN PATIENT", x = "Standard Deviation of Patient Timepoints", y = "Frequency") +
  theme_bw(base_size = 12) +
  theme(text = element_text(size=12))



