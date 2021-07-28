import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sixplot(var, auc, v_auc, pr_auc, v_pr_auc, prec, v_prec, rec, v_rec):
    """
        Function to produce the plots for accuracy, ROC-AUC, PR-AUC, recall, precision, and loss scores for the training and validation sets
        `var` is the variable that we set `model.fit` or `model.fit_generator`
        `auc`, `pr_auc`, `rec`, `prec`, etc. must be strings with quotes, i.e. `'auc_4'` or `'precision_8'`
    """

    f, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].plot(var.history['accuracy'], label = 'train')
    axs[0, 0].plot(var.history['val_accuracy'], label = 'validation')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()
    axs[0, 0].set_title('Accuracy Scores')
    
    axs[0, 1].plot(var.history[auc], label = 'train')
    axs[0, 1].plot(var.history[v_auc], label = 'validation')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('ROC-AUC')
    axs[0, 1].legend()
    axs[0, 1].set_title('ROC/AUC Scores')
    
    axs[0, 2].plot(var.history[pr_auc], label = 'train')
    axs[0, 2].plot(var.history[v_pr_auc], label = 'validation')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('PR-AUC')
    axs[0, 2].legend()
    axs[0, 2].set_title('PR/AUC Scores')
    
    axs[1, 0].plot(var.history[prec], label = 'train')
    axs[1, 0].plot(var.history[v_prec], label = 'validation')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].set_title('Precision Scores')
    
    axs[1, 1].plot(var.history[rec], label = 'train')
    axs[1, 1].plot(var.history[v_rec], label = 'validation')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()
    axs[1, 1].set_title('Recall Scores')
    
    axs[1, 2].plot(var.history['loss'], label = 'train')
    axs[1, 2].plot(var.history['val_loss'], label = 'validation')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].legend()
    axs[1, 2].set_title('Loss Scores')

    f.tight_layout(h_pad=5, w_pad=5)

def sixplot2(var, auc, v_auc, pr_auc, v_pr_auc, prec, v_prec, rec, v_rec):
    """
        Function to produce the plots for accuracy, ROC-AUC, PR-AUC, recall, precision, and loss scores for the training and validation sets
        `var` is the variable that we set `model.fit` or `model.fit_generator`
        `auc`, `pr_auc`, `rec`, `prec`, etc. must be strings with quotes, i.e. `'auc_4'` or `'precision_8'`
    """

    f, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].plot(var['accuracy'], label = 'train')
    axs[0, 0].plot(var['val_accuracy'], label = 'validation')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()
    axs[0, 0].set_title('Accuracy Scores')
    
    axs[0, 1].plot(var[auc], label = 'train')
    axs[0, 1].plot(var[v_auc], label = 'validation')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('ROC-AUC')
    axs[0, 1].legend()
    axs[0, 1].set_title('ROC/AUC Scores')
    
    axs[0, 2].plot(var[pr_auc], label = 'train')
    axs[0, 2].plot(var[v_pr_auc], label = 'validation')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('PR-AUC')
    axs[0, 2].legend()
    axs[0, 2].set_title('PR/AUC Scores')
    
    axs[1, 0].plot(var[prec], label = 'train')
    axs[1, 0].plot(var[v_prec], label = 'validation')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].set_title('Precision Scores')
    
    axs[1, 1].plot(var[rec], label = 'train')
    axs[1, 1].plot(var[v_rec], label = 'validation')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()
    axs[1, 1].set_title('Recall Scores')
    
    axs[1, 2].plot(var['loss'], label = 'train')
    axs[1, 2].plot(var['val_loss'], label = 'validation')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].legend()
    axs[1, 2].set_title('Loss Scores')

    f.tight_layout(h_pad=5, w_pad=5)


def make_confusion_matrix(y, y_pred):
    cnf = confusion_matrix(y, y_pred)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})


def train_val_metrics(epochs, model_training):
    epochs = range(1, epochs+1)
    metrics = model_training.history
    train_loss = metrics['loss']
    train_acc = metrics['accuracy']
    val_loss = metrics['val_loss']
    val_acc = metrics['val_accuracy']
    
    ax = plt.subplot(211)
    train, = ax.plot(epochs, train_loss)
    val, = ax.plot(epochs, val_loss)
    ax.legend([train, val], ['training', 'validation'])
    ax.set(xlabel='epochs', ylabel='categorical cross-entropy loss')

    ax2 = plt.subplot(212)
    train2, = ax2.plot(epochs, train_acc)
    val2, = ax2.plot(epochs, val_acc)
    ax2.legend([train2, val2], ['training', 'validation'])
    ax2.set(xlabel='epochs', ylabel='accuracy')

    plt.tight_layout(h_pad=5, w_pad=5)


def train_val_metrics2(epochs, metrics):
    epochs = range(1, epochs+1)
    train_loss = metrics['loss']
    train_acc = metrics['accuracy']
    val_loss = metrics['val_loss']
    val_acc = metrics['val_accuracy']
    
    ax = plt.subplot(211)
    train, = ax.plot(epochs, train_loss)
    val, = ax.plot(epochs, val_loss)
    ax.legend([train, val], ['training', 'validation'])
    ax.set(xlabel='epochs', ylabel='loss')

    ax2 = plt.subplot(212)
    train2, = ax2.plot(epochs, train_acc)
    val2, = ax2.plot(epochs, val_acc)
    ax2.legend([train2, val2], ['training', 'validation'])
    ax2.set(xlabel='epochs', ylabel='accuracy')

    plt.tight_layout(h_pad=5, w_pad=5)


## courtesy of flaboss on Github
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df


def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n



import itertools

def draw_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')