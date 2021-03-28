import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MergedGenerators(Sequence):
    def __init__(self, batch_size, generators=[], sub_batch_size=[]):
        self.generators = generators
        self.sub_batch_size = sub_batch_size
        self.batch_size = batch_size

    def __len__(self):
        return int(
            sum([(len(self.generators[idx]) * self.sub_batch_size[idx])
                 for idx in range(len(self.sub_batch_size))]) /
            self.batch_size)

    def __getitem__(self, index):
        """Getting items from the generators and packing them"""

        X_batch = []
        Y_batch = []
        for generator in self.generators:
            if generator.class_mode is None:
                x1 = generator[index % len(generator)]
                X_batch = [*X_batch, *x1]

            else:
                x1, y1 = generator[index % len(generator)]
                X_batch = [*X_batch, *x1]
                Y_batch = [*Y_batch, *y1]

        if self.generators[0].class_mode is None:
            return np.array(X_batch)
        return np.array(X_batch), np.array(Y_batch)


def build_datagenerator(dir1=None, dir2=None, batch_size=32):
    n_images_in_dir1 = sum([len(files) for r, d, files in os.walk(dir1)])
    n_images_in_dir2 = sum([len(files) for r, d, files in os.walk(dir2)])
    generator1_batch_size = int((n_images_in_dir1 * batch_size) /
                                (n_images_in_dir1 + n_images_in_dir2))
    generator2_batch_size = batch_size - generator1_batch_size
    generator1 = ImageDataGenerator(rescale=1./255).flow_from_directory(
        dir1,
        target_size=(224, 224),
        batch_size=generator1_batch_size,
        class_mode='binary'
    )
    generator2 = ImageDataGenerator(rescale=1./255).flow_from_directory(
        dir2,
        target_size=(224, 224),
        batch_size=generator2_batch_size,
        class_mode='binary'
    )
    return MergedGenerators(
        batch_size,
        generators=[generator1, generator2],
        sub_batch_size=[generator1_batch_size, generator2_batch_size])


def sixplot(var, auc, v_auc, pr_auc, v_pr_auc, prec, v_prec, rec, v_rec):
    """
        Function to produce the plots for accuracy, ROC-AUC, PR-AUC, recall, precision, and loss scores for the training and validation sets
        `var` is the variable that we set `model.fit` or `model.fit_generator`
        `auc`, `pr_auc`, `rec`, `prec`, etc. must be strings with quotes, i.e. `'auc_4'` or `'precision_8'`
    """
    plt.style.use('fivethirtyeight')

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

def precision(y, y_hat):
    """
        Function to calculate precision score
        where y is original labels and y_hat is predicted labels
    """
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fp = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 1])
    if tp + fp == 0:
        return np.nan
    else: 
        return tp / float(tp + fp)

def recall(y, y_hat):
    """
        Function to calculate recall score
        where y is original labels and y_hat are predicted labels
    """
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fn = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 0])
    return tp / float(tp + fn)
    if tp + fn == 0:
        return np.nan
    else:
        return tp / float(tp + fn)

def aps(X, y, model):
    """
        Function to calculate PR AUC Score based on predict_proba(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

def get_metrics(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    """
        Function to get training and validation F1, recall, precision, PR AUC scores
        Instantiate model and pass the model into function
        Pass X_train, y_train, X_val, Y_val datasets
        Pass in calculated model.predict(X) for y_pred
    """    
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps(X_tr, y_tr, model)
    aps_val = aps(X_val, y_val, model)
    
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)

def make_confusion_matrix(y, y_pred):
    cnf = confusion_matrix(y, y_pred)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})