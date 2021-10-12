# Building a Flask app on Image Classification of Dog/Cat Dataset implemented by Convolutional Neural Network (CNN)

### Phuong T.M. Chu & Minh H. Nguyen
This is the project that we finished after the 6th week of studying **Machine Learning**.

<p align="center">
  <img width="760" height="400" src="https://miro.medium.com/max/1838/1*oB3S5yHHhvougJkPXuc8og.gif">
</p>




## INTRODUCTION
### 1. The Dog vs. Cat Dataset
**Dogs vs. Cats** [dataset](https://www.kaggle.com/c/dogs-vs-cats/data) provided by  Microsoft Research contains 25,000 images of dogs and cats with the labels 
* 1 = dog
* 0 = cat 

### 2. Project goals
- Building a **deep neural network** using **TensorFlow** to classify dogs and cats images.

- Making a **Flask application** so user can upload their photos and receive the prediction.

### 3. Project plan

During this project, we need to answer these following questions:

**A. Build the model**
- How to import the data
- How to preprocess the images
- How to create a model
- How to train the model with the data
- How to export the model
- How to import the model
    
**B. Build the Flask app**

**Front end**
- HTML
    - How to connect frontend to backend
    - How to draw a number on HTML
    - How to make UI looks good

**Back end**
- Flask
    - How to set up Flask
    - How to handle backend error
    - How to make real-time prediction
    - Combine the model with the app


## SETUP ENVIRONMENT
* In order to run our model on a Flask application locally, you need to clone this repository and then set up the environment by these following commands:

```shell
python3 -m pip install --user pipx
python3 -m pipx ensurepath

pipx install pipenv

# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```
* On the Terminal, use these commands:
```
# enter the environment
pipevn shell
pipenv graph
set FLASK_APP=app.py
set FLASK_ENV=development
export FLASK_DEBUG=1
flask run
```
* If you have error `ModuleNotFoundError: No module named 'tensorflow'` then use
```
pipenv install tensorflow==2.0.0beta-1
```
* If  `* Debug mode: off` then use
```
export FLASK_DEBUG=1
```

* Run the model by 

```shell
pipenv run flask run
```

* If you want to exit `pipenv shell`, use `exit`

## HOW IT WORK: CONVOLUTIONAL NEURAL NETWORK (CNN)

> In deep learning, a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. (Wiki)

For this project, we used **pre-trained model [MobileNetV2](https://keras.io/applications/#mobilenetv2)** from keras. MobileNetV2 is a model that was trained on a large dataset to solve a **similar problem to this project**, so it will help us to save lots of time on buiding low-level layers and focus on the application.

***Note:** You can learn more about CNN architecture [here](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)*  

![](https://www.datascience.com/hs-fs/hubfs/CNN%202.png?width=650&name=CNN%202.png)

### 1. Load and preprocess images:

- Import **path, listdir** from **os library**.
- Find and save all the image's path to **all_image_paths**. (note that our images is in folder `train`). 

```python 
all_image_path = [path.join('train', p) for p in listdir('train') if isfile(path.join('train', p))]
```

- Define a function to load and preprocess image from path:

```python
def load_and_preprocess_image(path):
    file = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file , channels=3)
    image = tf.image.resize(image, [192, 192]) # resize all images to the same size.
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range
    return image
```

- Load and preprocess all images which path is in **all_image_path**:

```python
all_images = [load_and_preprocess_image(path) for path in all_image_path]
```
- Save all image labels in **all_image_labels**:

```python
dict = {'cat': 0, 'dog': 1}

# path.split('.')[0][-3:] return the name of the image ('dog' or 'cat')
labels = [path.split('.')[0][-3:] for path in all_image_path] 

# Transfer name-labels to number-labels:
all_image_labels = [dict[label] for label in labels]
```

- To implement batch training, we put the images and labels into Tensorflow dataset:

```python
ds = tf.data.Datasets.from_tensor_slices((all_images, all_image_labels))
```

### 2. Building CNN model: 

The CNN model contain **MobileNetV2, Pooling, fully-connected hidden layer and Output layer**.

- First we create **mobile_net** as an instance of **MobileNetV2**:

```python
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False # this told the model not to train the mobile_net.
```

- Then we build CNN model:

```python
cnn_model = keras.models.Sequential([
    mobile_net, # mobile_net is low-level layers
    keras.layers.GlobalAveragePooling2D(), 
    keras.layers.Flatten(), 
    keras.layers.Dense(64, activation="relu"), # fully-connected hidden layer 
    keras.layers.Dense(2, activation="softmax") # output layer
])
```

### 3. Training model:

We almost there! But before training our **cnn_model**, we need to implement batch to the training data so that the model will train faster.

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = ds.shuffle(buffer_size = len(all_image_labels))
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```

Now we train the model.

```python
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
```

```python
steps_per_epoch=tf.math.ceil(len(all_image_dirs)/BATCH_SIZE).numpy()
cnn_model.fit(train_ds, epochs=2, steps_per_epoch=steps_per_epoch)
```
After training, save the model for later use.
```python
cnn_model.save('my_model.h5')
```

## MODEL PERFOMANCE SUMARY
Our model has the accuracy of **97.79 %** for the train dataset and **97.32 %** for the test dataset. 

## FLASH APPLICATION
### Homepage
![](https://i.imgur.com/4CGnSNo.png)
### Example of results

![](https://i.imgur.com/PJ4f39B.png)

![](https://i.imgur.com/59cKpJQ.png)



## CONCLUSION

We successfully **built a deep neural network model** by implementing **Convolutional Neural Network (CNN)** to classify dog and cat images with very high accuracy **97.32 %**.
In addition, we also **built a Flask application** so user can upload their images and classify easily.
