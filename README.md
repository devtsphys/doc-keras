# Python Keras Complete Reference Card

## Table of Contents

1. [Core Imports](#core-imports)
1. [Model Architecture](#model-architecture)
1. [Layers Reference](#layers-reference)
1. [Activation Functions](#activation-functions)
1. [Optimizers](#optimizers)
1. [Loss Functions](#loss-functions)
1. [Metrics](#metrics)
1. [Callbacks](#callbacks)
1. [Data Preprocessing](#data-preprocessing)
1. [Model Training & Evaluation](#model-training--evaluation)
1. [Advanced Techniques](#advanced-techniques)
1. [Complete Examples](#complete-examples)

-----

## Core Imports

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

-----

## Model Architecture

### Sequential Model

```python
# Method 1: Layer by layer
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

# Method 2: List of layers
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

### Functional API

```python
# More flexible for complex architectures
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing (Custom Models)

```python
class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)
```

-----

## Layers Reference

|Layer Type        |Class               |Common Parameters                       |Example                               |
|------------------|--------------------|----------------------------------------|--------------------------------------|
|**Dense**         |`Dense`             |`units, activation, use_bias`           |`Dense(64, activation='relu')`        |
|**Convolutional** |`Conv2D`            |`filters, kernel_size, strides, padding`|`Conv2D(32, (3,3), activation='relu')`|
|**Pooling**       |`MaxPooling2D`      |`pool_size, strides, padding`           |`MaxPooling2D((2,2))`                 |
|**Recurrent**     |`LSTM`              |`units, return_sequences, dropout`      |`LSTM(50, return_sequences=True)`     |
|**Normalization** |`BatchNormalization`|`axis, momentum, epsilon`               |`BatchNormalization()`                |
|**Regularization**|`Dropout`           |`rate, noise_shape, seed`               |`Dropout(0.5)`                        |
|**Reshaping**     |`Flatten`           |None                                    |`Flatten()`                           |
|**Embedding**     |`Embedding`         |`input_dim, output_dim, input_length`   |`Embedding(1000, 64)`                 |

### Core Layers

```python
# Dense (Fully Connected)
layers.Dense(units=64, activation='relu', use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')

# Activation Layer
layers.Activation('relu')  # or 'sigmoid', 'tanh', 'softmax'

# Dropout (Regularization)
layers.Dropout(rate=0.5, noise_shape=None, seed=None)

# Batch Normalization
layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
```

### Convolutional Layers

```python
# 2D Convolution
layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
              padding='valid', activation='relu', input_shape=(28,28,1))

# Max Pooling
layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')

# Average Pooling
layers.AveragePooling2D(pool_size=(2,2))

# Global Average Pooling
layers.GlobalAveragePooling2D()

# 1D Convolution (for sequences)
layers.Conv1D(filters=64, kernel_size=3, activation='relu')
```

### Recurrent Layers

```python
# LSTM
layers.LSTM(units=50, activation='tanh', recurrent_activation='sigmoid',
            return_sequences=False, return_state=False, dropout=0.0)

# GRU
layers.GRU(units=50, activation='tanh', return_sequences=False)

# Simple RNN
layers.SimpleRNN(units=50, activation='tanh', return_sequences=False)

# Bidirectional RNN
layers.Bidirectional(layers.LSTM(50, return_sequences=True))
```

### Reshaping Layers

```python
# Flatten
layers.Flatten()

# Reshape
layers.Reshape(target_shape=(7, 7, 64))

# Permute dimensions
layers.Permute(dims=(2, 1))
```

-----

## Activation Functions

|Function     |Formula                 |Use Case                       |Example                      |
|-------------|------------------------|-------------------------------|-----------------------------|
|**ReLU**     |`max(0, x)`             |Hidden layers, CNN             |`activation='relu'`          |
|**Sigmoid**  |`1/(1+e^-x)`            |Binary classification output   |`activation='sigmoid'`       |
|**Softmax**  |`e^xi/Σe^xj`            |Multi-class classification     |`activation='softmax'`       |
|**Tanh**     |`(e^x-e^-x)/(e^x+e^-x)` |Hidden layers (centered output)|`activation='tanh'`          |
|**LeakyReLU**|`max(αx, x)`            |Avoid dying ReLU               |`layers.LeakyReLU(alpha=0.1)`|
|**ELU**      |`x if x>0 else α(e^x-1)`|Smooth negative values         |`activation='elu'`           |
|**Swish**    |`x * sigmoid(x)`        |Modern alternative to ReLU     |`activation='swish'`         |

```python
# Custom activation functions
def custom_activation(x):
    return keras.backend.sin(x)

# Using advanced activations
layers.LeakyReLU(alpha=0.1)
layers.ELU(alpha=1.0)
layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

-----

## Optimizers

|Optimizer  |Best For        |Key Parameters                 |Example                                |
|-----------|----------------|-------------------------------|---------------------------------------|
|**SGD**    |Simple problems |`learning_rate, momentum`      |`optimizers.SGD(lr=0.01, momentum=0.9)`|
|**Adam**   |General purpose |`learning_rate, beta_1, beta_2`|`optimizers.Adam(lr=0.001)`            |
|**RMSprop**|RNNs            |`learning_rate, rho, epsilon`  |`optimizers.RMSprop(lr=0.001)`         |
|**AdaGrad**|Sparse data     |`learning_rate, epsilon`       |`optimizers.Adagrad(lr=0.01)`          |
|**Adamax** |Large embeddings|`learning_rate, beta_1, beta_2`|`optimizers.Adamax(lr=0.002)`          |

```python
# Optimizer examples with parameters
optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7)
optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-7)
```

-----

## Loss Functions

|Loss Function                      |Problem Type                |Output Activation|Example                                 |
|-----------------------------------|----------------------------|-----------------|----------------------------------------|
|**Categorical Crossentropy**       |Multi-class                 |Softmax          |`losses.categorical_crossentropy`       |
|**Sparse Categorical Crossentropy**|Multi-class (integer labels)|Softmax          |`losses.sparse_categorical_crossentropy`|
|**Binary Crossentropy**            |Binary classification       |Sigmoid          |`losses.binary_crossentropy`            |
|**Mean Squared Error**             |Regression                  |Linear           |`losses.mean_squared_error`             |
|**Mean Absolute Error**            |Regression (robust)         |Linear           |`losses.mean_absolute_error`            |
|**Huber**                          |Regression (outliers)       |Linear           |`losses.Huber()`                        |

```python
# Loss function examples
losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
losses.binary_crossentropy(y_true, y_pred, from_logits=False)
losses.mean_squared_error(y_true, y_pred)
losses.mean_absolute_error(y_true, y_pred)

# Custom loss function
def custom_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true))
```

-----

## Metrics

|Metric       |Type          |Description                                        |Example                      |
|-------------|--------------|---------------------------------------------------|-----------------------------|
|**Accuracy** |Classification|Fraction of correct predictions                    |`metrics.accuracy`           |
|**Precision**|Classification|True positives / (True + False positives)          |`metrics.Precision()`        |
|**Recall**   |Classification|True positives / (True positives + False negatives)|`metrics.Recall()`           |
|**F1 Score** |Classification|Harmonic mean of precision and recall              |`metrics.F1Score()`          |
|**AUC**      |Classification|Area under ROC curve                               |`metrics.AUC()`              |
|**MSE**      |Regression    |Mean squared error                                 |`metrics.mean_squared_error` |
|**MAE**      |Regression    |Mean absolute error                                |`metrics.mean_absolute_error`|

```python
# Metrics examples
metrics.SparseCategoricalAccuracy(name='accuracy')
metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
metrics.Precision(name='precision')
metrics.Recall(name='recall')
metrics.AUC(name='auc')
metrics.MeanSquaredError(name='mse')
metrics.RootMeanSquaredError(name='rmse')
```

-----

## Callbacks

|Callback             |Purpose             |Key Parameters                           |Example                               |
|---------------------|--------------------|-----------------------------------------|--------------------------------------|
|**ModelCheckpoint**  |Save best model     |`filepath, monitor, save_best_only`      |`callbacks.ModelCheckpoint('best.h5')`|
|**EarlyStopping**    |Stop training early |`monitor, patience, restore_best_weights`|`callbacks.EarlyStopping(patience=5)` |
|**ReduceLROnPlateau**|Reduce learning rate|`monitor, factor, patience`              |`callbacks.ReduceLROnPlateau()`       |
|**TensorBoard**      |Visualization       |`log_dir, histogram_freq`                |`callbacks.TensorBoard('./logs')`     |
|**CSVLogger**        |Log metrics to CSV  |`filename, separator`                    |`callbacks.CSVLogger('training.log')` |

```python
# Callback examples
checkpoint = callbacks.ModelCheckpoint(
    filepath='model_{epoch:02d}_{val_loss:.2f}.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

tensorboard = callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)
```

-----

## Data Preprocessing

### Image Data

```python
# ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Flow from directory
train_generator = datagen.flow_from_directory(
    'train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Text Data

```python
# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
padded_sequences = pad_sequences(
    sequences, 
    maxlen=100, 
    padding='post', 
    truncating='post'
)
```

### Utilities

```python
# One-hot encoding
y_categorical = to_categorical(y, num_classes=10)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalization
X_normalized = X / 255.0  # For images
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

-----

## Model Training & Evaluation

### Model Compilation

```python
model.compile(
    optimizer='adam',           # or optimizer object
    loss='categorical_crossentropy',  # or loss function
    metrics=['accuracy', 'precision', 'recall']
)
```

### Model Training

```python
# Basic training
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Training with generator
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint, early_stop]
)
```

### Model Evaluation

```python
# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Prediction probabilities
probabilities = model.predict_proba(X_test)
```

### Model Saving & Loading

```python
# Save entire model
model.save('my_model.h5')
model.save('my_model')  # SavedModel format

# Save only weights
model.save_weights('model_weights.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Load weights
model.load_weights('model_weights.h5')
```

-----

## Advanced Techniques

### Transfer Learning

```python
# Load pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])
```

### Custom Layers

```python
class CustomLayer(layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Learning Rate Scheduling

```python
# Step decay
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

lr_scheduler = callbacks.LearningRateScheduler(step_decay)

# Exponential decay
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)
```

### Regularization Techniques

```python
# L1 and L2 regularization
layers.Dense(64, activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            activity_regularizer=keras.regularizers.l1(0.01))

# Dropout
layers.Dropout(0.5)

# Batch Normalization
layers.BatchNormalization()

# Early Stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
```

-----

## Complete Examples

### 1. Image Classification (CNN)

```python
# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[
        callbacks.EarlyStopping(patience=3),
        callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

### 2. Text Classification (LSTM)

```python
# Build LSTM model
model = keras.Sequential([
    layers.Embedding(vocab_size, 100, input_length=max_length),
    layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[callbacks.EarlyStopping(patience=3)]
)
```

### 3. Regression Model

```python
# Build regression model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1)  # No activation for regression
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error', 'mean_squared_error']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

-----

## Best Practices & Tips

### Model Design

- Start simple, then add complexity
- Use batch normalization after dense/conv layers
- Apply dropout before final layers
- Use appropriate activation functions for output layer

### Training Tips

- Monitor both training and validation metrics
- Use callbacks for early stopping and model checkpointing
- Implement learning rate scheduling
- Use data augmentation for images

### Debugging

- Check data shapes and types
- Verify loss function matches problem type
- Monitor gradient flow with TensorBoard
- Use `model.summary()` to inspect architecture

### Performance Optimization

- Use appropriate batch sizes (powers of 2)
- Enable mixed precision training for large models
- Use `tf.data` for efficient data pipelines
- Consider model quantization for deployment

-----

## Common Model Patterns

|Task                          |Architecture   |Loss Function               |Metrics                          |Activation    |
|------------------------------|---------------|----------------------------|---------------------------------|--------------|
|**Binary Classification**     |Dense layers   |`binary_crossentropy`       |`accuracy`, `precision`, `recall`|`sigmoid`     |
|**Multi-class Classification**|Dense layers   |`categorical_crossentropy`  |`accuracy`, `top_k_accuracy`     |`softmax`     |
|**Image Classification**      |CNN            |`categorical_crossentropy`  |`accuracy`                       |`softmax`     |
|**Object Detection**          |YOLO/R-CNN     |Custom                      |`mAP`                            |Multiple      |
|**Text Classification**       |RNN/LSTM       |`binary_crossentropy`       |`accuracy`, `f1_score`           |`sigmoid`     |
|**Regression**                |Dense layers   |`mse`, `mae`                |`mae`, `mse`                     |None          |
|**Time Series**               |LSTM/GRU       |`mse`, `mae`                |`mae`, `mse`                     |None          |
|**Autoencoder**               |Encoder-Decoder|`mse`, `binary_crossentropy`|`mse`                            |`sigmoid`/None|

This reference card covers the essential Keras functionality you’ll need for most deep learning projects. Keep it handy for quick lookups during development!