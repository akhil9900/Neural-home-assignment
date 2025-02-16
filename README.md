# Neural-home-assignment
Akhil Dondapati
700756446

1.Tensor Manipulations & Reshaping

Create a Random Tensor
 
tensor = tf.random.normal(shape=(4, 6))
Generates a random tensor with shape (4, 6) from a normal distribution.

Find Rank and Shape
 
rank = tf.rank(tensor)
shape = tf.shape(tensor)
tf.rank(tensor): Returns the number of dimensions (rank) of the tensor.
tf.shape(tensor): Returns the shape of the tensor as a TensorFlow tensor.

Reshape and Transpose
 
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
tf.reshape: Changes the shape of the tensor to (2, 3, 4).
tf.transpose: Reorders the tensor's dimensions to (3, 2, 4).

Broadcasting and Addition
 
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
broadcasted_tensor = tf.broadcast_to(small_tensor, transposed_tensor.shape)
result_tensor = transposed_tensor + broadcasted_tensor
A smaller tensor of shape (1, 4) is broadcasted to match (3, 2, 4).
Element-wise addition is performed between the transposed tensor and the broadcasted tensor.
Broadcasting Explanation
Broadcasting expands smaller tensors to match the shape of larger tensors for element-wise operations.

2. Loss Functions & Hyperparameter Tuning
Define True Values and Predictions
 
y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])
y_true: Represents the actual labels.
y_pred: Represents the predicted values.

Compute MSE and CCE Losses
 
mse_loss = tf.keras.losses.MeanSquaredError()
cce_loss = tf.keras.losses.BinaryCrossentropy()
MSE (Mean Squared Error): Measures the average squared difference between actual and predicted values.
CCE (Binary Cross-Entropy): Measures the performance of a classification model with probability-based predictions.

Modify Predictions and Recompute Losses
 
y_pred_modified = tf.constant([0.85, 0.15, 0.75, 0.25])
Predictions are slightly modified to analyze how loss values change.

Plot Loss Values
 
plt.bar(labels, losses, color=['blue', 'orange', 'green', 'red'])
A bar chart is created to compare loss values for MSE and CCE before and after modifying predictions.

3. Train a Model with Different Optimizers

Load MNIST Dataset
 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Loads the MNIST dataset and normalizes the pixel values by dividing them by 255.

Define a Simple Model
 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
A simple feedforward neural network with:
Flatten Layer: Converts the input into a 1D array.
Dense Layer (128 units, ReLU activation): A hidden layer.
Dense Layer (10 units, Softmax activation): The output layer for classification.

Train with Adam Optimizer
 
model_adam.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
Trains the model using the Adam optimizer for 5 epochs.

Train with SGD Optimizer
 
model_sgd.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
Trains the same model using the SGD optimizer for comparison.

Plot Accuracy Trends
 
plt.plot(history_adam.history['val_accuracy'], label='Adam')
plt.plot(history_sgd.history['val_accuracy'], label='SGD')
plt.legend()
plt.show()
Plots the validation accuracy trends for Adam and SGD optimizers.

4. Train a Neural Network and Log to TensorBoard
Load MNIST Dataset

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Loads the MNIST dataset and normalizes the images.

Define a Simple Model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
A simple neural network is defined with a Flatten layer, a hidden ReLU layer, and a softmax output layer.

Compile the Model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Uses the Adam optimizer and sparse categorical cross-entropy loss.

Enable TensorBoard Logging

import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
TensorBoard logging is set up with a unique timestamped directory.

Train the Model with TensorBoard Logging

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
Trains the model for 5 epochs while logging training details for visualization in TensorBoard.

Launch TensorBoard
Run the following command in the terminal:

tensorboard --logdir logs/fit
This command starts TensorBoard, allowing visualization of training metrics.


