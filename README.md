# Neural-home-assignment
Akhil Dondapati
700756446

**1. Tensor Manipulations & Reshaping**
Code Explanation
Create a Random Tensor:
tensor = tf.random.normal(shape=(4, 6))
A random tensor of shape (4, 6) is created using tf.random.normal. This generates values from a normal distribution.

Find Rank and Shape:
rank = tf.rank(tensor)
shape = tf.shape(tensor)
tf.rank gives the number of dimensions (rank) of the tensor.

tf.shape returns the shape of the tensor as a TensorFlow tensor.

Reshape and Transpose:
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
tf.reshape changes the tensor's shape to (2, 3, 4).

tf.transpose reorders the dimensions of the tensor to (3, 2, 4).

Broadcasting and Addition:
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
broadcasted_tensor = tf.broadcast_to(small_tensor, transposed_tensor.shape)
result_tensor = transposed_tensor + broadcasted_tensor
A smaller tensor of shape (1, 4) is broadcasted to match the shape of the larger tensor (3, 2, 4).

Element-wise addition is performed between the broadcasted tensor and the transposed tensor.

Broadcasting Explanation:

Broadcasting automatically expands smaller tensors to match the shape of larger tensors for element-wise operations.

**2. Loss Functions & Hyperparameter Tuning**
Code Explanation
Define True Values and Predictions:

y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])
y_true represents the true labels, and y_pred represents the model's predictions.

Compute MSE and CCE Losses:

mse_loss = tf.keras.losses.MeanSquaredError()
cce_loss = tf.keras.losses.BinaryCrossentropy()
Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses are computed using TensorFlow's built-in loss functions.

Modify Predictions and Recompute Losses:

y_pred_modified = tf.constant([0.85, 0.15, 0.75, 0.25])
Predictions are slightly modified, and the losses are recomputed to observe how they change.

Plot Loss Values:

plt.bar(labels, losses, color=['blue', 'orange', 'green', 'red'])
A bar chart is plotted to compare the loss values for MSE and CCE before and after modifying predictions.

**3. Train a Model with Different Optimizers**
Code Explanation
Load MNIST Dataset:

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
The MNIST dataset is loaded and normalized by dividing pixel values by 255.

Define a Simple Model:

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
A simple neural network with one hidden layer (128 units) and an output layer (10 units) is defined.

Train with Adam Optimizer:

model_adam.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
The model is compiled with the Adam optimizer and trained for 5 epochs.

Train with SGD Optimizer:

model_sgd.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
The same model is compiled with the SGD optimizer and trained for 5 epochs.

Plot Accuracy Trends:

plt.plot(history_adam.history['val_accuracy'], label='Adam')
plt.plot(history_sgd.history['val_accuracy'], label='SGD')
Validation accuracy trends for Adam and SGD are plotted for comparison.

**4. Train a Neural Network and Log to TensorBoard**
Code Explanation
Load MNIST Dataset:

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
The MNIST dataset is loaded and normalized.

Define a Simple Model:

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
A simple neural network with one hidden layer and an output layer is defined.

Compile the Model:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
The model is compiled with the Adam optimizer and accuracy as the metric.

Enable TensorBoard Logging:

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
TensorBoard logging is enabled, and logs are saved in a directory with a timestamp.

Train the Model:

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
The model is trained for 5 epochs, and logs are saved for visualization in TensorBoard.

Launch TensorBoard:

Run tensorboard --logdir logs/fit in the terminal to visualize the logs.
