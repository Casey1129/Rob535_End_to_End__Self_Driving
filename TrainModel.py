import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Flatten, Conv2D, Normalization, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def keras_model(training_data):
    """
    Builds and compiles a Keras sequential model for regression tasks.

    Parameters:
        training_data (numpy array): Training data used to adapt the normalization layer.

    Returns:
        model (Sequential): Compiled Keras model.
        callbacks_list (list): List of callbacks for model training.
    """
    model = Sequential()

    # Add normalization layer and adapt it to training data
    normalization_layer = Normalization(axis=-1)
    normalization_layer.adapt(training_data)
    model.add(normalization_layer)

    # Add convolutional layers with ReLU activation and max pooling
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")

    # Define checkpoint callback to save the best model
    filepath = "Autopilot.keras"
    checkpoint1 = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def loadFromPickle():
    """
    Loads features and labels from pickle files.

    Returns:
        features (numpy array): Feature data.
        labels (numpy array): Label data.
    """
    with open("features_40", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels

def augmentData(features, labels):
    """
    Augments the dataset by flipping the images and inverting labels.

    Parameters:
        features (numpy array): Original feature data.
        labels (numpy array): Original label data.

    Returns:
        augmented_features (numpy array): Augmented feature data.
        augmented_labels (numpy array): Augmented label data.
    """
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels

def plot_loss(history):
    """
    Plots the training and validation loss from the training history.

    Parameters:
        history (History): The History object returned by the fit method of the model.
    """
    # Extract loss and validation loss from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss values
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='Training Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to load data, preprocess, train the model, and plot results.
    """
    # Load and preprocess data
    features, labels = loadFromPickle()
    features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)

    # Split data into training and test sets
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 40, 40, 1)
    test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)

    # Build and train the model
    model, callbacks_list = keras_model(train_x)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10, batch_size=64,
              callbacks=callbacks_list)

    # Display model summary and save the final model
    model.summary()
    model.save('models/model.keras')  # Save in the .keras format

    # Plot the training and validation loss
    plot_loss(history)

if __name__ == "__main__":
    main()
