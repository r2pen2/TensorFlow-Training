import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

###########################MAGIC HAPPENS HERE##########################
# Change the hyper-parameters to get the model performs well
config = {
    'batch_size': 128,
    'image_size': (100,100),
    'epochs': 50,
    'optimizer': keras.optimizers.experimental.Adam(5e-4)
}
###########################MAGIC ENDS  HERE##########################

def read_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "./images/flower_photos",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        labels='inferred',
        label_mode = 'int'
    )
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    return train_ds, val_ds, test_ds

def data_processing(ds):
    data_augmentation = keras.Sequential(
        [
            ###########################MAGIC HAPPENS HERE##########################
            # Use dataset augmentation methods to prevent overfitting, 
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.3),
            layers.RandomTranslation(height_factor=0.3, width_factor=0.3),
            layers.RandomContrast(0.3)
            ###########################MAGIC ENDS HERE##########################
        ]
    )
    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    ###########################MAGIC HAPPENS HERE##########################
    # Build up a neural network to achieve better performance.
    # Use Keras API like `x = layers.XXX()(x)`
    # Hint: Use a Deeper network (i.e., more hidden layers, different type of layers)
    # and different combination of activation function to achieve better result.
    hidden_units = 128
    x = layers.Conv2D(128, activation='relu', kernel_size=(3, 3), input_shape=input_shape)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = layers.Conv2D(128, activation='relu', kernel_size=(3, 3), input_shape=input_shape)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(hidden_units, activation='relu')(x)
    # x = layers.Dense(hidden_units, activation='relu')(x)
    # x = layers.Dense(hidden_units, activation='relu')(x)



    ###########################MAGIC ENDS HERE##########################
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer='he_normal')(x)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model



if __name__ == '__main__':
    # Load and Process the dataset
    train_ds, val_ds, test_ds = read_data()
    train_ds = data_processing(train_ds)
    # Build up the ANN model
    model = keras.models.load_model("best_trained_model.keras")
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer=config['optimizer'],
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
    # Fit the model with training dataset
    # history = model.fit(
    #     train_ds,
    #     epochs=config['epochs'],
    #     validation_data=val_ds
    # )
    ###########################MAGIC HAPPENS HERE##########################
    # print(history.history)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest Accuracy: ", test_acc)
    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    test_prediction = np.argmax(model.predict(test_images),1)
    # 1. Visualize the confusion matrix by matplotlib and sklearn based on test_prediction and test_labels
    cm = confusion_matrix(test_labels, test_prediction)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels to the plot
    classes = ["sunflowers", "dandelion", "daisy", "tulips", "roses"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Display the values in the matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 2. Report the precision and recall for 10 different classes
    # Hint: check the precision and recall functions from sklearn package or you can implement these function by yourselves.
    # 3. Visualize three misclassified images
    # Hint: Use the test_images array to generate the misclassified images using matplotlib

    # Identify Misclassified Images
    misclassified_indices = np.where(test_prediction != test_labels)[0]
    # Choose first 3 misclassified images
    selected_indices = misclassified_indices[:25]
    # Visualize Misclassified Images
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))

    for i, index in enumerate(selected_indices):
        # Convert the image to grayscale
        gray_image = np.mean(test_images[index], axis=-1)
        row = i // 5
        col = i % 5
        axes[row, col].imshow(gray_image.astype(np.uint8), cmap='gray')
        axes[row, col].set_title(f"True: {test_labels[index]}, Pred: {test_prediction[index]}")
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

    ###########################MAGIC HAPPENS HERE##########################