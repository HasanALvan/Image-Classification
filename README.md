# Image Classification with Tensorflow Keras

## Project Overview

This project aims to predict whether an image belongs to a plane or not using TensorFlow Keras.

### Dependencies:
- TensorFlow
- Keras
- NumPy
- PIL
- scipy


## Dataset

To train and evaluate the model, you need to prepare a dataset with the following structure:
```
./dataset
├── test_set
│ ├── plane
│ └── nonplane
└── train_set
├── plane
└── nonplane
```

- `test_set` contains images for testing the model.
  - `plane` directory: Contains images of planes.
  - `nonplane` directory: Contains images that are not planes.
  
- `train_set` contains images for training the model.
  - `plane` directory: Contains images of planes.
  - `nonplane` directory: Contains images that are not planes.

## Instructions

1. **Download the Script:**
   Download the provided script that includes the implementation of the image classification model.

2. **Create Dataset:**
   Create the `dataset` folder as described above. Populate the `plane` and `nonplane` directories within both `train_set` and `test_set` with corresponding images.

3. **Change Paths:**
   Update the script to point to the correct paths for your `train_set` and `test_set` directories.

4. **Train the Model:**
   Run the script to train the model with your dataset.

5. **Test the Model:**
   After training, you can use the model to predict whether new images belong to a plane or not. Update the script with the paths of the images you want to test and observe the results.

## Example Usage

To test the model with an image, modify the script to include the path of the image you want to test. The script will output whether the image is classified as a plane or not.

```python
# Example code snippet for testing
image_path = "path_to_your_image.jpg"
prediction = model.predict(image_path)
print(f"The image {image_path} is classified as: {'plane' if prediction else 'nonplane'}")

