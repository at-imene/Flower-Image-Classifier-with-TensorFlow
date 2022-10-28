# Project 2 : Flower image classifier
In this project, I built and trained a deep neural network classifier to recognize different species of flowers and then export it to use it in a console application.

The project is broken down into multiple steps:

 - Load the image dataset and create a pipeline.
 - Build and Train an image classifier on this dataset.
 - Use the trained model to perform inference on flower images.
 - Build a command Line Application that uses the trained model.


The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

#### Basic usage:

```sh
$ python predict.py /path/to/image saved_model
```
#### Options 
 - ---top-k : Return the top KK most likely classes:
 ```sh 
$ python predict.py /path/to/image saved_model --top_k KK 
```
 - --category_names : Path to a JSON file mapping labels to flower names:
 ```sh 
$ python predict.py /path/to/image saved_model --category_names map.json
```
#### Example 
```sh
$ python predict.py ./test_images/orchid.jpg my_model.h5
```
