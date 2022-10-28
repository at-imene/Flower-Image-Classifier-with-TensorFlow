import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

import argparse
import json

def process_image(np_img):
    ''' resize and normalize an image '''
    IMG_SHAPE = 224
    img = tf.image.resize(np_img, (IMG_SHAPE, IMG_SHAPE))/255.0
    return img.numpy()

def predict(image_path, model, top_k, class_names):
    ''' get the Top k probabilities of an image '''

    
    image_obj = Image.open(image_path)
    np_img = np.asarray(image_obj)
    processed_img= process_image(np_img)
    
    preds = model.predict(np.expand_dims(processed_img, axis=0))

    result= tf.math.top_k(preds, k= top_k)
    probs = result.values.numpy()[0]
    idx = result.indices.numpy()[0]
    
    classes = [class_names[f'{i+1}'] for i in idx ]
    
    return probs, classes

def getArgs():
    parser = argparse.ArgumentParser(description='Predict Flower name')
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('model_path', type=str, help='saved model path')
    parser.add_argument('--top_k', type=int, help='the top KK most likely classes')
    parser.add_argument('--category_names', type=str, help='category names path')
    
    return parser.parse_args()


def getCategories(args):
    # check if category names path was provided or not
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f) 
    else:
        with open('label_map.json', 'r') as f:
            class_names = json.load(f) 
    return class_names

def main():    
    args = getArgs()
    #     python predict.py ./test_images/orange_dahlia.jpg my_model.h5 --category_names label_map.json
    print(args)

    loaded_model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    image_path = args.image_path
    
    top_k =  args.top_k if (args.top_k is not None) else 5
    class_names= getCategories(args)
    
    probs, classes = predict(image_path, loaded_model, top_k, class_names)
    
    for i in range(top_k):
        print(f'{i+1}. {classes[i]} with probability : {probs[i]:.3f}')

if __name__ == "__main__":
    main()
