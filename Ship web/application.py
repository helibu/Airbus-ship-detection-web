import os
import sys
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops
from skimage.io import imread
import os
import gc; gc.enable()
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import logging


from flask import Flask, request, render_template, send_from_directory

__author__ = 'He Li'

application = app = Flask(__name__)
app.secret_key = "define secret key of your own choice"
app.config['SESSION_TYPE'] = 'filesystem'
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
global ship_model, graph
ship_model = load_model('full_ship_model.h5')
#ship_model.summary()
fullres_model = load_model('seg_model.h5', compile=False)

graph = tf.get_default_graph()

dg_args=dict(featurewise_center=False,
                           samplewise_center=False,
                           rotation_range=45,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.01,
                           zoom_range=[0.9, 1.25],
                           brightness_range=[0.5, 1.5],
                           horizontal_flip=True,

                           vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last',
              preprocessing_function = preprocess_input)
valid_args = dict(fill_mode = 'reflect',
                   data_format = 'channels_last',
                  preprocessing_function = preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload")
def upload_html():
    return render_template("upload.html")

@app.route('/gallery', methods=["POST","GET" ])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)

    predict_target = os.path.join(APP_ROOT, 'predict_images/')
    print(predict_target)
    if not os.path.isdir(predict_target):
        os.mkdir(predict_target)
    else:
        print("Couldn't create upload directory: {}".format(predict_target))


    image_names = os.listdir('./images')
    # print(image_names)
    # print(image_names)
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 64*2
    path = []
    for image in image_names:
        path.append('./images/' + image)

    print(path)
    # ship_model.predict(im)
    if len(image_names) != 0:
        df = pd.DataFrame({'ImageId': image_names, 'path': path})
        test = flow_from_dataframe(valid_idg,
                                   df,
                                   path_col='path',
                                   y_col='ImageId',
                                   target_size=IMG_SIZE,
                                   color_mode='rgb',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
        gc.collect();
        for (t_x, c_img) in test:
            with graph.as_default():
                predict = ship_model.predict(t_x)
            break
        pred = []
        for i in range(predict.shape[0]):
            pred.append(predict[i][0])
        print(pred)
        my_prediction = dict(zip(image_names, pred))
        test_image_dir = './images/'
        font = ImageFont.truetype('Copilme_Bold.ttf', 50)
        with graph.as_default():
            for c_img_name in image_names:
                description = []
                c_path = os.path.join(test_image_dir, c_img_name)

                if my_prediction[c_img_name] >= 0.5:

                    c_img = imread(c_path)
                    first_img = np.expand_dims(c_img, 0) / 255.0
                    first_seg = fullres_model.predict(first_img)
                    lbl = label(first_seg[0, :, :, 0] * 2)
                    # skimage.measure.regionprops:Measure properties of labeled image regions. input is the labeled image
                    props = regionprops(lbl)

                    for prop in props:
                        # regionprops.bbox : tuple Bounding box (min_row, min_col, max_row, max_col) of the labeled image
                        cv2.rectangle(c_img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

                    # c_img_name_predict = "predict" + c_img_name
                    c_path_predict = os.path.join(predict_target, c_img_name)
                    cv2.imwrite(c_path_predict, cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR))
                    os.remove(os.path.join('./images/', c_img_name))

                    description.append("Has Ship")
                    chance = "{:.5f}".format(my_prediction[c_img_name])
                    description.append(chance)
                    string = "\n".join(description)

                    (x, y) = (0, 0)
                    im = Image.open(c_path_predict)
                    draw = ImageDraw.Draw(im)
                    draw.text((x, y), string, (255, 255, 255, 0), font=font)
                    im.save(c_path_predict , 'JPEG')
                    im.close()


                else:
                    description.append("No Ship")
                    chance = "{:.5f}".format(my_prediction[c_img_name])
                    description.append(chance)
                    string = "\n".join(description)

                    (x, y) = (0, 0)
                    im = Image.open(c_path).convert('RGB')
                    draw = ImageDraw.Draw(im)
                    draw.text((x, y), string, (255,255,255,0), font=font)
                    c_path_predict = os.path.join(predict_target, c_img_name)
                    im.save(c_path_predict , 'JPEG')
                    #cv2.imwrite(os.path.join(predict_target, c_img_name), cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR))
                    os.remove(os.path.join('./images/', c_img_name))
    else:
        my_prediction = []
    image_names = []
    image_names_predict = os.listdir('./predict_images')
    return render_template("gallery.html", image_names=image_names_predict)



@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/galleryclear', methods=["POST","GET" ])
def clear_page():
    image_names = os.listdir('./predict_images')
    for file in image_names:
        os.remove(os.path.join('./predict_images/', file))
    return render_template("clear.html", image_names=[], prediction=[])



@app.route('/gallery/<filename>')
def send_image(filename):
    return send_from_directory("predict_images", filename)

@app.route('/samples/<filename>')
def sample_image(filename):
    return send_from_directory("sample_images", filename)

@app.route('/about/<filename>')
def about_image(filename):
    return send_from_directory("about_images", filename)


@app.route('/samples', methods=["POST","GET" ])
def sample_page():
    sample_images = os.listdir('./sample_images')
    return render_template("samples.html", image_names=sample_images)


if __name__ == "__main__":
    app.run(port=4555,debug=True)