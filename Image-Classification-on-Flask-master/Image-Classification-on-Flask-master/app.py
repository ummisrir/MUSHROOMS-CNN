import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
import numpy as np


app = Flask(__name__)



class_names = ['Boletus(Beracun)','Boletus(Konsumsi)', 'Ganoderma(Beracun)', 'Ganoderma(Konsumsi)', 'Russula(Beracun)','Russula(Konsumsi)']


model = load_model('model2_89_82_100_64.h5')
# model._make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(160, 160))
	i = img_to_array(i)
	i = np.expand_dims(i, axis=0)
	i /= 255. 
	#i = i.reshape(1, 200,200,3)
	p = model.predict(i)
	return p


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"



@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

		prediction = np.argmax(p)

		label = class_names[prediction]
		
		probability= round( float(p[0][np.argmax(p)] * 100), 2)

		# print(pred_class)

	return render_template("home.html", prediction = p.any(), probability = probability, label = label, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)