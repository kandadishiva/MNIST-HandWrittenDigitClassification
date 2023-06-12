from django.shortcuts import render
import json
from django.http import JsonResponse
import base64
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Create your views here.

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        print("hello")
        data = json.loads(request.body)
        image_data = data.get('image')
        # Perform your recognition tasks using the image data
        
        # Remove the data URL prefix and decode the image data
        _, encoded_data = image_data.split(',', 1)
        decoded_data = base64.b64decode(encoded_data)

        # Convert the decoded data into a PIL Image object
        image = Image.open(io.BytesIO(decoded_data))
        print(type(image))


        img = image.resize((28,28))
        #convert rgb to grayscale
        img = img.convert('L')
        image = np.array(img)
        img = image.reshape(1,28,28,1)
        img = img/255.0

        model = load_model('models\\mnist.h5')
        res = model.predict([img])[0]
        prediction = np.argmax(res)
        print("answer ",prediction)
        #plt.imshow(image, cmap='gray')
        #plt.axis('off')
        #plt.show()
    
    return render(request, 'index.html')