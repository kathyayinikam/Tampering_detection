import os
from django.shortcuts import render
from django.core.files.storage import default_storage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.conf import settings

# Load the model once at the start
MODEL_PATH = os.path.join(settings.BASE_DIR, 'main_app/models/tumor_classification_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Save uploaded image
        image_file = request.FILES['image']
        file_path = default_storage.save('uploaded_images/' + image_file.name, image_file)

        # Preprocess the image
        img_path = os.path.join(settings.MEDIA_ROOT, file_path)
        img = load_img(img_path, target_size=(150, 150))  # Adjust size based on your model's input
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = tf.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        result = "Tampered" if prediction[0][0] > 0.5 else "Original"

        return render(request, 'result.html', {'result': result, 'image_url': file_path})
    
    return render(request, 'upload.html')
