import os
import requests
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt

# ==== Load DL Model ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'food_classifier_model.h5')
model = load_model(MODEL_PATH)

# ==== Class Labels ====
TRAIN_DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'train_data')
labels = sorted(os.listdir(TRAIN_DATA_PATH))

# ==== Spoonacular Nutrition Info ====
def get_nutrition_info(dish_name):
    try:
        api_key = settings.SPOONACULAR_API_KEY
        url = f"https://api.spoonacular.com/recipes/guessNutrition?title={dish_name}&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()

        return {
            "calories": data.get("calories", {}).get("value", "N/A"),
            "carbs": data.get("carbs", {}).get("value", "N/A"),
            "fat": data.get("fat", {}).get("value", "N/A"),
            "protein": data.get("protein", {}).get("value", "N/A")
        }
    except Exception as e:
        return {"error": str(e)}

# ==== Main Page (Login + Signup in main.html) ====
def main_page(request):
    return render(request, 'main.html')

# ==== Handle Login from main.html ====
@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # ✅ Check against hardcoded values
        if username == 'admin' and password == 'abc@123':
            return redirect('home')
        else:
            return render(request, 'main.html', {'error': 'Invalid credentials'})
    return redirect('main')


# ==== Home Page (after login) ====
def home(request):
    return render(request, 'home.html')

# ==== Upload Image Page ====
def upload_image(request):
    prediction = None
    image_url = None
    nutrition = None

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        file_path = default_storage.save('temp.jpg', img_file)
        img_path = os.path.join(os.getcwd(), file_path)

        # Preprocess image
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = labels[np.argmax(preds)]
        prediction = predicted_class
        image_url = file_path

        # Nutrition info (optional)
        nutrition = get_nutrition_info(predicted_class)

    return render(request, 'upload.html', {
        'prediction': prediction,
        'image_url': image_url,
        'nutrition': nutrition
    })

# ==== Generate Recipe from Ingredients ====
@csrf_exempt
def enter_ingredients(request):
    recipe = None

    if request.method == 'POST':
        ingredients = request.POST.get('ingredients', '')

        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are a professional chef."},
                {"role": "user", "content": f"Generate a complete recipe using these ingredients: {ingredients}"}
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()
            recipe = data["choices"][0]["message"]["content"]
        except Exception as e:
            recipe = f"⚠️ Error generating recipe: {str(e)}"

    return render(request, 'ingredients.html', {
        'recipe': recipe
    })
