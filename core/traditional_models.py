import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
from keras.models import load_model
from django.conf import settings

# 1. Modeli sunucu başlarken belleğe al
# Modelinin isminin 'best_casia_ela_model.h5' olduğunu varsayıyoruz
MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'ai_models', 'best_casia_ela_model.h5')

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ ELA Modeli başarıyla yüklendi!")
else:
    model = None
    print(f"⚠️ UYARI: Model dosyası bulunamadı: {MODEL_PATH}")

def apply_ela(image_path, target_size=(299, 299), ela_quality=90):
    """
    Eğitim kodundaki 'CasiaELAGenerator' sınıfının içindeki ELA fonksiyonunun aynısı.
    """
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, format='JPEG', quality=ela_quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_image = ela_image.resize(target_size)
    img_array = np.array(ela_image) / 255.0
    
    return img_array

def predict_with_h5_model(image_path):
    """
    Arayüzden gelen resmi ELA işleminden geçirip modele sokar.
    """
    if model is None:
         return [{"feature": "Hata", "result": "Model dosyası (.h5) bulunamadı!"}]

    try:
        # 2. Resmi ELA işleminden geçir (Xception modeli 299x299 bekler)
        img_array = apply_ela(image_path, target_size=(299, 299))
        
        # Batch boyutu ekle (1, 299, 299, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Tahmin İşlemi
        predictions = model.predict(img_array)
        score = float(predictions[0][0])

        fake_prob = score * 100
        real_prob = (1 - score) * 100

        if score >= 0.80:

            class_name = "Sahte / Oynanmış (Tampered)"
            confidence = f"%{fake_prob:.2f}"
            color_class = "text-red-500"

        elif score >= 0.55:

            class_name = "Şüpheli / Düşük Güven"
            confidence = f"%{fake_prob:.2f}"
            color_class = "text-yellow-500"

        else:

            class_name = "Orijinal (Authentic)"
            confidence = f"%{real_prob:.2f}"
            color_class = "text-green-500"

        # Arayüze gönderilecek sonuç listesi
        detection_results = [
            {"feature": "Analiz Sonucu", "result": class_name},
            {"feature": "Modelin Güven Oranı", "result": confidence},
            {"feature": "Ön İşleme", "result": "Error Level Analysis (ELA)"},
            {"feature": "Altyapı", "result": "Xception CNN Modeli"},
        ]
        
        return detection_results

    except Exception as e:
        return [{"feature": "Hata", "result": str(e)}]