from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# ELA işlemi ve .h5 modelini çalıştıran fonksiyonumuzu içeri aktarıyoruz
from .traditional_models import predict_with_h5_model 

def index(request):
    # Başlangıç değerleri
    original_image_url = None
    result_image_url = None
    detection_results = None
    selected_model = 'traditional' # Artık varsayılan olarak geleneksel (ELA) modelimiz aktif

    if request.method == 'POST' and request.FILES.get('image'):
        
        # 1. Görüntüyü Diske Kaydetme (Veritabanı olmadan)
        upload = request.FILES['image']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        
        # Dosya yolları
        original_image_url = fss.url(file)
        absolute_path = fss.path(file)

        # 2. Hangi Modelin Seçildiğini Alma
        selected_model = request.POST.get('selected_model_input', 'traditional')

        # 3. İlgili Modeli Çalıştırma
        if selected_model == 'traditional':
            # ELA + Xception modeline resmi gönderiyoruz
            detection_results = predict_with_h5_model(absolute_path)
            
            # Modelimiz yeni bir çıktı görseli üretmiyor, metinsel/skor bazlı sonuç veriyor. 
            # Bu yüzden sonuç görseli olarak yine orijinali gösteriyoruz.
            result_image_url = original_image_url 
        
        elif selected_model == 'deep_learning':
            # Diğer buton için yer tutucu
            result_image_url = original_image_url
            detection_results = [
                {"feature": "Durum", "result": "Derin Öğrenme (Deep Learning) modeli henüz bağlanmadı."}
            ]

    # 4. Şablona Gönderilecek Veriler
    context = {
        'original_image_url': original_image_url,
        'result_image_url': result_image_url,
        'detection_results': detection_results,
        'selected_model': selected_model,
    }
    
    return render(request, 'index.html', context)