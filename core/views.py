from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# İki modelimizi de içeri aktarıyoruz
from .traditional_models import predict_with_h5_model 
from .deep_learning_models import predict_with_deep_learning 

def index(request):
    original_image_url = None
    result_image_url = None
    detection_results = None
    selected_model = 'deep_learning' 

    if request.method == 'POST' and request.FILES.get('image'):
        
        # 1. Görüntüyü diske kaydet
        upload = request.FILES['image']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        
        original_image_url = fss.url(file)
        absolute_path = fss.path(file)

        # 2. Hangi butonun seçildiğini al
        selected_model = request.POST.get('selected_model_input', 'traditional')

        # 3. İlgili Yapay Zeka Modelini Çalıştır
        if selected_model == 'traditional':
            # Geleneksel/Hibrit ELA modeli (.h5)
            detection_results = predict_with_h5_model(absolute_path)
            result_image_url = original_image_url 
            
        elif selected_model == 'deep_learning':
            # Yeni PyTorch Fusion Modeli (.pth)
            detection_results = predict_with_deep_learning(absolute_path)
            result_image_url = original_image_url

    context = {
        'original_image_url': original_image_url,
        'result_image_url': result_image_url,
        'detection_results': detection_results,
        'selected_model': selected_model,
    }
    
    return render(request, 'index.html', context)