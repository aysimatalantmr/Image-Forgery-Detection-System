from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def index(request):
    uploaded_file_url = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        upload = request.FILES['image']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        uploaded_file_url = fss.url(file)
        
        # Burada model tahmini işlemlerini yapabilirsin.
        # model_sonucu = predict_image(file) vb.

    return render(request, 'index.html', {
        'uploaded_file_url': uploaded_file_url
    })