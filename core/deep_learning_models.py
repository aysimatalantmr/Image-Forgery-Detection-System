import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from django.conf import settings

# =====================================================================
# 1. MODEL MİMARİLERİNİN TANIMLANMASI (Notebook'tan Birebir Alındı)
# =====================================================================

# --- A. RGB Akışı ---
class RGBStream(nn.Module):
    def __init__(self, num_classes=2):
        super(RGBStream, self).__init__()
        # Uyumluluk için pretrained=True kullanıldı (ResNet50_Weights.DEFAULT ile aynıdır)
        self.backbone = models.resnet50(pretrained=True) 
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, num_classes))
    def forward(self, x): return self.backbone(x)

# --- B. SRM Filtreleri ve Gürültü (Noise) Akışı ---
srm_filter1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
srm_filter2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
srm_filter3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0
srm_weights = np.zeros((3, 3, 3, 3), dtype=np.float32)
for i in range(3):
    srm_weights[0, i, :, :] = srm_filter1
    srm_weights[1, i, :, :] = srm_filter2
    srm_weights[2, i, :, :] = srm_filter3 

class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.srm_conv.weight = nn.Parameter(torch.from_numpy(srm_weights))
    def forward(self, x): return self.srm_conv(x)

class NoiseStream(nn.Module):
    def __init__(self, num_classes=2):
        super(NoiseStream, self).__init__()
        self.srm_layer = SRMLayer()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, num_classes))
    def forward(self, x): return self.backbone(self.srm_layer(x))

# --- C. Büyük Birleşme (Two-Stream Fusion) ---
class TwoStreamFusion(nn.Module):
    def __init__(self, rgb_model, noise_model, num_classes=2):
        super(TwoStreamFusion, self).__init__()
        self.rgb_stream = rgb_model
        self.noise_stream = noise_model
        
        # Son katmanları iptal edip 2048'er özellik çıkartıyoruz
        self.rgb_stream.backbone.fc = nn.Identity()
        self.noise_stream.backbone.fc = nn.Identity()
        
        # Sınıflandırıcı (Hakem) Katmanı
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        rgb_features = self.rgb_stream(x)
        noise_features = self.noise_stream(x)
        fused_features = torch.cat((rgb_features, noise_features), dim=1)
        return self.fusion_classifier(fused_features)

# =====================================================================
# 2. MODELİ YÜKLEME VE HAZIRLAMA
# =====================================================================
DL_MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'ai_models', 'best_fusion_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_model = None

if os.path.exists(DL_MODEL_PATH):
    try:
        print("Büyük Birleşme (Two-Stream Fusion) modeli yükleniyor...")
        # 1. Boş iskeletleri yarat
        rgb_net = RGBStream(num_classes=2)
        noise_net = NoiseStream(num_classes=2)
        dl_model = TwoStreamFusion(rgb_net, noise_net, num_classes=2)
        
        # 2. Eğitilmiş ağırlıkları iskelete giydir
        dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=device))
        dl_model.to(device)
        dl_model.eval() # Tahmin moduna al
        print("✅ PyTorch Fusion Modeli Başarıyla Hazırlandı!")
    except Exception as e:
        print(f"⚠️ MODEL YÜKLEME HATASI: {e}")
else:
    print(f"⚠️ UYARI: Model dosyası bulunamadı: {DL_MODEL_PATH}")

# =====================================================================
# 3. TAHMİN (PREDICTION) FONKSİYONU
# =====================================================================
def predict_with_deep_learning(image_path):
    if dl_model is None:
        return [{"feature": "Hata", "result": "Fusion modeli yüklenemedi!"}]

    try:
        # Eğitim kodundaki val_transforms ile BİREBİR aynı dönüşümler
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        # Görüntüyü oku ve hazırla
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Modele gönder ve sonucu al
        with torch.no_grad():
            outputs = dl_model(img_tensor)
            # Softmax ile çıktıları olasılığa (0-1 arası) çevir
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Eğitimde: 0 -> Orijinal (Au), 1 -> Sahte (Tp)
        score = probabilities[1].item() 
        
        # Sonucu formatla
        if score >= 0.5:
            class_name = "Sahte / Manipüle Edilmiş"
            confidence = f"%{score * 100:.2f}"
        else:
            class_name = "Orijinal (Authentic)"
            confidence = f"%{(1 - score) * 100:.2f}"

        return [
            {"feature": "Analiz Sonucu", "result": class_name},
            {"feature": "Yapay Zeka Güven Skoru", "result": confidence},
            {"feature": "Altyapı", "result": "Two-Stream Fusion (RGB + Noise)"},
            {"feature": "Ön İşleme", "result": "SRM Filters & Normalization"},
        ]

    except Exception as e:
        return [{"feature": "Hata", "result": f"Analiz Hatası: {str(e)}"}]