# 👾 Retro Pixel Art Dönüştürücü

Bu proje, Görüntü İşleme dersi kapsamında geliştirilmiş; yüksek çözünürlüklü fotoğrafları nostaljik "Pixel Art" formatına dönüştüren web tabanlı bir araçtır.

## 🚀 Kullanılan Teknolojiler ve Yöntemler
* **Python & Flask:** Web sunucusu ve backend.
* **OpenCV:** Görüntü işleme kütüphanesi.
* **K-Means Clustering:** Görüntüdeki baskın renkleri bulup palet oluşturmak için (Renk Nicemleme).
* **Floyd-Steinberg Dithering:** Sınırlı renk paletinde yumuşak geçişler sağlamak için (Hata Dağıtımı).
* **Nearest Neighbor Scaling:** Keskin piksel görünümünü korumak için.

## 🛠️ Kurulum
Projeyi bilgisayarınızda çalıştırmak için:
1. Depoyu klonlayın:
   `git clone https://github.com/Tiginn/Pixel-Art-Converter.git`
2. Gereksinimleri yükleyin:
   `pip install -r requirements.txt`
3. Uygulamayı başlatın:
   `python app.py`
