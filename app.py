from flask import Flask, render_template, request, url_for, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# Klasör ayarları
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def pixel_art_yap(image_path, pixel_boyutu=96, renk_sayisi=16):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Resmi küçült (Downscaling)
    yukseklik, genislik = img.shape[:2]
    kucuk = cv2.resize(img, (pixel_boyutu, pixel_boyutu), interpolation=cv2.INTER_LINEAR)
    
    # Renk Azaltma (Uniform Quantization)
    aralik = 256 // renk_sayisi
    indirgenmis = (kucuk // aralik) * aralik + (aralik // 2)
    
    img_islenen = np.float32(indirgenmis)
    h, w, c = img_islenen.shape

    # Dithering İşlemi (Floyd-Steinberg)
    for y in range(h):
        for x in range(w):
            old_val = img_islenen[y, x].copy()
            
            # Pikseli belirlenen renk aralığına yuvarla
            new_val = np.round(old_val / aralik) * aralik + (aralik // 2)
            new_val = np.clip(new_val, 0, 255)
            
            img_islenen[y, x] = new_val
            
            # Hata hesabı
            error = old_val - new_val
            
            # Hatayı komşulara dağıt
            if x + 1 < w: 
                img_islenen[y, x+1] += error * 7 / 16
            if x - 1 > 0 and y + 1 < h: 
                img_islenen[y+1, x-1] += error * 3 / 16
            if y + 1 < h: 
                img_islenen[y+1, x] += error * 5 / 16
            if x + 1 < w and y + 1 < h: 
                img_islenen[y+1, x+1] += error * 1 / 16

    # Sonuç resmi büyüt (Upscaling - Nearest Neighbor)
    sonuc = np.uint8(np.clip(img_islenen, 0, 255))
    sonuc_buyuk = cv2.resize(sonuc, (genislik, yukseklik), interpolation=cv2.INTER_NEAREST)
    
    sonuc_adi = 'pixel_art_' + os.path.basename(image_path)
    sonuc_yolu = os.path.join(app.config['UPLOAD_FOLDER'], sonuc_adi)
    cv2.imwrite(sonuc_yolu, sonuc_buyuk)
    
    return sonuc_adi

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return 'Dosya yok'
        file = request.files['file']
        if file.filename == '': return 'Dosya seçilmedi'
        
        try:
            secilen_renk = int(request.form.get('color_count', 16))
        except ValueError:
            secilen_renk = 16

        if file:
            dosya_yolu = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(dosya_yolu)
            sonuc_dosyasi = pixel_art_yap(dosya_yolu, renk_sayisi=secilen_renk)
            return render_template('index.html', orijinal=file.filename, sonuc=sonuc_dosyasi)
            
    return render_template('index.html', orijinal=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)