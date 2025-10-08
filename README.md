# 🧠 Optimizasyon Algoritmalarının Karşılaştırılması (C)

Bu proje, **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)** ve **Adam** optimizasyon algoritmalarını bir sınıflandırma problemi üzerinde karşılaştırmak amacıyla geliştirilmiştir.  
MNIST Fashion veri seti kullanılarak, sınıflandırma başarımı ve eğitim süreçleri analiz edilmiştir.

---

## 📚 İçindekiler
- [📌 Proje Özeti](#-proje-özeti)
- [📊 Kullanılan Veri Seti](#-kullanılan-veri-seti)
- [⚙️ Kullanılan Parametreler](#️-kullanılan-parametreler)
- [🧠 Algoritmalar](#-algoritmalar)
- [🛠️ Kurulum](#-kurulum)
- [▶️ Çalıştırma](#️-çalıştırma)
- [📈 Sonuçlar & Grafikler](#-sonuçlar--grafikler)
- [📽️ Video Açıklaması](#️-video-açıklaması)
- [👨‍💻 Geliştirici](#-geliştirici)
- [📄 Lisans](#-lisans)

---

## 📌 Proje Özeti

- 📐 **Amaç:** Farklı optimizasyon algoritmalarının eğitim performanslarını ve doğruluk oranlarını karşılaştırmak.  
- 💻 **Dil:** C  
- 🧮 **Yapı:** Projede veri okuma, aktivasyon, ileri besleme, ağırlık güncelleme ve değerlendirme adımları yer alır.  
- 🧠 **Algoritmalar:** Gradient Descent, Stochastic Gradient Descent, Adam  
- 📝 **Sonuçlar:** Her algoritma için loss (kayıp) değerleri dosyaya yazılır, doğruluk oranı hesaplanır ve epoch-loss grafikleri çizilebilir.

---

## 📊 Kullanılan Veri Seti

Proje, **Fashion MNIST** veri setinin 28×28 piksellik gri tonlu görüntülerinden oluşturulan bir alt kümesini kullanır:

- 👟 **Sneaker** sınıfı → 1250 örnek  
- 👗 **Dress** sınıfı → 1250 örnek  
- Toplam veri sayısı: **2500**
- Eğitim/Test oranı: %80 eğitim / %20 test

Veriler `.csv` formatında okunur, etiketler yalnızca “3” (sneaker, +1) ve “7” (dress, -1) olacak şekilde filtrelenir.

---

## ⚙️ Kullanılan Parametreler

| Parametre | Açıklama |
|-----------|----------|
| `N` | Görüntü boyutu (28) |
| `PIXELS` | 28×28 = 784 |
| `DATA_SIZE` | 2500 |
| `TRAIN_RATIO` | 0.8 |
| `LEARNING_RATE` | Öğrenme oranı |
| `EPOCHS` | Eğitim döngüsü sayısı |
| `BETA1, BETA2, EPSILON` | Adam için hiperparametreler |

---

## 🧠 Algoritmalar

### 1️⃣ Gradient Descent (GD)
- Tüm veri seti üzerinden tek seferde gradyan hesaplanır.  
- Sabit öğrenme oranı ile ağırlıklar güncellenir.  
- Paralel güncelleme yapısı sayesinde simetrik bir ağırlık uzayı izler.

### 2️⃣ Stochastic Gradient Descent (SGD)
- Her adımda rastgele bir örnek seçilir.  
- Mini-batch yerine tek örnekle güncelleme yapılır.  
- Ağırlık uzayında farklı bölgelerin optimize edilmesini sağlar.

### 3️⃣ Adam
- Momentum ve adaptif öğrenme oranı kullanır.  
- `m` (hareketli ortalama) ve `v` (kare ortalama) hesaplanır.  
- Bias düzeltmesi yapılarak ağırlıklar güncellenir.

---

## 🛠️ Kurulum

1. Projeyi klonla:
```bash
git clone https://github.com/kullaniciadi/optimizasyon-algoritmalari-c.git
cd optimizasyon-algoritmalari-c
```

2. Derle:
```bash
gcc main.c -o optimizasyon -lm
```

> `-lm` bayrağı matematik fonksiyonları için gereklidir.

---

## ▶️ Çalıştırma

Veri dosyasının (`fashion_mnist_subset.csv` vb.) proje dizininde olduğundan emin olun.

```bash
./optimizasyon
```

Program sırasıyla:

1. Veriyi yükler ve eğitim/test kümelerine ayırır  
2. Ağırlıkları rastgele başlatır  
3. GD, SGD ve Adam algoritmalarını ayrı ayrı çalıştırır  
4. Her birinin **loss değerlerini** kaydeder  
5. Test doğruluğunu hesaplar  
6. Sonuçları `.csv` dosyalarına yazar

---

## 📈 Sonuçlar & Grafikler

Eğitim sırasında loss değerleri epoch bazında kaydedilir. Bu veriler kullanılarak çizilen bazı grafik örnekleri 👇

### Epoch vs Loss  
Her algoritma için epoch sayısına göre loss eğrileri çizilir.

### t-SNE Görselleştirmesi  
Ağırlıkların optimizasyon sürecindeki evrimi t-SNE yöntemiyle 2B düzleme indirgenerek görselleştirilmiştir.  
- SGD → farklı dallara ayrılan ağırlık grupları  
- Adam → 4 kümeye ayrışmış net bölgeler  
- GD → yıldız biçiminde simetrik yapı

---

## 📽️ Video Açıklaması

Kodu anlatan video:  
👉 [YouTube - Optimizasyon Algoritmaları Açıklama](https://www.youtube.com/watch?v=qH5uKz3aNZU)

---

## 👨‍💻 Geliştirici

**Yusuf Başar Gündüz**  
📧 basar.gunduz@std.yildiz.edu.tr  
🎓 Yıldız Teknik Üniversitesi - Bilgisayar Mühendisliği
