import numpy as np
import matplotlib.pyplot as plt

# Dosya adı
file_name = "accuracy4VTest.npy"

# Dosyayı yükle
accuracy = np.load(file_name)

# Değerleri yüzdeye çevir
accuracy_percentage = accuracy * 100

# Yüzde olarak doğruluk değerlerini yazdır
print(accuracy_percentage)

# Grafik oluştur
plt.plot(accuracy_percentage)

# Grafiğe başlık ekle
plt.title(file_name)

# Grafiği göster
plt.show()