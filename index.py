import numpy as np
import matplotlib.pyplot as plt


# Dosyayı yükle
accuracy = str(list(np.load('LabelsTest_30_Fold5.npy')))

# Değerleri yüzdeye çevir
#accuracy_percentage = accuracy * 100

# Yüzde olarak doğruluk değerlerini yazdır
print(accuracy)

# Grafik oluştur
#plt.plot(accuracy)

# Grafiğe başlık ekle
#plt.title(file_name)

# Grafiği göster
#plt.show()