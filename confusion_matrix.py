import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix'iniz
cm = [[52, 6, 3], [12, 20, 25], [9, 12, 38]]

# Sınıflarınızın isimleri
class_names = ['Alert', 'Semisleepy', 'Sleepy']

# Matplotlib figure oluştur
plt.figure(figsize=(10,7))

# Seaborn ile heatmap oluştur
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 16, 'weight':'bold'})

# Başlık ve eksen isimleri
plt.title('Confusion Matrix', fontsize=20)
plt.ylabel('True Label', fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)

# Göster
plt.show()