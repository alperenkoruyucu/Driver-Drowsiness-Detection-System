import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Literatürdeki sonuçları ve kendi sonuçlarınızı manuel olarak girme
RLDD_literature_results = [65.2]  # Buraya literatür sonuçlarınızı girin
RLDD_my_results = [57.6]  # Buraya kendi sonuçlarınızı girin
DROZY_literature_results = [78.4]  # Buraya literatür sonuçlarınızı girin
DROZY_my_results = [55.2]  # Buraya kendi sonuçlarınızı girin

# Sonuçları bir DataFrame'e dönüştürme
results_df = pd.DataFrame({
    'Datasets': ['RLDD', 'DROZY'],
    'Literature Results': [RLDD_literature_results[0], DROZY_literature_results[0]],
    'Our Results': [RLDD_my_results[0], DROZY_my_results[0]]
})

# DataFrame'i uzun formattan geniş formata dönüştürme
results_df = results_df.melt('Datasets', var_name='Type', value_name='Results')

# Sonuçları çizdirme
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Datasets', y='Results', hue='Type', data=results_df)
plt.title('Dataset Results Comparison')
plt.ylabel('Results')
plt.ylim(0, 100)  # Y ekseninin sınırlarını belirleme

# Her bir çubuğun içine yüzde değerini yazdırma
for p in bar_plot.patches:
    bar_plot.annotate('{:.1f}%'.format(p.get_height()),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')

plt.show()