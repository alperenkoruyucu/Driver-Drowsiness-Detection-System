import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Literatürdeki sonuçları ve kendi sonuçlarınızı manuel olarak girme
literature_results = [65]  # Buraya literatür sonuçlarınızı girin
my_results = [55]  # Buraya kendi sonuçlarınızı girin

# Sonuçları bir DataFrame'e dönüştürme
results_df = pd.DataFrame({
    'Type': ['Literature Results', 'My Results'],
    'Results': [literature_results[0], my_results[0]]
})

print(results_df)

# Sonuçları çizdirme
sns.catplot(x='Type', y='Results', kind='bar', data=results_df)
plt.title('RLDD Dataset Results Comparison')
plt.ylabel('Results')
plt.ylim(0, 100)  # Y ekseninin sınırlarını belirleme
plt.show()