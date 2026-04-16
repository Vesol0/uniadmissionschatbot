import pandas as pd
import matplotlib.pyplot as plt

## read document
df = pd.read_csv('data_semantic_retrieval.csv')

plot_df = df.set_index('user_input')[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']]

# Create a grouped bar chart
ax = plot_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')

plt.title('RAGAS Semantic Retrieval Evaluation Metrics Across Queries', fontsize=16)
plt.xlabel('Test Queries', fontsize=12)
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.ylim(0, 1.05)  # Ragas scores are between 0 and 1
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# save plot to directory
plt.savefig('ragas_evaluation_results_semantic.png', dpi=300, bbox_inches='tight')