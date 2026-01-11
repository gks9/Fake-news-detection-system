import pandas as pd
import os

# Read both files
print("Reading ISOT fake.csv and real.csv...")
fake_df = pd.read_csv('data/fake.csv')
real_df = pd.read_csv('data/real.csv')

print(f"Fake articles: {len(fake_df)}")
print(f"Real articles: {len(real_df)}")

# Add labels
fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

# Combine
df_combined = pd.concat([fake_df, real_df], ignore_index=True)

print(f"Total combined: {len(df_combined)}")
print(f"\nLabel distribution:")
print(df_combined['label'].value_counts())

# Save as single file
output_path = 'data/fake_or_real_news.csv'
df_combined.to_csv(output_path, index=False)
print(f"\n✓ Saved combined dataset to: {output_path}")

# Verify
print("\nVerifying...")
verify_df = pd.read_csv(output_path)
print(f"Columns: {verify_df.columns.tolist()}")
print(f"Rows: {len(verify_df)}")
print(f"Labels: {verify_df['label'].value_counts().to_dict()}")
