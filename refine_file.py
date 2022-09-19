import pandas as pd

df = pd.read_csv('./search_output_TO_CUT.csv')
# df1 = df['Description'] # .map(lambda d: d.replace('\n', ''))
# df1 = df1.map(lambda d: d.replace('\n', ''))
df['Description'] = df['Description'].apply(lambda d: d.replace('\n', ''))
df = df.drop(df.columns[[0,1]], axis=1)
df.to_csv('names_TO_CUT.csv')
print()