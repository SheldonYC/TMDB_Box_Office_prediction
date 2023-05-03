import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor

def clean_date(date):
  l = date.split("/")
  l[-1] = "20"+l[-1] if int(l[-1])<21 else "19"+l[-1]  # actually had a movie in 1921: tt0012349
  return "/".join(l)

def parse_rows(row, token):
  try:
    items = list(eval(row)) # convert to list of dictionaries
    l = [item[token] for item in items]
    return l
  except:
    return ["empty"]
  
def count_elements(col):
  counting_dict = {}
  for li in col: # element all store in a list for each row
    for element in li:
      if element in counting_dict:
        counting_dict[element] = counting_dict[element] + 1
      else:
        counting_dict[element] = 1
  return counting_dict

# clean & extract features
to_drop = ['id', 'imdb_id', 'original_title', 'overview', 'status', 'title', 'poster_path', 'runtime']
df = pd.read_csv(r'./datasets/train.csv')
df = df.drop(columns=to_drop)
# print(df.columns)
# print(df.isnull().sum())

# Binary encoding
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 1 if type(x)==str else 0)
df['homepage'] = df['homepage'].apply(lambda x: 1 if type(x)==str else 0)
df['tagline'] = df['tagline'].apply(lambda x: 1 if type(x)==str else 0)
df['Keywords'] = df['Keywords'].apply(lambda x: 1 if type(x)==str else 0)

# DateTime
df['release_date'] = df['release_date'].apply(clean_date)
df['release_date'] = pd.to_datetime(df['release_date'], format="%m/%d/%Y")
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

# Dominant feature
df['orginal_en'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
df['spoken_languages'] = df['spoken_languages'].apply(lambda x: parse_rows(x, 'iso_639_1'))
df['spoken_en'] = df['spoken_languages'].apply(lambda x: 1 if 'en' in x else 0)
# df['spoken_es'] = df['spoken_languages'].apply(lambda x: 1 if 'es' in x else 0)
df['production_countries'] = df['production_countries'].apply(lambda x: parse_rows(x, 'iso_3166_1'))
df['prod_US'] = df['production_countries'].apply(lambda x: 1 if 'US' in x else 0)
# df['prod_FR'] = df['production_countries'].apply(lambda x: 1 if 'FR' in x else 0)
# df['prod_DE'] = df['production_countries'].apply(lambda x: 1 if 'DE' in x else 0)

# Counting
df['production_companies'] = df['production_companies'].apply(lambda x: parse_rows(x, 'id'))
df['companies_count'] = df['production_companies'].apply(lambda x: len(x) if x[0] != 'empty' else 0)
df['companies_count'] = np.log1p(df['companies_count'])
df['cast'] = df['cast'].apply(lambda x: parse_rows(x, 'gender'))
df['cast_count'] = df['cast'].apply(lambda x: len(x) if x != ['empty'] else 0)
df['cast_count'] = np.log1p(df['cast_count'])
df['crew'] = df['crew'].apply(lambda x: parse_rows(x, 'gender'))
df['crew_count'] = df['crew'].apply(lambda x: len(x) if x != ['empty'] else 0)
df['crew_count'] = np.log1p(df['crew_count'])

# One-hot encoding
df['genres'] = df['genres'].apply(lambda x: parse_rows(x, 'name'))
df['genres_count'] =  df['genres'].apply(lambda x: len(x) if x[0] != 'empty' else 0)
s = df['genres']
mlb = MultiLabelBinarizer()
temp = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index)
df = pd.concat([df, temp], axis=1)

# Numerical
df['log_popularity'] = np.log10(df['popularity'])
df['log_budget'] = np.log10(df['budget']+1)

df = df.drop(columns=['release_date', 'original_language', 'spoken_languages', 'production_countries', 'release_date', 'popularity', 'budget', 'genres', 'TV Movie', 'production_companies', 'cast', 'crew'])
# print(df.columns)
train_cols = set(df.columns)

# test
to_drop = ['imdb_id', 'original_title', 'overview', 'status', 'title', 'poster_path', 'runtime']
test = pd.read_csv(r'./datasets/test.csv')
test = test.drop(columns=to_drop)

# Binary encoding
test['belongs_to_collection'] = test['belongs_to_collection'].apply(lambda x: 1 if type(x)==str else 0)
test['homepage'] = test['homepage'].apply(lambda x: 1 if type(x)==str else 0)
test['tagline'] = test['tagline'].apply(lambda x: 1 if type(x)==str else 0)
test['Keywords'] = test['Keywords'].apply(lambda x: 1 if type(x)==str else 0)

# # DateTime
test['release_date'] = test['release_date'].fillna('09/15/13') # fill NaN with mode of year and month
test['release_date'] = test['release_date'].apply(clean_date)
test['release_date'] = pd.to_datetime(test['release_date'], format="%m/%d/%Y")
test['release_year'] = test['release_date'].dt.year
test['release_month'] = test['release_date'].dt.month

# Dominant feature
test['orginal_en'] = test['original_language'].apply(lambda x: 1 if x == 'en' else 0)
test['spoken_languages'] = test['spoken_languages'].apply(lambda x: parse_rows(x, 'iso_639_1'))
test['spoken_en'] = test['spoken_languages'].apply(lambda x: 1 if 'en' in x else 0)
# test['spoken_es'] = test['spoken_languages'].apply(lambda x: 1 if 'es' in x else 0)
test['production_countries'] = test['production_countries'].apply(lambda x: parse_rows(x, 'iso_3166_1'))
test['prod_US'] = test['production_countries'].apply(lambda x: 1 if 'US' in x else 0)
# test['prod_FR'] = test['production_countries'].apply(lambda x: 1 if 'FR' in x else 0)
# test['prod_DE'] = test['production_countries'].apply(lambda x: 1 if 'DE' in x else 0)

# Counting
test['production_companies'] = test['production_companies'].apply(lambda x: parse_rows(x, 'id'))
test['companies_count'] = test['production_companies'].apply(lambda x: len(x) if x[0] != 'empty' else 0)
test['companies_count'] = np.log1p(test['companies_count'])
test['cast'] = test['cast'].apply(lambda x: parse_rows(x, 'gender'))
test['cast_count'] = test['cast'].apply(lambda x: len(x) if x != ['empty'] else 0)
test['cast_count'] = np.log1p(test['cast_count'])
test['crew'] = test['crew'].apply(lambda x: parse_rows(x, 'gender'))
test['crew_count'] = test['crew'].apply(lambda x: len(x) if x != ['empty'] else 0)
test['crew_count'] = np.log1p(test['crew_count'])

# One-hot encoding
test['genres'] = test['genres'].apply(lambda x: parse_rows(x, 'name'))
test['genres_count'] =  test['genres'].apply(lambda x: len(x) if x[0] != 'empty' else 0)
s = test['genres']
mlb = MultiLabelBinarizer()
temp = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=test.index)
test = pd.concat([test, temp], axis=1)

# Numerical
test['log_popularity'] = np.log10(test['popularity'])
test['log_budget'] = np.log10(test['budget']+1)

test = test.drop(columns=['release_date', 'original_language', 'spoken_languages', 'production_countries', 'release_date', 'popularity', 'budget', 'genres', 'production_companies', 'cast', 'crew'])

test_cols = set(test.columns)
extra_cols = test_cols - train_cols
# print(extra_cols)
# test.drop(columns=extra_cols)

df.to_csv(r'./datasets/transformed_train.csv', index=False)
test.to_csv(r'./datasets/transformed_test.csv', index=False)

# partition for different range of budgets (seems bad idea as without budget performs worse)
# df_with_budget = df[df['log_budget']>4]
# df_without_budget = df[df['log_budget']<=4]
# test_with_budget = test[test['log_budget']>4]
# test_without_budget = test[test['log_budget']<=4]
# df_without_budget = df_without_budget.drop(columns='log_budget')
# test_without_budget = test_without_budget.drop(columns='log_budget')
# df_with_budget.to_csv(path_or_buf=r'./datasets/transformed_train_budget.csv', index=False)
# df_without_budget.to_csv(path_or_buf=r'./datasets/transformed_train_nobudget.csv', index=False)
# test_with_budget.to_csv(path_or_buf=r'./datasets/transformed_test_budget.csv', index=False)
# test_without_budget.to_csv(path_or_buf=r'./datasets/transformed_test_nobudget.csv', index=False)