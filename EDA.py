import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

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

def visualize_correlation(df):
  numerical_features = ['budget', 'popularity', 'runtime', 'revenue']
  numerical = df.loc[:, numerical_features]
  corr = numerical.corr(method='pearson')
  sns.heatmap(corr, cmap="Blues", annot=True)
  plt.show()

def visualize_runtime(df):
  runtime = df.loc[:, ['runtime', 'revenue']]
  runtime_x, runtime_y = runtime.loc[:, 'runtime'], runtime.loc[:, 'revenue']
  runtime_corr = np.corrcoef(runtime_x, runtime_y)[1][0]
  plt.scatter(runtime_x, runtime_y)
  plt.title(f'Revenue vs Runtime, corr = {runtime_corr :.4f}')
  plt.xlabel('Runtime')
  plt.ylabel('Revenue')
  plt.show()

def visualize_budget(df):
  budget = df.loc[:, ['budget', 'revenue']]
  # budget =  budget[budget['budget'] > 10000]
  budget['log_budget'] = np.log10(budget['budget']+1)
  budget_x, log_budget_x, budget_y = budget.loc[: ,'budget'], budget.loc[: ,'log_budget'], budget.loc[:, 'revenue']
  budget_corr = np.corrcoef(log_budget_x, budget_y)[1][0]
  plt.plot(log_budget_x, budget_y, 'o')
  plt.title(f'Revenue vs Budget, corr = {budget_corr :.4f}')
  plt.xlabel('Budget')
  plt.ylabel('Revenue')
  plt.show()

def visualize_popularity(df):
  popularity = df.loc[:, ['popularity', 'revenue']]
  popularity['log_popularity'] = np.log10(popularity['popularity'])
  popularity_x, log_popularity_x, popularity_y = popularity.loc[: ,'popularity'], popularity.loc[: ,'log_popularity'], popularity.loc[:, 'revenue']
  popularity_corr = np.corrcoef(log_popularity_x, popularity_y)[1][0]
  plt.plot(log_popularity_x, popularity_y, 'o')
  plt.title(f'Revenue vs log_Popularity, corr = {popularity_corr :.4f}')
  plt.xlabel('Popularity')
  plt.ylabel('Revenue')
  plt.show()

def visualize_collection(df):
  df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 1 if type(x)==str else 0)
  collection = df.loc[:, ['belongs_to_collection', 'revenue']]
  collection['revenue'] = np.log10(collection['revenue'])
  collection.rename(columns={'revenue':'log_revenue'})
  boxplot = collection.boxplot(by='belongs_to_collection')
  plt.show()

def visualize_homepage(df):
  df['homepage'] = df['homepage'].apply(lambda x: 1 if type(x)==str else 0)
  homepage = df.loc[:, ['homepage', 'revenue']]
  homepage['revenue'] = np.log10(homepage['revenue'])
  homepage.rename(columns={'revenue':'log_revenue'})
  boxplot = homepage.boxplot(by='homepage')
  plt.show()

def visualize_tagline(df):
  df['tagline'] = df['tagline'].apply(lambda x: 1 if type(x)==str else 0)
  tagline = df.loc[:, ['tagline', 'revenue']]
  tagline['revenue'] = np.log10(tagline['revenue'])
  tagline.rename(columns={'revenue':'log_revenue'})
  boxplot = tagline.boxplot(by='tagline')
  plt.show()

def visualize_keywords(df):
  df['Keywords'] = df['Keywords'].apply(lambda x: 1 if type(x)==str else 0)
  Keywords = df.loc[:, ['Keywords', 'revenue']]
  Keywords['revenue'] = np.log10(Keywords['revenue'])
  Keywords.rename(columns={'revenue':'log_revenue'})
  boxplot = Keywords.boxplot(by='Keywords')
  plt.show()

def visualize_date_year(df):
  date = df.loc[:, ['release_date', 'revenue']]
  date['release_date'] = date['release_date'].apply(clean_date)
  date['release_date'] = pd.to_datetime(date['release_date'], format="%m/%d/%Y")
  date['release_year'] = date['release_date'].dt.year
  plot = sns.boxplot(x=date['release_year'], y=date['revenue'], color='orange')
  plot.set(xlim=(60, 90))
  plot.set_xlabel("Year")
  plot.set_ylabel("Revenue")
  plot.set_title("Revenue vs year")
  plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
  plt.show()

def visualize_date_month(df):
  date = df.loc[:, ['release_date', 'revenue']]
  date['release_date'] = date['release_date'].apply(clean_date)
  date['release_date'] = pd.to_datetime(date['release_date'], format="%m/%d/%Y")
  date['release_month'] = date['release_date'].dt.month
  plot = sns.boxplot(x=date['release_month'], y=date['revenue'], color='orange')
  plot.set_xlabel("month")
  plot.set_ylabel("Revenue")
  plot.set_title("Revenue vs month")
  plot.set_xticklabels(plot.get_xticklabels())
  plt.show()

def visualize_original_language(df):
  original_language = df.loc[:, ['original_language', 'revenue']]
  # print(original_language['original_language'].value_counts()) # en is dominant
  original_language['orginal_en'] = original_language['original_language'].apply(lambda x: 1 if x == 'en' else 0)
  original_language['log_revenue'] = np.log10(original_language['revenue'])
  original_language = original_language[['orginal_en', 'log_revenue']]
  boxplot = original_language.boxplot(by='orginal_en')
  plt.show()

def visualize_spoken_languages(df):
  spoken_languages = df.loc[:, ['spoken_languages', 'revenue']]
  spoken_languages['spoken_languages'] = spoken_languages['spoken_languages'].apply(lambda x: parse_rows(x, 'iso_639_1'))
  lang_count = count_elements(spoken_languages['spoken_languages'].values)
  # lang_count = sorted(lang_count.items(), key=lambda item: item[1], reverse=True)
  # print(lang_count)
  popular_lang = [k for k, v in lang_count.items() if v > 150]
  spoken_languages['log_revenue'] = np.log10(spoken_languages['revenue'])
  spoken_languages = spoken_languages[['spoken_languages', 'log_revenue']]
  spoken_languages = spoken_languages.explode('spoken_languages')
  spoken_languages = spoken_languages[spoken_languages['spoken_languages'].apply(lambda x: True if x in popular_lang else False)]
  plot = sns.boxplot(x=spoken_languages['spoken_languages'], y=spoken_languages['log_revenue'])
  plot.set(title='Log revenue vs Popular languages')
  plt.show()

def visualize_production_countries(df):
  production_countries = df.loc[:, ['production_countries', 'revenue']]
  production_countries['production_countries'] = production_countries['production_countries'].apply(lambda x: parse_rows(x, 'iso_3166_1'))
  country_count = count_elements(production_countries['production_countries'].values)
  # country_count = sorted(country_count.items(), key=lambda item: item[1], reverse=True)
  # print(country_count)
  popular_country = [k for k, v in country_count.items() if v > 150]
  production_countries['log_revenue'] = np.log10(production_countries['revenue'])
  production_countries = production_countries[['production_countries', 'log_revenue']]
  production_countries = production_countries.explode('production_countries')
  production_countries = production_countries[production_countries['production_countries'].apply(lambda x: True if x in popular_country else False)]
  plot = sns.boxplot(x=production_countries['production_countries'], y=production_countries['log_revenue'])
  plot.set(title='Log revenue vs Popular countries')
  plt.show()

def visualize_title(df):
  title = df.loc[:, ['title', 'revenue']]
  title['log_revenue'] = np.log10(title['revenue'])
  title['length'] = title['title'].apply(len) # only 8 <= length <= 15 has >150 count
  # print(np.corrcoef(title['length'], title['revenue'])[1][0])
  # print(np.corrcoef(np.log1p(title['length']), title['revenue'])[1][0])
  title = title[(title['length'] >= 8) & (title['length'] <= 15)]
  plot = sns.boxplot(x=title['length'], y=title['log_revenue'])
  plot.set(title='Log revenue vs Title length')
  plt.show()

def visualize_companies(df):
  companies = df.loc[:, ['production_companies', 'revenue']]
  companies['log_revenue'] = np.log10(companies['revenue'])
  companies['production_companies'] = companies['production_companies'].apply(lambda x: parse_rows(x, 'id'))
  companies['companies_count'] = companies['production_companies'].apply(lambda x: len(x) if x[0] != 'empty' else 0)
  # print(np.corrcoef(companies['companies_count'], companies['revenue'])[1][0])
  # print(np.corrcoef(np.log1p(companies['companies_count']), companies['revenue'])[1][0])
  plot = sns.boxplot(x=companies['companies_count'], y=companies['log_revenue'])
  plot.set(title='Log revenue vs companies count')
  plt.show()

def visualize_cast(df):
  cast = df.loc[:, ['cast', 'revenue']]
  cast['log_revenue'] = np.log10(cast['revenue'])
  cast['cast'] = cast['cast'].apply(lambda x: parse_rows(x, 'gender'))
  cast['cast_count'] = cast['cast'].apply(lambda x: len(x) if x != ['empty'] else 0)
  # print(np.corrcoef(cast['cast_count'], cast['revenue'])[1][0])
  # print(np.corrcoef(np.log1p(cast['cast_count']), cast['revenue'])[1][0])
  plot = sns.boxplot(x=cast['cast_count'], y=cast['log_revenue'])
  plot.set(title='Log revenue vs cast count')
  plt.show()

def visualize_crew(df):
  crew = df.loc[:, ['crew', 'revenue']]
  crew['log_revenue'] = np.log10(crew['revenue'])
  crew['crew'] = crew['crew'].apply(lambda x: parse_rows(x, 'gender'))
  crew['crew_count'] = crew['crew'].apply(lambda x: len(x) if x != ['empty'] else 0)
  # print(np.corrcoef(crew['crew_count'], crew['revenue'])[1][0])
  # print(np.corrcoef(np.log1p(crew['crew_count']), crew['revenue'])[1][0])
  plot = sns.boxplot(x=crew['crew_count'], y=crew['log_revenue'])
  plot.set(title='Log revenue vs crew count')
  plt.show()

def visualize_genre(df):
  genre = df.loc[:, ['genres', 'revenue']]
  genre['genres'] = genre['genres'].apply(lambda x: parse_rows(x, 'name'))
  genre_count = count_elements(genre['genres'].values)
  genre_count = sorted(genre_count.items(), key=lambda item: item[1], reverse=True)
  # print(genre_count)

  # Multiple class plot
  genre['log_revenue'] = np.log10(genre['revenue'])
  genre = genre[['genres', 'log_revenue']]
  genre = genre.explode('genres')
  plot = sns.boxplot(x=genre['genres'], y=genre['log_revenue'])
  plot.set(title='Log revenue vs genres')
  plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
  plt.show()

  # count of genres
  # genre['count'] = genre['genres'].apply(lambda x: len(x))
  # genre['log_revenue'] = np.log10(genre['revenue'])
  # genre = genre[['count', 'log_revenue']]
  # boxplot = genre.boxplot(by='count')
  # plt.show()


plt.style.use('ggplot')
# sns.set(font_scale=0.7)
df = pd.read_csv(r'./datasets/train.csv')
useless_cols = ['id', 'imdb_id', 'original_title', 'overview', 'status', 'title', 'poster_path'] #  useless due to majority of missing value or uninformative features
# df = df.drop(columns=useless_cols)
# print(df.columns)
# print(df.describe().T)
# print(df.isnull().sum())

# Examine numerical features
df['runtime'] = df['runtime'].fillna(0)
# visualize_correlation(df)
# visualize_runtime(df)
# visualize_budget(df)
# visualize_popularity(df)

# Release_date
# visualize_date_year(df)
# visualize_date_month(df)

# Binary feature
# visualize_collection(df)
# visualize_homepage(df)
# visualize_tagline(df)
# visualize_keywords(df)

# contain_xxx feature
# visualize_original_language(df)
# visualize_spoken_languages(df)
# visualize_production_countries(df)

# counting feature
# visualize_title(df)
# visualize_companies(df)
# visualize_cast(df)
visualize_crew(df)

# one-hot encoding feature
# visualize_genre(df)