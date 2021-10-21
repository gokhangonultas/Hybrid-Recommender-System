###################################Hybrid Recommender System##########################################


############# Değişkenler ###########
# movie.csv
# movieId - Eşsiz fim numarası(UniqueID)
# title – Film adı
# rating.csv
# userid – Eşsiz kullanıcı numarası. (UniqueID)
# movieId – Eşsiz film numarası. (UniqueID)
# rating – Kullanıcı tarafından filme verilen puan
# timestamp – Değerlendirme tarihi

import pandas as pd
pd.set_option('display.max_columns', 5)

#Görev1: Veri Hazırlama işlemlerini gerçekleştiriniz.

movie = pd.read_csv('Data_sets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Data_sets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


#Görev2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
user = 108170
user_df = user_movie_df[user_movie_df.index == user]
movies_watched = user_df.columns[user_df.notna().any()].tolist()
len(movies_watched)


#Görev3:Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz.

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.head()
users_same_movies.count()

#Görev 4:Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
#Korelasyon Katsayıları

top_users = corr_df[(corr_df["user_id_1"] == user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

#Görev 5:Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

#Önerilen Filmler
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].\
    sort_values("weighted_rating", ascending=False)

#Görev 6:Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.

#5 user-based öneri
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
    sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_name = df[df["movieId"] == movie_id]["title"].values[0]

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][0:5]

#5 item-based önerisi  olacak şekilde 10 öneri yapınız.

def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movies_form_item_based = item_based_recommender(movie_name, user_movie_df)
movies_form_item_based[1:6].index

