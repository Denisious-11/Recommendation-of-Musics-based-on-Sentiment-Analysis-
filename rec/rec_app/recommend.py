import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



#read the dataset (based on popularity { users liked })
data=pd.read_csv('static/data.csv.zip',compression='zip')
# print(data)
# print(data.columns)

#remove duplicate song names
data.drop_duplicates(inplace=True,subset=['name'])
name=data['name']
# print(data)

#select features for clustering
col_features = ['danceability', 'energy', 'valence', 'loudness']
#perform feature scaling
X = MinMaxScaler().fit_transform(data[col_features])
#kmeans initialization
kmeans = KMeans(init="k-means++",
                n_clusters=2,
                random_state=15).fit(X)
#add new columns to dataframe
data['kmeans'] = kmeans.labels_
data['song_name']=name
print(data)
print(data.columns)

#clustering
cluster=data.groupby(by=data['kmeans'])

#sorting based on popularity
df=cluster.apply(lambda x: x.sort_values(["popularity"],ascending=False))
print(df)
#df.reset_index(level=0, inplace=True)
df.reset_index(drop=True)

def get_recommendations(emotion):
	NUM_RECOMMEND=5
	# happy_set=[]
	# sad_set=[]
	if emotion==0:
		get_lists1=df[df['kmeans']==0]['song_name']
		print(get_lists1)
		shuffled = get_lists1.sample(frac = 1)
		print(shuffled)
		happy_set=shuffled.head(NUM_RECOMMEND)
		print("*******************************")
		print(happy_set)
		happy_set=happy_set.reset_index(drop=True)
		print("^^^^^^^^^^^^^^^^^^^^^^^^")
		print(happy_set)
		print(type(happy_set))
		happy_set = happy_set.tolist()
		
		return happy_set
		
	else:
		get_lists2=df[df['kmeans']==1]['song_name']
		print(get_lists2)
		shuffled1 = get_lists2.sample(frac = 1)
		print(shuffled1)
		sad_set=shuffled1.head(NUM_RECOMMEND)
		print("*******************************")
		print(sad_set)
		sad_set=sad_set.reset_index(drop=True)
		print("^^^^^^^^^^^^^^^^^^^^^^^^")
		print(sad_set)
		sad_set = sad_set.tolist()

		return sad_set

result=get_recommendations(1)
print("####################")
print(result)
print(type(result))
print(result[0])

