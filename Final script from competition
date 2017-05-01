import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import string
import random
import math

            
train_file = '/home/pavel/anaconda3/Scripts/Two Sigma Rental Listings/train.json'
test_file = '/home/pavel/anaconda3/Scripts/Two Sigma Rental Listings/test.json'

data_image = '/home/pavel/anaconda3/Scripts/Two Sigma Rental Listings/listing_image_time.csv'

new_feature = '/home/pavel/anaconda3/Scripts/Two Sigma Rental Listings/Builiding_manager_features.csv'

train = pd.read_json(train_file)
test = pd.read_json(test_file)
listing_id = test.listing_id.values

data_image = pd.read_csv(data_image)
data_newf = pd.read_csv(new_feature)

# rename columns so you can join tables later on
data_image.columns = ["listing_id", "time_stamp"]

# reassign the only one timestamp from April, all others from Oct/Nov
data_image.loc[80240,"time_stamp"] = 1478129766 
data_image["img_date"]                  = pd.to_datetime(data_image["time_stamp"], unit="s")
data_image["img_days_passed"]           = (data_image["img_date"].max() - data_image["img_date"]).astype("timedelta64[D]").astype(int)
data_image["img_date_month"]            = data_image["img_date"].dt.month
data_image["img_date_week"]             = data_image["img_date"].dt.week
data_image["img_date_day"]              = data_image["img_date"].dt.day
data_image["img_date_dayofweek"]        = data_image["img_date"].dt.dayofweek
data_image["img_date_dayofyear"]        = data_image["img_date"].dt.dayofyear
data_image["img_date_hour"]             = data_image["img_date"].dt.hour
data_image["img_date_monthBeginMidEnd"] = data_image["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)


y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level'] = train['interest_level'].apply(lambda x: y_map[x])
y_train = train.interest_level.values


ntrain = train.shape[0] 

train_test = pd.concat((train, test), axis=0).reset_index(drop=True)

train_test = pd.merge(train_test, data_image, on="listing_id", how="left") # Add Magic Kazanova FEATURE

train_test = pd.merge(train_test,data_newf,on="listing_id",how="left") # Add manager_building features



    features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
                        "img_days_passed", "img_date_month","img_date_week","img_date_day",
                        "img_date_dayofweek", "img_date_dayofyear","img_date_hour","img_date_monthBeginMidEnd"]
    
    # count of photos #
    train_test["num_photos"] = train_test["photos"].apply(len)
    
    # count of "features" #
    train_test["num_features"] = train_test["features"].apply(len)

    train_test["listing_id"] = train_test["listing_id"] - 68119576.0
    
    # count of words present in description column #
    train_test["num_description_words"] = train_test["description"].apply(lambda x: len(x.split(" ")))
    
    train_test["num_price_by_furniture"] = (train_test["price"])/ (train_test["bathrooms"] + train_test["bedrooms"] + 1.0)
    
    
    train_test['price_per_bed'] = train_test['price'] / (train_test['bedrooms'] + 1)
    train_test['price_per_bath'] = train_test['price'] / (train_test['bathrooms'] + 1)
    train_test['price_per_room'] = train_test['price'] / (train_test['bathrooms'] * 0.5 + train_test['bedrooms'] + 1)
        
    train_test["price_latitue"] = (train_test["price"])/ (train_test["latitude"]+1.0)
    
    train_test["price_longtitude"] = (train_test["price"])/ (train_test["longitude"]-1.0)

    train_test["num_furniture"] =  train_test["bathrooms"] + train_test["bedrooms"] 
  
    train_test["num_furniture"] = train_test["num_furniture"].apply(lambda x:  str(x) if float(x)<9.5 else '10')


### GEO LOCATION ###

R = 6373.0     

location_dict = {
    'manhattan_loc': [40.728333, -73.994167],
    'brooklyn_loc': [40.624722, -73.952222],
    'bronx_loc': [40.837222, -73.886111],
    'queens_loc': [40.75, -73.866667],
    'staten_loc': [40.576281, -74.144839]}

for location in location_dict.keys():

    lat1 = train_test['latitude'].apply(np.radians)
    lon1 = train_test['longitude'].apply(np.radians)
    lat2 = np.radians(location_dict[location][0])
    lon2 = np.radians(location_dict[location][1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    def power(x):
        return x**2

    a = (dlat/2).apply(np.sin).apply(power) + lat1.apply(np.cos) * np.cos(lat2) * (dlon/2).apply(np.sin).apply(power)
    c = 2 * a.apply(np.sqrt).apply(np.sin)

    ### Add a new column called distance
    train_test['distance_' + location] = R * c
    features_to_use.append('distance_' + location)

       
    # convert the created column to datetime object so as to extract more features 
    train_test["created"] = pd.to_datetime(train_test["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    train_test["created_month"] = train_test["created"].dt.month
    train_test["created_day"] = train_test["created"].dt.day           
    train_test["created_hour"] = train_test["created"].dt.hour
    train_test["total_days"] =   (train_test["created_month"] -4.0)*30 + train_test["created_day"] +  train_test["created_hour"] /25.0   
    train_test["diff_rank"]= train_test["total_days"]/train_test["listing_id"]
    train_test['created_wday'] = train_test['created'].dt.dayofweek
#######################################################################


train_test['Zero_building_id'] = train_test['building_id'].apply(lambda x: 1 if x == '0' else 0)

train_test['desc'] = train_test['description']
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('!<br /><br />', ''))

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

train_test['desc'] = train_test['desc'].apply(lambda x: x.translate(remove_punct_map))

train_test['desc_letters_count'] = train_test['description'].apply(lambda x: len(x.strip()))
train_test['desc_words_count'] = train_test['desc'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

#######################################################################
text = ''
text_da = ''
text_desc = ''
for ind, row in train_test.iterrows():
    for feature in row['features']:
        text = " ".join([text, "_".join(feature.strip().split(" "))])
    text_da = " ".join([text_da,"_".join(row['display_address'].strip().split(" "))])
    #text_desc = " ".join([text_desc, row['description']])
text = text.strip()  

def newfeat(name, df, series):
    """Create a Series for my feature building loop to fill"""
    feature = pd.Series(0, df.index, name=name)
    """Now populate the new Series with numeric values"""
    for row, word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return(df)

train_test = newfeat('Elevator', train_test, train_test.features)
train_test = newfeat('Hardwood Floors',train_test,train_test.features)
train_test = newfeat('Dogs Allowed', train_test, train_test.features)
train_test = newfeat('Cats Allowed', train_test, train_test.features)
train_test = newfeat('Doorman', train_test, train_test.features)
train_test = newfeat('Dishwasher', train_test, train_test.features)
train_test = newfeat('Laundry in Unit', train_test, train_test.features)
train_test = newfeat('No Fee', train_test, train_test.features)
train_test = newfeat('Laundry in Building', train_test, train_test.features)
train_test = newfeat('Fitness Center', train_test, train_test.features)
train_test = newfeat('Roof Deck', train_test, train_test.features) 
train_test = newfeat('Outdoor Space', train_test, train_test.features)
train_test = newfeat('Dining Room', train_test, train_test.features)
train_test = newfeat('High Speed Internet', train_test, train_test.features)

train_test = newfeat('Swimming Pool', train_test, train_test.features)
train_test = newfeat('Laundry In Building', train_test, train_test.features)
train_test = newfeat('New Construction', train_test, train_test.features)

### Adding some new features ###
train_test = newfeat('Pre-War',train_test, train_test.features)
train_test = newfeat('Exclusive',train_test, train_test.features)


train_test['address1'] = train_test['display_address']
train_test['address1'] = train_test['address1'].apply(lambda x: x.lower())

address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
}


def address_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in address_map:
            out.append(address_map[x])
        else:
            out.append(x)
    return ' '.join(out)



train_test['address1'] = train_test['address1'].apply(lambda x: x.translate(remove_punct_map))
train_test['address1'] = train_test['address1'].apply(lambda x: address_map_func(x))

# Make 6 new binary columns 
new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

for col in new_cols:
    train_test[col] = train_test['address1'].apply(lambda x: 1 if col in x else 0)

### Parsing managers by their listings amount

managers_count = train_test['manager_id'].value_counts()

train_test['top_10_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
train_test['top_25_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
train_test['top_5_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
train_test['top_50_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
train_test['top_1_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
train_test['top_2_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
train_test['top_15_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
train_test['top_20_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
train_test['top_30_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 70)] else 0)

buildings_count = train_test['building_id'].value_counts()

train_test['top_10_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
train_test['top_25_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
train_test['top_5_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
train_test['top_50_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
train_test['top_1_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
train_test['top_2_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
train_test['top_15_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
train_test['top_20_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
train_test['top_30_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)

 del train_test['interest_level'] 
    
 
  train_test.info()  

X_train = train_test.iloc[:ntrain,:]
X_test = train_test.iloc[ntrain:,:]

Y_train = pd.DataFrame(y_train,index=None)

Y_train.columns = ['interest_level']

X_train = pd.concat([X_train,Y_train],axis=1)
   

 ### CV statisticts (GDY5) #####

# 1. Manager Skills Feature
index=list(range(X_train.shape[0]))
random.shuffle(index)
a=[np.nan]*len(X_train)
b=[np.nan]*len(X_train)
c=[np.nan]*len(X_train)

for i in range(5):
    building_level={}
    for j in X_train['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*X_train.shape[0])/5):int(((i+1)*X_train.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=X_train.iloc[j]
        if temp['interest_level']==2:
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']==0:
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=X_train.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
X_train['manager_level_low']=a
X_train['manager_level_medium']=b
X_train['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in X_train['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(X_train.shape[0]):
    temp=X_train.iloc[j]
    if temp['interest_level']==2:
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']==1:
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']==0:
        building_level[temp['manager_id']][2]+=1

for i in X_test['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
X_test['manager_level_low']=a
X_test['manager_level_medium']=b
X_test['manager_level_high']=c


del X_train['interest_level']
 
 # 2. Building Popularity Feature #index=list(range(X_train.shape[0]))
index=list(range(X_train.shape[0]))
random.shuffle(index)
a=[np.nan]*len(X_train)
b=[np.nan]*len(X_train)
c=[np.nan]*len(X_train)

for i in range(5):
    building_level={}
    for j in X_train['building_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*X_train.shape[0])/5):int(((i+1)*X_train.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=X_train.iloc[j]
        if temp['interest_level']==2:
            building_level[temp['building_id']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['building_id']][1]+=1
        if temp['interest_level']==0:
            building_level[temp['building_id']][2]+=1
    for j in test_index:
        temp=X_train.iloc[j]
        if sum(building_level[temp['building_id']])!=0:
            a[j]=building_level[temp['building_id']][0]*1.0/sum(building_level[temp['building_id']])
            b[j]=building_level[temp['building_id']][1]*1.0/sum(building_level[temp['building_id']])
            c[j]=building_level[temp['building_id']][2]*1.0/sum(building_level[temp['building_id']])
X_train['building_level_low']=a
X_train['building_level_medium']=b
X_train['building_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in X_train['building_id'].values:
    building_level[j]=[0,0,0]
for j in range(X_train.shape[0]):
    temp=X_train.iloc[j]
    if temp['interest_level']==2:
        building_level[temp['building_id']][0]+=1
    if temp['interest_level']==1:
        building_level[temp['building_id']][1]+=1
    if temp['interest_level']==0:
        building_level[temp['building_id']][2]+=1

for i in X_test['building_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
X_test['building_level_low']=a
X_test['building_level_medium']=b
X_test['building_level_high']=c

del X_train['interest_level']  
  
X_train.info()  

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

X_train['building_popularity'] = X_train['building_level_high']*2 + X_train['building_level_medium']
X_test['building_popularity'] = X_test['building_level_high']*2 + X_test['building_level_medium']

X_train['manager_skill'] = X_train['manager_level_high']*2 + X_train['manager_level_medium']
X_test['manager_skill'] = X_test['manager_level_high']*2 + X_test['manager_level_medium']

X_train['rooms'] = X_train['bathrooms'] + X_train['bedrooms'] 
X_test['rooms'] = X_test['bathrooms'] + X_test['bedrooms'] 

X_train['Laundry in Building'] = X_train['Laundry in Building'] + X_train['Laundry In Building']
X_test['Laundry in Building'] = X_test['Laundry in Building'] + X_test['Laundry In Building']

del X_train['Laundry In Building']
del X_test['Laundry In Building']

# del features_to_use

 list(X_train.columns)
#####################################################################    
    categorical = [ "display_address", "manager_id", "building_id","street_address","num_furniture"]#,"num_furniture","latitude_binned"]#"", "","street_address"
    lencat=len(categorical)

    for f in range (0,lencat):
        for s in range (f+1,lencat): 
            X_train[categorical[f] + "_" +categorical[s]] =X_train[categorical[f]]+"_" + X_train[categorical[s]]
            X_test[categorical[f] + "_" +categorical[s]] =X_test[categorical[f]]+"_" + X_test[categorical[s]]           
            categorical.append(categorical[f] + "_" +categorical[s])
       
    # adding all these new features to use list #
    features_to_use.extend(["num_photos", "num_features", "num_description_words", "created_month", "created_day",
                            "listing_id", "created_hour","total_days","diff_rank","created_wday",#"listing_rank","total_days_rank",
                            "num_price_by_furniture","price_latitue","price_longtitude","price_per_bed",
                            "price_per_bath","price_per_room","Zero_building_id","desc_letters_count",
                            "desc_words_count","Elevator","Hardwood Floors","Dogs Allowed","Cats Allowed",
                            "Doorman","Dishwasher","Laundry in Unit","No Fee","Laundry in Building",
                            "Fitness Center","Roof Deck","Outdoor Space","Dining Room","High Speed Internet",
                            "Swimming Pool","New Construction","street","avenue",
                            "east","west","north","south","top_10_manager","top_25_manager",
                            "top_5_manager","top_50_manager","top_1_manager","top_2_manager",
                            "top_15_manager","top_20_manager","top_30_manager","top_10_building",
                            "top_25_building","top_5_building","top_50_building","top_1_building",
                            "top_2_building","top_15_building","top_20_building","top_30_building",
                            "manager_level_low","manager_level_medium","manager_level_high",
                      #      "building_level_low","building_level_medium","building_level_high",
                     #       "building_popularity",
                            "manager_skill","rooms","Pre-War","Exclusive",
                            "building_id_mean_medium", "building_id_mean_high",
                            "manager_id_mean_medium", "manager_id_mean_high"])

 
    result = pd.concat([X_train,X_test])



    for f in categorical:
            if X_train[f].dtype=='object':

                cases=defaultdict(int)
                temp=np.array(result[f]).tolist()
                for k in temp:
                    cases[k]+=1
                print (f, len(cases) )
                
                X_train[f]=X_train[f].apply(lambda x: cases[x])
                X_test[f]=X_test[f].apply(lambda x: cases[x])               
                
                features_to_use.append(f)  

    X_train['features'] =  X_train['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    X_test['features'] = X_test['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))  

    X_train['description'] =  X_train['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc") 
    X_test['description'] = X_test['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc") 
    
    tfidfdesc=TfidfVectorizer(min_df=20, max_features=50, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), use_idf=False,smooth_idf=False, 
                        sublinear_tf=True, stop_words = 'english')  
    
    print(X_train["features"].head())
       
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    
    te_sparse = tfidf.fit_transform (X_test["features"])  
    tr_sparse = tfidf.transform(X_train["features"])   

    te_sparsed = tfidfdesc. fit_transform (X_test["description"])  
    tr_sparsed = tfidfdesc.transform(X_train["description"])
    print(features_to_use)
    

    train_X = sparse.hstack([X_train[features_to_use], tr_sparse,tr_sparsed]).tocsr() #
    test_X = sparse.hstack([X_test[features_to_use], te_sparse,te_sparsed]).tocsr()#
  # Done !
  

    print(train_X.shape, test_X.shape)    


#create average value of the target variabe given a categorical feature        
def convert_dataset_to_avg(xc,yc,xt, rounding=2,cols=None):
    xc=xc.tolist()
    xt=xt.tolist()
    yc=yc.tolist()
    if cols==None:
        cols=[k for k in range(0,len(xc[0]))]
    woe=[ [0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good=[]
    bads=[]
    for col in cols:
        dictsgoouds=defaultdict(int)        
        dictsbads=defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)        
    total_count=0.0
    total_sum =0.0

    for a in range (0,len(xc)):
        target=yc[a]
        total_sum+=target
        total_count+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            good[j][round(xc[a][col],rounding)]+=target
            bads[j][round(xc[a][col],rounding)]+=1.0  
            
    
    for a in range (0,len(xt)):    
        for j in range(0,len(cols)):
            col=cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j]=float(good[j][round(xt[a][col],rounding)])/float(bads[j][round(xt[a][col],rounding)])  
            else :
                 woe[a][j]=round(total_sum/total_count)
    return woe            
    

#converts the select categorical features to numerical via creating averages based on the target variable within kfold. 
def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]    
      
    X=X.tolist()
    Xt=Xt.tolist() 
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]    
    
    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1      
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns) 

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):           
            woetest[real_index][j]=Xt[real_index][j]
            
    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]
            
    return np.array(woetrain), np.array(woetest)

        
    

        #training and test files, created using SRK's python script
        train_file="train_stacknet.csv"
        test_file="test_stacknet.csv"
        
        ######### Load files ############

        #create to numpy arrays (dense format)        
        X=train_X.toarray()
        X_test=test_X.toarray()  
        y = y_train
        ids = listing_id
        
        print ("scalling") 
        #scale the data
        stda=StandardScaler()  
        X_test=stda.fit_transform (X_test)          
        X=stda.transform(X)

        
        CO=[0,14,21] # columns to create averages on
        
        #Create Arrays for meta
        train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(X.shape[0])) ]
        test_stacker=[[0.0 for s in range(3)]   for k in range (0,(X_test.shape[0]))]
        
        number_of_folds=5 # number of folds to use
        print("kfolder")
        #cerate 5 fold object
        mean_logloss = 0.0
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=123)   # Change on KFold

        #xgboost_params
        param = {}
        param['booster']='gbtree'
        param['objective'] = 'multi:softprob'
        param['bst:eta'] = 0.03 # 0.04
        param['seed']=  1
        param['bst:max_depth'] = 6
        param['bst:min_child_weight']= 1.
        param['silent'] =  1  
        param['nthread'] = 12 
        param['bst:subsample'] = 0.7
        param['gamma'] = 1.0
        param['colsample_bytree']= 1.0
        param['num_parallel_tree']= 3   
        param['colsample_bylevel']= 0.7                  
        param['lambda']=5  
        param['num_class']= 3 


        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                #create past averages for some fetaures
                W_train,W_cv=convert_to_avg(X_train,y_train, X_cv, seed=1, cvals=5, roundings=2, columns=CO)
                W_train=np.column_stack((X_train,W_train[:,CO]))
                W_cv=np.column_stack((X_cv,W_cv[:,CO])) 
                print (" train size: %d. test size: %d, cols: %d " % ((W_train.shape[0]) ,(W_cv.shape[0]) ,(W_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(csr_matrix(W_train), label=np.array(y_train),missing =-999.0)
                X1cv=xgb.DMatrix(csr_matrix(W_cv), missing =-999.0)
                bst = xgb.train(param.items(), X1, 1000) 
                #predictions
                predictions = bst.predict(X1cv)     
                preds=predictions.reshape( W_cv.shape[0], 3)

   
                logs = log_loss(y_cv,preds)
                print ("size train: %d size cv: %d loglikelihood (fold %d/%d): %f" % ((W_train.shape[0]), (W_cv.shape[0]), i + 1, number_of_folds, logs))
             
                mean_logloss += logs

                no=0
                for real_index in test_index:
                    for d in range (0,3):
                        train_stacker[real_index][d]=(preds[no][d])
                    no+=1
                i+=1
        mean_logloss/=number_of_folds
        print (" Average Lolikelihood: %f" % (mean_logloss) )
                
        #calculating averages for the train data
        W,W_test=convert_to_avg(X,y, X_test, seed=2, cvals=5, roundings=2, columns=CO)
        W=np.column_stack((X,W[:,CO]))
        W_test=np.column_stack((X_test,W_test[:,CO]))          
        #X_test=np.column_stack((X_test,woe_cv))      
        print (" making test predictions ")
        
        X1=xgb.DMatrix(csr_matrix(W), label=np.array(y) , missing =-999.0)
        X1cv=xgb.DMatrix(csr_matrix(W_test), missing =-999.0)
        bst = xgb.train(param.items(), X1, 1000) 
        predictions = bst.predict(X1cv)     
        preds=predictions.reshape( W_test.shape[0], 3)        
       
        for pr in range (0,len(preds)):  
                for d in range (0,3):            
                    test_stacker[pr][d]=(preds[pr][d]) 
        
        
        
        print ("merging columns")   
        #stack xgboost predictions
        X=np.column_stack((X,train_stacker))
        # stack id to test
        X_test=np.column_stack((X_test,test_stacker))        
        
        # stack target to train
        X=np.column_stack((y,X))
        # stack id to test
        X_test=np.column_stack((ids,X_test))
        
        #export to txt files (, del.)
        print ("exporting files")
        np.savetxt(train_file, X, delimiter=",", fmt='%.5f')
        np.savetxt(test_file, X_test, delimiter=",", fmt='%.5f')        


                             
                  

