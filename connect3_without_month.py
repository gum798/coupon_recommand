# !pip3 install pure_sasl-0.6.2-py3-none-any.whl
# !pip3 install thrift_sasl-0.4.3-py2.py3-none-any.whl
# !pip3 install impyla-0.17.0-py2.py3-none-any.whl

import os
from impala.dbapi import connect
import pandas as pd

IMPALA_HOST = os.getenv('IMPALA_HOST', 'sc-cutil01-sk.konadc.com')

conn = connect(host=IMPALA_HOST
               ,port=21050
               ,use_ssl=False
               ,auth_mechanism='LDAP'
               ,user='jhseo'
               ,password='Hadoop1!'
              )


cursor = conn.cursor()

#cursor.execute('select * from sas.raw_ias_tran_v2 limit 3')
cursor.execute('''
SELECT
    loi.order_date_time date_time
    , TRUNC(DATE_ADD(loi.order_date_time, interval -1 MONTHS), 'W') last_month
    , TRUNC(loi.order_date_time,'W') this_month
    , loi.user_id user_id
    , kpt.point_after_amt - kpt.point_amt point_amt
    -- , MONTH(loi.order_date_time) as month
    -- , HOUR(loi.order_date_time) as hour
    -- , DAYOFWEEK(loi.order_date_time) as weekday
    , cast(((YEAR(loi.order_date_time) - YEAR(mwu.dob))/5)as int)*5 age
    , mwu.gender
    , lp.category
FROM
    lop.order_info loi left outer join (select * from kps.pnt_tran where acc_yn ='N') kpt on loi.payment_nr_number = kpt.nr_number
    , lop.place lp
    , kmap.wallet_user mwu
where
    1=1
    and loi.place_id = lp.id
    and cast(loi.user_id as decimal(19,0)) = mwu.id
    and loi.status = 'OK'
order by  
    loi.order_date_time
;
''')

results = cursor.fetchall()
#print(results)
cursor.close()

column_name = ['date_time','last_month', 'this_month', 'user_id', 'point_amt', 'age', 'gender', 'category']
df = pd.DataFrame(results, columns=column_name)

df.info()
df.head()

df.isnull().any()
df['point_amt'] = df['point_amt'].fillna(0)
df['point_amt'] = df['point_amt'].astype(int)
df.isnull().any()

#count = pd.DataFrame()
count = []
for userId, last_month, this_month in zip(df['user_id'],df['last_month'],df['this_month']):
#  print(userId, last_month, this_month)
  temp = df[df['user_id'] == userId]
  temp = temp[temp['date_time'] >= last_month]
  temp = temp[temp['date_time'] < this_month]
#  print(temp)
  if temp['category'].count() > 0:
    count.append(temp.groupby('user_id').count()['category'][0]-1)
  else:
    count.append(0)
#  print(temp)
#print(count)
df['count'] = pd.DataFrame(count)

df.info()
df
df = df.dropna()
df.reset_index(drop=True)
df.info()

df['category'] = df['category'].astype('category')
df['age'] = df['age'].astype('category')
#df['month'] = df['month'].astype('category')

#bins = [0,4,9,11,14,17,21,24]
#bins_label = [0,1,2,3,4,5,6,7]
#df['hour'] = pd.cut(df['hour'], bins, right=False, labels=bins_label[:-1])
df.head()

df = pd.concat([df, pd.get_dummies(df['gender'])], 1)
df
df.drop('gender', axis=1, inplace=True)
df.drop('date_time', axis=1, inplace=True)
df.drop('last_month', axis=1, inplace=True)
df.drop('this_month', axis=1, inplace=True)
df.drop('user_id', axis=1, inplace=True)
df.head()

import numpy as np
labels = np.array(df.pop('category'))
print(labels)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train.info()

from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
X_train = robustScaler.fit_transform(X_train)
X_test = robustScaler.transform(X_test)
X_train.shape

import tensorflow as tf
import autokeras as ak



clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10)
clf.fit(X_train, y_train, epochs=500, use_multiprocessing=True, workers=8)
model = clf.export_model()
print(type(model))

try:
    model.save("model_autokeras_category_without_month", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")
    
model.summary()
loaded_model = tf.keras.models.load_model("model_autokeras_category_without_month", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(tf.expand_dims(X_test, -1))
print(predicted_y)

