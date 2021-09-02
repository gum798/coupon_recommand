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
    kpt.point_after_amt - kpt.point_amt point_amt
    , MONTH(loi.order_date_time) as month
    , HOUR(loi.order_date_time) as hour
    , DAYOFWEEK(loi.order_date_time) as weekday
    , cast(((YEAR(loi.order_date_time) - YEAR(mwu.dob))/5)as int)*5 age
    , mwu.gender
    , lp.category
FROM
    lop.order_info loi
    left outer join (select * from kps.pnt_tran where acc_yn ='N') kpt on loi.payment_nr_number = kpt.nr_number
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
print(results)
cursor.close()

column_name = ['point_amt', 'month', 'hour', 'weekday', 'age', 'gender', 'category']
df = pd.DataFrame(results, columns=column_name)

df.info()
df.head()
#df['point_amt'] = df['point_amt']=='None'

df.isnull().any()
df['point_amt'] = df['point_amt'].fillna(0)
df['point_amt'] = df['point_amt'].astype(int)
df.isnull().any()

df.info()
df = df.dropna()
df.reset_index(drop=True)
df.info()

df['category'].unique()
df['category'] = df['category'].astype('category')
df['category'] = df['category'].cat.reorder_categories(df['category'].unique(), ordered=True)
df['category'] = df['category'].cat.codes
df['category']
df.info()

df['age'] = df['age'].astype('category')
df['month'] = df['month'].astype('category')

bins = [0,4,9,11,14,17,21,24]
bins_label = [0,1,2,3,4,5,6,7]
df['hour_group'] = pd.cut(df['hour'], bins, right=False, labels=bins_label[:-1])
df.head()

df.drop('hour', axis=1, inplace=True)

#df['hour'] = df['hour'].astype('category')
#df['hour']
df.info()

df['Female'] = pd.get_dummies(df['gender'])['Female']
df['Male'] = pd.get_dummies(df['gender'])['Male']
df.drop('gender', axis=1, inplace=True)
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
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")
    
model.summary()
loaded_model = tf.keras.models.load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(tf.expand_dims(X_test, -1))
print(predicted_y)

