# Kütüphaneler.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import *
import numpy as np

veriler = pd.read_excel('GlobalData.xlsx')

DataCount = len(veriler.index)

# Veri Ön İşleme

#imputer = SimpleImputer(missing_values=np.nan, strategy='median')

sayisalVeriler = veriler.iloc[:, 22:]

#imputer = imputer.fit(sayisalVeriler)
#sayisalVeriler = imputer.transform(sayisalVeriler)

EvSahibitakimlar = veriler.iloc[:, 2:3]
RakipTakimlar = veriler.iloc[:, 3:4]
FullTimeResult = veriler.iloc[:, 6:7]
#HalfTimeResult = veriler.iloc[:, 9:10]

labelencoder_X = LabelEncoder()

EvSahibitakimlar.values[:, 0] = labelencoder_X.fit_transform(
    EvSahibitakimlar.values[:, 0])
RakipTakimlar.values[:, 0] = labelencoder_X.fit_transform(
    RakipTakimlar.values[:, 0])
# FullTimeResult.values[:, 0] = labelencoder_X.fit_transform(
#   FullTimeResult.values[:, 0])
# HalfTimeResult.values[:, 0] = labelencoder_X.fit_transform(
#   HalfTimeResult.values[:, 0])

sonuc = pd.DataFrame(data=FullTimeResult,
                     index=range(DataCount), columns=['FTR'])
# sonuc2 = pd.DataFrame(data=HalfTimeResult,
#                 index=range(DataCount), columns=['HTR'])
sonuc3 = pd.DataFrame(data=EvSahibitakimlar, index=range(DataCount), columns=['HomeTeam'])
sonuc4 = pd.DataFrame(data=RakipTakimlar, index=range(DataCount), columns=['AwayTeam'])
sonuc5 = pd.DataFrame(data=sayisalVeriler, index=range(DataCount),
                      columns=['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
                               'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH',
                               'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5',
                               'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA',
                               'BbAvAHA', 'PSCH', 'PSCD', 'PSCA'])


sonuc3.reset_index(drop=True, inplace=True)
sonuc4.reset_index(drop=True, inplace=True)
sonuc5.reset_index(drop=True, inplace=True)

s = pd.concat([sonuc3,sonuc4,sonuc5],axis=1)