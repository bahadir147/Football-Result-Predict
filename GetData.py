# Kütüphaneler.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def GetData(dataFileName):

    # Veri Yükleme
    veriler = pd.read_excel(dataFileName)

    DataCount = len(veriler.index)

    # Veri Ön İşleme

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    sayisalVeriler = veriler.iloc[:, 22:]

    imputer = imputer.fit(sayisalVeriler)
    sayisalVeriler = imputer.transform(sayisalVeriler)

    EvSahibitakimlar = veriler.iloc[:, 2:3]
    RakipTakimlar = veriler.iloc[:, 3:4]
    FullTimeResult = veriler.iloc[:, 6:7]
    HalfTimeResult = veriler.iloc[:, 9:10]

    labelencoder_X = LabelEncoder()

    EvSahibitakimlar.values[:, 0] = labelencoder_X.fit_transform(
        EvSahibitakimlar.values[:, 0])
    RakipTakimlar.values[:, 0] = labelencoder_X.fit_transform(
        RakipTakimlar.values[:, 0])
    FullTimeResult.values[:, 0] = labelencoder_X.fit_transform(
        FullTimeResult.values[:, 0])
    HalfTimeResult.values[:, 0] = labelencoder_X.fit_transform(
        HalfTimeResult.values[:, 0])

    sonuc = pd.DataFrame(data=FullTimeResult,
                         index=range(DataCount), columns=['FTR'])
    # sonuc2 = pd.DataFrame(data=HalfTimeResult,
    #                 index=range(DataCount), columns=['HTR'])
    sonuc3 = pd.DataFrame(data=EvSahibitakimlar, index=range(
        DataCount), columns=['HomeTeam'])
    sonuc4 = pd.DataFrame(data=RakipTakimlar, index=range(
        DataCount), columns=['AwayTeam'])
    sonuc5 = pd.DataFrame(data=sayisalVeriler, index=range(DataCount), columns=['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'PSCH', 'PSCD', 'PSCA'
                                                                                ])

    s = pd.concat([sonuc5])

    x_train, x_test, y_train, y_test = train_test_split(
        s, sonuc, test_size=0.33, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train.astype(float))
    x_test = sc.transform(x_test.astype(float))

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return [x_train, y_train, x_test, y_test]
