###############################################################
# B2B INVOICE LATE PAYMENT PREDICTION
# AND COLLECTION PRIORITIZATION
###############################################################
"""
İŞ PROBLEMİ

B2B şirketlerde müşterilerin faturalarını vadesinden sonra ödemesi,
şirketlerin nakit akışı yönetimini zorlaştırabilir ve tahsilat
ekiplerinin sınırlı kaynaklarını verimsiz kullanmasına neden olabilir.

Bu projede geçmiş fatura ve müşteri ödeme bilgilerinden yararlanılarak
bir faturanın vadesinden sonra ödenme olasılığının tahmin edilmesi
amaçlanmaktadır.

Makine öğrenmesi modeli ile elde edilen geç ödeme olasılıkları,
faturaların finansal büyüklükleriyle birlikte değerlendirilerek
tahsilat ekiplerinin hangi açık faturaları öncelikli olarak takip
etmesi gerektiğini belirleyen bir karar destek sistemi oluşturulacaktır.
"""
###############################################################
# DEĞİŞKENLER
###############################################################

"""
business_code:
Faturanın ilişkili olduğu işletme/şirket birimini temsil eden kod.
Örnek değerler: U001, CA02.

cust_number:
Müşteriyi tanımlayan müşteri numarasıdır.
Aynı müşteriye ait geçmiş faturaları belirlemek için kullanılabilir.

name_customer:
Müşterinin adını/unvanını ifade eder.

clear_date:
Faturanın gerçek ödeme/kapanış tarihidir.

buisness_year:
Faturanın ait olduğu iş/mali yılı ifade eder.

doc_id:
Muhasebe veya fatura belgesini tanımlayan belge numarasıdır.
Büyük ölçüde benzersiz bir ID niteliğindedir.

posting_date:
Faturanın muhasebe sistemine işlendiği tarihi ifade eder.

document_create_date:
Belgenin oluşturulduğu tarihi ifade eder.

document_create_date.1:
Veri setinde bulunan ikinci belge oluşturma tarihi değişkenidir.


due_in_date:
Faturanın son ödeme/vade tarihidir.

invoice_currency:
Faturanın para birimini ifade eder.
Örnek değerler: USD, CAD.

document type:
Muhasebe belgesinin türünü temsil eden koddur.
Örnek değer: RV.


posting_id:
Muhasebe kayıt süreciyle ilişkili bir ID/kod değişkenidir.


area_business:
İş alanı/bölgesi ile ilişkili bir değişken olarak adlandırılmıştır.

total_open_amount:
Faturanın parasal tutarını ifade eder.

baseline_create_date:
Ödeme vadesinin hesaplanmasında kullanılan başlangıç/baz tarihini
ifade eden tarih değişkenidir.

cust_payment_terms:
Müşteriye uygulanan ödeme koşullarını/vade koşullarını temsil eden
koddur.
Örnek değerler: NAH4, NAD1, NAA8, CA10.

invoice_id:
Faturayı tanımlayan benzersiz veya büyük ölçüde benzersiz fatura
numarasıdır.

isOpen:
Faturanın açık olup olmadığını gösterir.

0 = kapanmış/ödenmiş fatura
1 = açık fatura
"""
###############################################################
# TARGET VARIABLE
###############################################################

"""
LATE_PAYMENT:

Projede oluşturulacak hedef değişkendir.

0 = Fatura vadesinde veya vadesinden önce ödendi.
1 = Fatura vadesinden sonra ödendi.

Kapalı faturalar için clear_date ile due_in_date karşılaştırılarak
oluşturulacaktır.

clear_date <= due_in_date  -> LATE_PAYMENT = 0
clear_date > due_in_date   -> LATE_PAYMENT = 1
"""


###############################################################
# 1. LIBRARIES
###############################################################

import joblib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


###############################################################
# 2. LOAD DATASET
###############################################################


from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "dataset.csv"
df = pd.read_csv(DATA_PATH)


###############################################################
# 3. INITIAL DATA CHECK
###############################################################

def check_df(dataframe, head=5):

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1],numeric_only=True).T)


check_df(df)



###############################################################
# 4. CLEAR_DATE - ISOPEN CONTROL
###############################################################

# clear_date değişkenindeki eksikliklerin açık faturalarla ilişkili olup olmadığını kontrol ediyoruz.

print(pd.crosstab(df["isOpen"],df["clear_date"].isnull()))


# SONUÇ:
#
# isOpen = 0 olan kapanmış faturaların clear_date bilgisi bulunmaktadır.
# isOpen = 1 olan açık faturaların clear_date bilgisi bulunmamaktadır.
#
# Dolayısıyla clear_date değişkenindeki eksiklikler rastgele değildir.
# Açık faturalar henüz ödenmediği için ödeme tarihi bilinmemektedir.




###############################################################
# 5. COLUMN NAME STANDARDIZATION
###############################################################

df.rename(columns={

    "cust_number": "customer_id",
    "name_customer": "customer_name",
    "buisness_year": "business_year",
    "doc_id": "document_id",
    "document_create_date.1":"document_create_date_1",
    "due_in_date": "due_date",
    "document type": "document_type",
    "area_business": "business_area",
    "cust_payment_terms":"customer_payment_terms",
    "isOpen": "is_open"
}, inplace=True)



###############################################################
# 6. DUPLICATE CONTROL
###############################################################

print("Duplicate rows before cleaning:",df.duplicated().sum())

# Tamamen aynı olan 1661 tane duplicate kayıtları kaldırıyoruz.

df.drop_duplicates(inplace=True)

# Tekrar kontrol ediyoruz.

print("Duplicate rows after cleaning:",df.duplicated().sum())

###############################################################
# 7. DATA TYPE CONVERSION
###############################################################

# İlk veri kontrolünde bazı tarih değişkenlerinin object, int veya float olarak tutuluyordu.

# Gerçekte tarih bilgisi taşıyan değişkenler datetime formatına dönüştürülür.

df["clear_date"] = pd.to_datetime(df["clear_date"],errors="coerce")


df["posting_date"] = pd.to_datetime(df["posting_date"],errors="coerce")


date_cols_numeric = ["document_create_date","document_create_date_1","due_date","baseline_create_date"]

for col in date_cols_numeric:

    df[col] = pd.to_datetime(

        df[col]
        .astype("Int64")
        .astype("string"),

        format="%Y%m%d",
        errors="coerce"

    )


df["business_year"] = (df["business_year"].astype("Int64"))


# Veri tipi dönüşümlerini kontrol ediyoruz.
date_cols = [
    "clear_date",
    "posting_date",
    "document_create_date",
    "document_create_date_1",
    "due_date",
    "baseline_create_date"
]

print(df.dtypes)
print(df[date_cols].head())


###############################################################
# 8. DATA CLEANING
###############################################################

# business_area:
# Tüm gözlemleri eksik olduğu için bilgi taşımamaktadır.

# posting_id:
# Tüm gözlemlerde aynı değeri taşıdığı için ayırt edici bilgi sağlamamaktadır.


drop_cols = ["business_area","posting_id", "document_type"]

df.drop(columns=drop_cols,inplace=True)

check_df(df)




###############################################################
# 10. CLOSED AND OPEN INVOICES
###############################################################

# Temizlenmiş veri setini faturanın açık/kapalı durumuna göre iki gruba ayırıyoruz.


closed_df = df[df["is_open"] == 0].copy()

open_df = df[df["is_open"] == 1].copy()

print("Closed invoices:",closed_df.shape)
print("Open invoices:",open_df.shape)

# closed_df:
# Sonucunu bildiğimiz geçmiş/kapanmış faturalar.
# clear_date bilgisi bulunmaktadır.
# Model geliştirme aşamasında kullanılacaktır.

# open_df:
# Henüz sonucu bilinmeyen açık faturalar.
# clear_date bilgisi bulunmamaktadır.
# Model tamamlandıktan sonra tahmin yapılacaktır.

print(closed_df[["clear_date", "due_date"]].isnull().sum())



###############################################################
# 11. TARGET CREATION
###############################################################

# LATE_PAYMENT yalnızca closed_df üzerinde oluşturulabilir.
# Çünkü sadece kapanmış faturaların gerçek ödeme tarihi olan clear_date bilinmektedir.


closed_df["LATE_PAYMENT"] = (closed_df["clear_date"]>closed_df["due_date"]).astype(int)


# LATE_PAYMENT = 0
# Fatura vadesinde veya vadesinden önce ödendi.

# LATE_PAYMENT = 1
# Fatura vadesinden sonra ödendi.



###############################################################
# 12. TARGET DISTRIBUTION
###############################################################

print("##################### Target Counts #####################")
print(closed_df["LATE_PAYMENT"].value_counts())
print("##################### Target Ratios #####################")
print(closed_df["LATE_PAYMENT"].value_counts(normalize=True)* 100)

# TARGET DISTRIBUTION - FINDINGS
#
# Kapalı faturaların 22.740'ı (%58.07) vadesinde veya vadesinden
# önce ödenirken, 16.418'i (%41.93) vadesinden sonra ödenmiştir.
#
# Target sınıfları arasında ciddi bir dengesizlik bulunmamaktadır.
# Bu nedenle mevcut aşamada özel bir class balancing işlemine
# ihtiyaç olduğu görülmemektedir.
#
# Faturaların yaklaşık %42'sinin geç ödenmiş olması, geç ödeme
# probleminin veri setinde anlamlı bir büyüklüğe sahip olduğunu
# ve tahsilat riskinin tahmin edilmesinin iş açısından önemli
# olduğunu göstermektedir.

###############################################################
# CATEGORICAL VARIABLE ANALYSIS
###############################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


###############################################################
#  NUMERICAL VARIABLE ANALYSIS
###############################################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

###############################################################
# 13.3 TARGET ANALYSIS WITH NUMERICAL VARIABLES
###############################################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


###############################################################
# 13.4 TARGET ANALYSIS WITH CATEGORICAL VARIABLES
###############################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")




###############################################################
# GRAB COLUMN NAMES
###############################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Değişken türlerinin ayrıştırılması

cat_cols, num_cols, cat_but_car = grab_col_names(closed_df,cat_th=5,car_th=20)

# Projeye özel değişken gruplarının düzenlenmesi

date_cols = [
    "clear_date",
    "posting_date",
    "document_create_date",
    "document_create_date_1",
    "due_date",
    "baseline_create_date"
]

id_cols = [
    "customer_id",
    "customer_name",
    "document_id",
    "invoice_id"
]


# customer_payment_terms iş anlamı gereği kategorik değişkendir.
cat_cols.append("customer_payment_terms")


# Target ve sabit değişken kategorik analizden çıkarılır.
cat_cols = [
    col for col in cat_cols
    if col not in ["is_open", "LATE_PAYMENT"]
]


# Tarih ve ID değişkenleri numerik analizden çıkarılır.
num_cols = [
    col for col in num_cols
    if col not in date_cols + id_cols
]


# customer_payment_terms artık kategorik değişken olarak ele alınacağı
# için cardinal listesinden çıkarılır.
cat_but_car = [
    col for col in cat_but_car
    if col != "customer_payment_terms"
]


print("Categorical Variables:")
print(cat_cols)

print("\nNumerical Variables:")
print(num_cols)

print("\nID Variables:")
print(id_cols)

print("\nDate Variables:")
print(date_cols)

print("\nCategorical but Cardinal Variables:")
print(cat_but_car)


############## Categorical Variable Analysis###############

for col in cat_cols:
    cat_summary(closed_df, col)

# CATEGORICAL VARIABLES - FINDINGS
#
# business_code değişkeninde U001 yaklaşık %90 oranıyla baskın
# kategoridir. Bazı business_code kategorilerinin gözlem sayısı
# oldukça düşüktür. Bu kategoriler ilerleyen aşamalarda rare category
# açısından değerlendirilecektir.
#
# invoice_currency değişkeninde faturaların yaklaşık %92'si USD,
# %8'i CAD para birimindedir.
#
# document_type değişkeninin yaklaşık %99.98'i RV kategorisinden
# oluşmaktadır. Değişkenin varyansı çok düşük olduğu için modele
# ayırt edici bilgi sağlama ihtimali sınırlıdır ve drop adayı olarak
# değerlendirilecektir.
#
# business_year değişkeninde gözlemlerin yaklaşık %86.8'i 2019,
# %13.2'si 2020 yılına aittir.
#
# customer_payment_terms değişkeninde 70 farklı kategori bulunmaktadır.
# Bazı ödeme koşulları oldukça sık görülürken bazıları yalnızca birkaç
# gözlemde bulunmaktadır. Ödeme koşulları iş problemi açısından önemli
# olabileceğinden değişken korunacak, düşük frekanslı kategoriler
# ilerleyen aşamalarda rare category açısından değerlendirilecektir.

################### Numerical Variable Analysis################

num_summary(closed_df, "total_open_amount", plot=True)

# NUMERICAL VARIABLE ANALYSIS - FINDINGS
#
# total_open_amount değişkeninde 39.158 gözlemin tamamında veri
# bulunmaktadır ve eksik değer yoktur.
#
# Ortalama fatura tutarı yaklaşık 32.058 iken medyan yaklaşık
# 17.246'dır. Ortalamanın medyandan belirgin şekilde yüksek olması,
# dağılımın sağa çarpık olduğunu göstermektedir.
#
# Faturaların %90'ı yaklaşık 81.795'in, %95'i 110.980'in ve
# %99'u 166.083'ün altındadır. Maksimum fatura tutarı ise
# 668.593'tür.
#
# total_open_amount değişkeninin geniş bir dağılıma sahip olması,
# fatura büyüklüğünün geç ödeme davranışı ve tahsilat
# önceliklendirmesi açısından önemli bir değişken olabileceğini
# göstermektedir.

######################Target Analysis with Numerical Variables##############

target_summary_with_num(closed_df,"LATE_PAYMENT","total_open_amount")

# TARGET ANALYSIS WITH NUMERICAL VARIABLE - FINDINGS
#
# Zamanında veya erken ödenen faturaların ortalama tutarı yaklaşık
# 30.549 iken, geç ödenen faturaların ortalama tutarı yaklaşık
# 34.149'dur.
#
# Geç ödenen faturaların ortalama tutarı bir miktar daha yüksek
# olmakla birlikte iki grup arasında belirgin bir ayrışma
# görülmemektedir.
#
# Bu nedenle yalnızca ortalama değerler üzerinden total_open_amount
# ile geç ödeme arasında güçlü bir ilişki olduğu söylenemez.

######################Target Analysis with Categorical Variables##############

for col in cat_cols:
    target_summary_with_cat(closed_df, "LATE_PAYMENT", col)

# TARGET ANALYSIS WITH CATEGORICAL VARIABLES - FINDINGS
#
# business_code değişkeninde kategoriler arasında belirgin geç ödeme
# oranı farklılıkları görülmektedir. Özellikle CA02 grubunda geç ödeme
# oranı U001 grubuna göre oldukça yüksektir.
#
# invoice_currency değişkeninde CAD faturalarının geç ödeme oranı
# USD faturalarına göre belirgin şekilde daha yüksektir.
# Ancak CAD ile CA02 business_code dağılımlarının oldukça benzer olması
# nedeniyle bu iki değişken arasındaki ilişki ayrıca incelenecektir.
#
# document_type değişkeninde X2 kategorisinin geç ödeme oranı yüksek
# görünse de yalnızca çok az gözlem içerdiği için bu oran güvenilir
# değildir. Değişkenin büyük bölümü RV kategorisinden oluştuğundan
# ayırt edici gücü sınırlı görünmektedir.
#
# business_year değişkeninde 2019 ve 2020 yıllarının geç ödeme
# oranları birbirine oldukça yakındır. Bu nedenle yıl değişkeninin
# tek başına güçlü bir ayrıştırıcı olmadığı görülmektedir.
#
# customer_payment_terms kategorileri arasında belirgin geç ödeme
# oranı farklılıkları bulunmaktadır. Ancak bazı kategorilerin gözlem
# sayısı çok düşük olduğundan oranlar kategori frekanslarıyla birlikte
# değerlendirilmelidir.


#business_code ile invoice_currency çok büyük ölçüde aynı bilgiyi taşıyor mu kontrolü

pd.crosstab(closed_df["business_code"],closed_df["invoice_currency"])

# REDUNDANT VARIABLE ANALYSIS - BUSINESS CODE & INVOICE CURRENCY

# business_code ile invoice_currency arasında çok güçlü bir ilişki
# olduğu görülmüştür.

# CA02 koduna sahip 3.122 faturanın tamamı CAD, U001, U002,
# U005 ve U007 kodlarına sahip faturaların tamamı USD'dir.
# U013 grubunda ise yalnızca 2 CAD fatura bulunmaktadır.

# Dolayısıyla invoice_currency değişkeni business_code tarafından
# neredeyse tamamen temsil edilmektedir.

# business_code daha ayrıntılı bilgi içerdiği için korunacak,
# invoice_currency ise redundant değişken olarak modelleme öncesinde
# drop adayı olarak değerlendirilecektir.

################################################
#  Data Preprocessing & Feature Engineering
################################################

###############################################################
# 14.1 WORKING DATASET
###############################################################

model_df = closed_df.copy()

###############################################################
# 14.2 MISSING VALUES - ORİJİNAL DEĞİŞKENLER
###############################################################

print(model_df.isnull().sum())

# 6 missing value  invoice_id içerisinde

###############################################################
# 14.3 FEATURE CREATION
###############################################################

#  Tarih bazlı değişkenler
model_df["payment_term_days"] = (model_df["due_date"] - model_df["baseline_create_date"]).dt.days
model_df["posting_month"] = model_df["posting_date"].dt.month
model_df["due_month"] = model_df["due_date"].dt.month

#  Müşteri geçmişi
model_df = model_df.sort_values(["customer_id", "posting_date"]).reset_index(drop=True)

model_df["delay_days_raw"] = (model_df["clear_date"] - model_df["due_date"]).dt.days.clip(lower=0)
grp = model_df.groupby("customer_id")

model_df["cust_past_invoice_count"] = grp.cumcount()
model_df["cust_late_payment_rate_hist"] = grp["LATE_PAYMENT"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
model_df["cust_avg_delay_days_hist"] = grp["delay_days_raw"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
model_df["cust_last_n_late_ratio"] = grp["LATE_PAYMENT"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
model_df["days_since_last_invoice"] = grp["posting_date"].diff().dt.days

model_df.drop(columns=["delay_days_raw"], inplace=True)

check_df(model_df)

###############################################################
# 14.4 MISSING VALUES - YENİ TÜRETİLEN FEATURE'LAR
###############################################################

global_late_rate = model_df["LATE_PAYMENT"].mean()   #tüm veri setindeki genel geç ödeme oranı

model_df["cust_late_payment_rate_hist"] = model_df["cust_late_payment_rate_hist"].fillna(global_late_rate)
model_df["cust_last_n_late_ratio"] = model_df["cust_last_n_late_ratio"].fillna(global_late_rate)
model_df["cust_avg_delay_days_hist"] = model_df["cust_avg_delay_days_hist"].fillna(0)
model_df["days_since_last_invoice"] = model_df["days_since_last_invoice"].fillna(0)

print(model_df.isnull().sum().sum())   # 6 missing value

missing = model_df.isnull().sum()
print(missing[missing > 0]) # 6 missing value in invoice_id


###############################################################
# 14.5 OUTLIER (original features + new created numerical features)
###############################################################

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "LATE_PAYMENT":
      print(col, check_outlier(model_df, col))


# Aykırı değerlerin baskılanması
outlier_steps = []

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    outlier_steps.append({
        "column": variable,
        "low_limit": low_limit,
        "up_limit": up_limit
    })

    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


for col in num_cols:
    if col != "LATE_PAYMENT":
        replace_with_thresholds(model_df,col)


num_cols_updated = [
    "total_open_amount", "payment_term_days",
    "cust_avg_delay_days_hist", "days_since_last_invoice"
]

for col in num_cols_updated:
    print(col, check_outlier(model_df, col))

for col in num_cols_updated:
    replace_with_thresholds(model_df, col)


###############################################################
# 14.6 RARE ENCODING
###############################################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                             "RATIO": dataframe[col].value_counts() / len(dataframe),
                             "TARGET_MEAN": dataframe.groupby(col)["LATE_PAYMENT"].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc, cat_cols):

    temp_df = dataframe.copy()
    rare_rules = {}

    rare_columns = [
        col for col in cat_cols
        if (temp_df[col].value_counts() / len(temp_df) < rare_perc).sum() > 1
    ]

    for col in rare_columns:

        tmp = temp_df[col].value_counts() / len(temp_df)

        rare_labels = tmp[tmp < rare_perc].index.tolist()

        rare_rules[col] = rare_labels

        temp_df[col] = np.where(
            temp_df[col].isin(rare_labels),
            "Rare",
            temp_df[col]
        )

    return temp_df, rare_rules

rare_analyser(model_df, "LATE_PAYMENT", cat_cols)

model_df, rare_rules = rare_encoder(model_df,0.01,cat_cols)
# %1 altındakiler "Rare"

rare_analyser(model_df, "LATE_PAYMENT", cat_cols)   # kontrol


###############################################################
# 14.7 ONE-HOT ENCODING
###############################################################

ohe_cols = [col for col in cat_cols if col not in ["is_open"]]

known_categories = {col: model_df[col].dropna().unique().tolist()for col in ohe_cols}

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


model_df = one_hot_encoder(model_df, ohe_cols, drop_first=True)

# TIME-BASED SPLIT İÇİN INDEXLERİ KAYDETME

time_order = model_df.sort_values("posting_date").index

split_index = int(len(time_order) * 0.80)

train_index = time_order[:split_index]
test_index = time_order[split_index:]

###############################################################
# 14.8 MODELLEMEYE GİRMEYECEK SÜTUNLARI ÇIKARMA
###############################################################

drop_before_model = [
    "customer_name", "document_id", "invoice_id",
    "clear_date", "posting_date", "document_create_date",
    "document_create_date_1", "due_date", "baseline_create_date",
    "invoice_currency", "is_open"
]

final_df = model_df.drop(columns=[c for c in drop_before_model if c in model_df.columns])

print(final_df.isnull().sum().sum())

###############################################################
# 15.1 FINAL DATA CONTROL
###############################################################

# customer_id modelde doğrudan kullanılmayacak.
final_df.drop(columns=["customer_id"], inplace=True, errors="ignore")


invoice_currency_cols = [col for col in final_df.columns if col.startswith("invoice_currency_")]

final_df.drop(columns=invoice_currency_cols,inplace=True,errors="ignore")

# Boolean dummy değişkenleri 0-1 integer yapıyoruz.
bool_cols = final_df.select_dtypes(include="bool").columns
final_df[bool_cols] = final_df[bool_cols].astype(int)

print("Final dataframe shape:", final_df.shape)

print("\nMissing Values:")
print(final_df.isnull().sum().sum())

print("\nObject Columns:")
print(final_df.select_dtypes(include="object").columns.tolist())

print("\nTarget Distribution:")
print(final_df["LATE_PAYMENT"].value_counts())
print(final_df["LATE_PAYMENT"].value_counts(normalize=True))

###############################################################
# 15.2 DEPENDENT AND INDEPENDENT VARIABLES
###############################################################

y = final_df["LATE_PAYMENT"]

X = final_df.drop("LATE_PAYMENT", axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)


###############################################################
# 15.3 TRAIN - TEST SPLIT
###############################################################

X_train = X.loc[train_index]
X_test = X.loc[test_index]

y_train = y.loc[train_index]
y_test = y.loc[test_index]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain target:")
print(y_train.value_counts(normalize=True))

print("\nTest target:")
print(y_test.value_counts(normalize=True))

###############################################################
# 15.4 BASE MODELS
###############################################################

models = [
    ("LR",make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000, random_state=17))),
    ("KNN", make_pipeline(StandardScaler(),KNeighborsClassifier())),
    ("CART",DecisionTreeClassifier(random_state=17)),
    ("RF",RandomForestClassifier(random_state=17)),
    ("GBM",GradientBoostingClassifier(random_state=17)),
    ("XGBoost",XGBClassifier(random_state=17,eval_metric="logloss")),
    ("LightGBM",LGBMClassifier(random_state=17,verbose=-1)),
    ("CatBoost",CatBoostClassifier(random_state=17,verbose=False,allow_writing_files=False
))]

###############################################################
# 15.5 MODEL COMPARISON WITH CROSS VALIDATION
###############################################################

cv_results_list = []

for name, classifier in models:

    cv_results = cross_validate(
        classifier,
        X_train,
        y_train,
        cv=5,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1
    )

    accuracy = cv_results["test_accuracy"].mean()
    f1 = cv_results["test_f1"].mean()
    roc_auc = cv_results["test_roc_auc"].mean()

    cv_results_list.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

    print(
    f"{name}: "
    f"Accuracy: {accuracy:.4f} | "
    f"F1: {f1:.4f} | "
    f"ROC_AUC: {roc_auc:.4f}")

#LR: Accuracy: 0.7508 | F1: 0.6858 | ROC_AUC: 0.8073
#KNN: Accuracy: 0.7190 | F1: 0.6568 | ROC_AUC: 0.7670
#CART: Accuracy: 0.6509 | F1: 0.5956 | ROC_AUC: 0.6446
#RF: Accuracy: 0.7350 | F1: 0.6705 | ROC_AUC: 0.7970
#GBM: Accuracy: 0.7546 | F1: 0.6955 | ROC_AUC: 0.8210
#XGBoost: Accuracy: 0.7320 | F1: 0.6481 | ROC_AUC: 0.8029
#LightGBM: Accuracy: 0.7561 | F1: 0.6972 | ROC_AUC: 0.8265
#CatBoost: Accuracy: 0.7488 | F1: 0.6811 | ROC_AUC: 0.8227

model_results = pd.DataFrame(cv_results_list)

model_results = model_results.sort_values(by="ROC_AUC",ascending=False)

print(model_results)

###############################################################
# 15.6 BEST BASE MODEL
###############################################################

best_model_name = model_results.iloc[0]["Model"]

print("Best Base Model:", best_model_name)   # LightGBM

model_dict = dict(models)

best_model = model_dict[best_model_name]


###############################################################
# 15.7 TEST PERFORMANCE OF BEST BASE MODEL
###############################################################

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

print("Accuracy:",round(accuracy_score(y_test, y_pred), 4))

print("Precision:",round(precision_score(y_test, y_pred), 4))

print("Recall:",round(recall_score(y_test, y_pred), 4))

print("F1:",round(f1_score(y_test, y_pred), 4))

print("ROC AUC:",round(roc_auc_score(y_test, y_prob), 4))

#Accuracy: 0.7584
#Precision: 0.7333
#Recall: 0.6028
#F1: 0.6617
#ROC AUC: 0.8179

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

###############################################################
# 16. CATBOOST HYPERPARAMETER OPTIMIZATION
###############################################################

catboost_model = CatBoostClassifier(random_state=17,verbose=False,allow_writing_files=False)

catboost_params = {
    "iterations": [300, 500],
    "learning_rate": [0.03, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [3, 5]
}

catboost_best_grid = GridSearchCV(
    catboost_model,
    catboost_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
).fit(X_train, y_train)

print("Best Parameters:")
print(catboost_best_grid.best_params_)

print("Best CV ROC-AUC:")
print(catboost_best_grid.best_score_)      #Best CV ROC-AUC: 0.8344964751838928

# En iyi parametrelerle modelin tekrar kurulması
catboost_final = CatBoostClassifier(
    **catboost_best_grid.best_params_,
    random_state=17,
    verbose=False,
    allow_writing_files=False
)

catboost_final.fit(X_train, y_train)

y_pred = catboost_final.predict(X_test)
y_prob = catboost_final.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))   #ROC AUC: 0.8397

#CatBoost modeli için GridSearchCV ile hiperparametre optimizasyonu gerçekleştirilmiştir.
# Optimize modelde Accuracy, Recall ve F1 skorlarında çok sınırlı artış gözlenirken ROC-AUC değeri 0.8412'den 0.8397'ye düşmüştür.
# Bu nedenle optimizasyonun anlamlı bir performans artışı sağlamadığı değerlendirilmiş ve daha yüksek ROC-AUC
# değerine sahip base CatBoost modelinin final model olarak kullanılmasına karar verilmiştir

###############################################################
# 17. FINAL MODEL SELECTION
###############################################################

final_model = CatBoostClassifier(
    random_state=17,
    verbose=False,
    allow_writing_files=False
)
final_model.fit(X_train, y_train)

###############################################################
# 18. FEATURE IMPORTANCE
###############################################################

def plot_importance(model, features, num=20):

    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    })

    feature_imp = feature_imp.sort_values(
        by="Value",
        ascending=False
    )

    print(feature_imp.head(num))

    plt.figure(figsize=(10, 8))

    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.head(num)
    )

    plt.title("CatBoost Feature Importance")
    plt.tight_layout()
    plt.show()


plot_importance(final_model, X_train)

###############################################################
# 19. FINAL CATBOOST MODEL
###############################################################

final_model = CatBoostClassifier(
    random_state=17,
    verbose=False,
    allow_writing_files=False
)

final_model.fit(X, y)

print("Final CatBoost model trained.")

###############################################################
# 20. SAVE FINAL MODEL
###############################################################

joblib.dump(final_model,"catboost_late_payment_model.pkl")

joblib.dump(X.columns.tolist(),"model_columns.pkl")

###############################################################
# 21. SAVE PREPROCESSING RULES  (Eğitimde öğrendiğimiz preprocessing kurallarını kaydetme adımı)
###############################################################
#Eğitim sırasında kullanılan preprocessing kurallarını kaydederek, yeni verilerin de modelin eğitimindekiyle
# aynı şekilde hazırlanmasını sağlıyoruz.

preprocessing_bundle = {
    "outlier_steps": outlier_steps,
    "rare_rules": rare_rules,
    "known_categories": known_categories,
    "ohe_cols": ohe_cols,
    "global_late_rate": global_late_rate,
    "model_columns": X.columns.tolist()
}

joblib.dump(preprocessing_bundle,"preprocessing_bundle.pkl")

print("Preprocessing bundle saved.")

###############################################################
# 22. PREPARE OPEN INVOICES FOR PREDICTION (Yeni open_df'yi modelin anlayacağı forma dönüştüren fonksiyonu oluşturma adımı)
###############################################################
#ham/temizlenmiş open_df'yi alıp CatBoost'un beklediği 26 feature'a dönüştürüyor.


def prepare_open_data(open_df, closed_df, preprocessing_bundle):

    df = open_df.copy()
    history = closed_df.copy()

    # OPEN FATURADAN OLUŞTURULABİLEN FEATURE'LAR

    df["payment_term_days"] = (df["due_date"] - df["baseline_create_date"]).dt.days
    df["posting_month"] = df["posting_date"].dt.month
    df["due_month"] = df["due_date"].dt.month

    # MÜŞTERİ GEÇMİŞİ

    history = history.sort_values(["customer_id", "posting_date"])

    history["delay_days_raw"] = (history["clear_date"] - history["due_date"]).dt.days.clip(lower=0)

    customer_history = (history.groupby("customer_id").agg(
            cust_past_invoice_count=("LATE_PAYMENT", "count"),
            cust_late_payment_rate_hist=("LATE_PAYMENT", "mean"),
            cust_avg_delay_days_hist=("delay_days_raw", "mean"),
            last_closed_invoice_date=("posting_date", "max")
        ).reset_index())


    last_3 = (history.groupby("customer_id", group_keys=False).tail(3).groupby("customer_id")["LATE_PAYMENT"].mean()
        .rename("cust_last_n_late_ratio")
        .reset_index())

    customer_history = customer_history.merge(last_3,on="customer_id",how="left")

    df = df.merge(customer_history,on="customer_id",how="left")

    df["days_since_last_invoice"] = (df["posting_date"]- df["last_closed_invoice_date"]).dt.days

    # COLD START (yeni kayıtlı müşteriler)

    global_rate = preprocessing_bundle["global_late_rate"]

    df["cust_past_invoice_count"] = (df["cust_past_invoice_count"].fillna(0))

    df["cust_late_payment_rate_hist"] = (df["cust_late_payment_rate_hist"].fillna(global_rate))

    df["cust_last_n_late_ratio"] = (df["cust_last_n_late_ratio"].fillna(global_rate))

    df["cust_avg_delay_days_hist"] = (df["cust_avg_delay_days_hist"].fillna(0))

    df["days_since_last_invoice"] = (df["days_since_last_invoice"].fillna(0))

    df.drop(columns="last_closed_invoice_date",inplace=True)

    # PRIORITY_SCORE için gerçek değerleri koruyoruz
    result_df = df.copy()
    model_df = df.copy()

    # KAYITLI OUTLIER KURALLARINI UYGULAMA

    for step in preprocessing_bundle["outlier_steps"]:
        col = step["column"]
        if col in model_df.columns:
            model_df[col] = model_df[col].clip(
                lower=step["low_limit"],
                upper=step["up_limit"]
            )

    # KAYITLI RARE KURALLARINI UYGULAMA

    for col, labels in preprocessing_bundle["rare_rules"].items():
        if col in model_df.columns:
            model_df[col] = np.where(
                model_df[col].isin(labels),
                "Rare",
                model_df[col]
            )


    # Eğitim sırasında hiç görülmemiş kategoriler
    for col in preprocessing_bundle["ohe_cols"]:
        if col not in model_df.columns:
            continue
        known = preprocessing_bundle["known_categories"][col]
        unknown = ~model_df[col].isin(known)
        if "Rare" in known:
            model_df.loc[unknown, col] = "Rare"

    # ONE HOT ENCODING

    model_df = pd.get_dummies(
        model_df,
        columns=[col for col in preprocessing_bundle["ohe_cols"]if col in model_df.columns],drop_first=False)

    # MODELİN EĞİTİLDİĞİ 26 FEATURE

    model_df = model_df.reindex(columns=preprocessing_bundle["model_columns"],fill_value=0)
    bool_cols = model_df.select_dtypes(include="bool").columns
    model_df[bool_cols] = (model_df[bool_cols].astype(int))

    return result_df, model_df

###############################################################
# 23. OPEN INVOICE PREDICTION (Kaydedilmiş modeli yükleyip açık faturaların geç ödeme olasılığını hesaplıyoruz.)
###############################################################

final_model = joblib.load("catboost_late_payment_model.pkl")

preprocessing_bundle = joblib.load("preprocessing_bundle.pkl")

open_pred_df, open_model_df = prepare_open_data(open_df,closed_df,preprocessing_bundle)

print("Open model shape:", open_model_df.shape)
print("Missing:", open_model_df.isnull().sum().sum())

print("Columns correct:",list(open_model_df.columns)== preprocessing_bundle["model_columns"])  #Tahmin datasındaki kolonlar ile modelin eğitim kolonları birebir aynı mı (bool)

open_pred_df["LATE_PAYMENT_PROBABILITY"] = (final_model.predict_proba(open_model_df)[:, 1]) #faturanın geç ödeme olasılığını verir

#Hazırlanan açık faturalar için predict_proba() kullanarak geç ödeme olasılıkları hesaplanır

###############################################################
# 24. PRIORITY SCORE
###############################################################
#Geç ödeme olasılığını faturanın parasal büyüklüğüyle birleştirerek finansal öncelik skoru oluşturulur

open_pred_df["PRIORITY_SCORE"] = (open_pred_df["LATE_PAYMENT_PROBABILITY"] * open_pred_df["total_open_amount"])

###############################################################
# 25. PRIORITY RANKING
###############################################################

open_pred_df = open_pred_df.sort_values(by="PRIORITY_SCORE",ascending=False).reset_index(drop=True)

open_pred_df["PRIORITY_RANK"] = (np.arange(1, len(open_pred_df) + 1)) #faturayı önelik sırasına göre sıralama

open_pred_df.head(5)

###############################################################
# 26. PRIORITY LEVEL
###############################################################
#Tahsilat ekibinin binlerce faturayı tek tek incelemek yerine faturaları kolayca öncelik gruplarına ayırabilmesini sağlanır.

high_threshold = (open_pred_df["PRIORITY_SCORE"].quantile(0.80))

medium_threshold = (open_pred_df["PRIORITY_SCORE"].quantile(0.50))


def assign_priority(score):
    if score >= high_threshold:
        return "HIGH"
    elif score >= medium_threshold:
        return "MEDIUM"
    else:
        return "LOW"


open_pred_df["PRIORITY_LEVEL"] = (open_pred_df["PRIORITY_SCORE"].apply(assign_priority))

#kontrol
print(open_pred_df["PRIORITY_LEVEL"].value_counts())     #LOW :4840  ,MEDIUM :2904,  HIGH: 1937

print(open_pred_df["PRIORITY_LEVEL"].value_counts(normalize=True) * 100)

###############################################################
# 27. FINAL RESULTS
###############################################################

results_df = open_pred_df[
    [
        "invoice_id",
        "customer_id",
        "customer_name",
        "posting_date",
        "due_date",
        "total_open_amount",
        "LATE_PAYMENT_PROBABILITY",
        "PRIORITY_SCORE",
        "PRIORITY_RANK",
        "PRIORITY_LEVEL"
    ]
].copy()

#Not: tahmin müşteri bazında değil, fatura bazında yapılıyor.
# Aynı müşterinin 3 açık faturası varsa CSV’de o müşteri 3 ayrı satırda görülür.
###############################################################
# 28. SAVE RESULTS
###############################################################

results_df.to_csv("invoice_collection_priorities.csv",index=False)
