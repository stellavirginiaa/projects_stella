import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Fungsi-fungsi
def sigmoid_func(x):
    hit = 1 / (1 + np.exp(-x))
    return hit

def calculate_www(url):
    hostlen = urlparse(url).hostname
    host_www = 1 if hostlen and 'www' in hostlen.lower() else 0
    return host_www

def calculate_com(url):
    hostlen = urlparse(url).hostname
    host_com = 1 if hostlen and '.com' in hostlen.lower() else 0
    return host_com

def calculate_dot(url):
    hostlen = urlparse(url).hostname
    if hostlen:
        host_dot = hostlen.lower().count('.')
        return host_dot
    else:
        return 0

def calculate_slash(url):
    path = urlparse(url).path
    slash_count = path.count('/')
    return slash_count

def count_digits(url):
    digit_count = sum(char.isdigit() for char in url)
    return digit_count

def calculate_hostname_length(url):
    parsed_url = urlparse(url)
    return len(parsed_url.netloc)

def calculate_ratio_digits(string):
    if len(string) == 0:
        return 0
    digit_count = sum(char.isdigit() for char in string)
    return digit_count / len(string)

def data_cleaning(df):
    duplicate_rows = df.duplicated()
    duplicate_data = df[duplicate_rows]
    # print("Duplicate Rows:")
    # print(duplicate_data)
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def clf_rf_class(dataX, dataY, tsize, rstate):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    best_score = 0
    for n_esti in [10, 20, 30, 40, 50, 100]:
        clfRFR = RandomForestClassifier(n_estimators=n_esti, random_state=rstate)
        clfRFR.fit(X_train, y_train)
        score = clfRFR.score(X_test, y_test)
        if score > best_score:
            best_score = score
            total_tree = {'n_estimators': n_esti}
    return (best_score, total_tree, clfRFR)

# Load datasets
floadDt1 = pd.read_excel('https://raw.githubusercontent.com/stellavirginiaa/projects_stella/dataset/master/dataset_pakfelik.xlsx')
floadDt2 = pd.read_excel('https://raw.githubusercontent.com/stellavirginiaa/projects_stella/dataset/master/dataset_tambahan.xlsx')


status_map = {'phishing': 1, 'legitimate': 0}
floadDt1['status'] = floadDt1['status'].map(status_map)
floadDt2['status'] = floadDt2['status'].map(status_map)

floadDt = pd.concat([floadDt1, floadDt2], ignore_index=True)
# print(floadDt)

# Feature engineering
floadDt['length_url'] = floadDt['url'].apply(lambda url: len(str(url)))
floadDt['nb_www'] = floadDt['url'].apply(calculate_www)
floadDt['nb_com'] = floadDt['url'].apply(calculate_com)
floadDt['length_dot'] = floadDt['url'].apply(calculate_dot)
floadDt['length_slash'] = floadDt['url'].apply(calculate_slash)
floadDt['length_digits'] = floadDt['url'].apply(count_digits)
floadDt['length_hostname'] = floadDt['url'].apply(calculate_hostname_length)
floadDt['ratio_digits_url'] = floadDt['url'].apply(calculate_ratio_digits)
floadDt['status'] = floadDt['status']
floadDt['Hostname'] = floadDt['url'].apply(lambda url: urlparse(url).netloc)
floadDt['ratio_digits_host'] = floadDt['Hostname'].apply(calculate_ratio_digits)
# print(floadDt)

# Data cleaning
floadDt = data_cleaning(floadDt)
# print("Data Setelah Data Cleaning:")
# print(floadDt)

# Check null
floadDt = floadDt.dropna()
# print("Data Setelah Menghilangkan Nilai Null:")
# print(floadDt)

null_status = floadDt['status'].isnull().sum()
# print("Jumlah nilai null di kolom status setelah pembersihan data:", null_status)

# Data engineering
floadDt['status'] = floadDt['status'].astype(int)
# print(floadDt)

# EDA
# plt.figure(figsize=(10, 8))
# numeric_cols = floadDt.select_dtypes(include=['float64', 'int64'])

# # Bangun heatmap korelasi dari kolom-kolom numerik
# sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# sns.heatmap(floadDt.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Korelasi Antar Fitur')
# plt.show()

# Modelling
dt_x = floadDt[['nb_www', 'nb_com', 'length_dot', 'length_slash', 'length_digits', 'length_url', 'length_hostname', 'ratio_digits_url', 'ratio_digits_host']]
dt_y = floadDt['status']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(dt_x, dt_y, test_size=0.2, random_state=42)

# Buat model Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

total_hit_model = 0
for i in range(5):
    hit_model = clf_rf_class(dt_x, dt_y, 0.1, i)
    total_hit_model += hit_model[0]
    print(hit_model)

rat_model = total_hit_model / 5
# print('Nilai Rataan Model:', rat_model)

# Simpan model
with open('m1_rf_mean_acc87p.pkl', 'wb') as model:
    pickle.dump(hit_model, model)

# Muat model
with open('m1_rf_mean_acc87p.pkl', 'rb') as model:
    load_model = pickle.load(model)

# Simpan model dengan joblib
import joblib
joblib.dump(load_model, 'm1_rf_mean_acc87p.joblib')

# Ekstrak model dari tuple
clfRFR = load_model[2]

# Latih model pada seluruh data
clfRFR.fit(dt_x, dt_y)

# Visualisasi fitur penting
feature_importances = clfRFR.feature_importances_
feature_names = ["nb_www", "nb_com", 'length_dot', 'length_slash' ,'length_digits',"length_url", "length_hostname", "ratio_digits_url", "ratio_digits_host"]

plt.barh(range(len(feature_importances)), feature_importances, align="center")
plt.yticks(range(len(feature_importances)), feature_names)
plt.xlabel("Feature Importance")
plt.title("Random Forest Classification - Feature Importances")

# Prediksi
y_pred = clfRFR.predict(dt_x)
accuracy = accuracy_score(dt_y, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# print("Classification Report:")
# print(classification_report(dt_y, y_pred))

# print("Confusion Matrix:")
# print(confusion_matrix(dt_y, y_pred))

# Evaluasi
scores = cross_val_score(clfRFR, dt_x, dt_y, cv=5)
# print("Cross-Validation Scores:", scores)
# print("Mean CV Score:", scores.mean())

# Hyperparameter Tuning
train_scores = []
test_scores = []
trees = []

for n_tree in range(10, 110, 10):    
    clfRFR.fit(X_train, y_train)

    # Evaluasi pada data pelatihan
    train_score = clfRFR.score(X_train, y_train)
    train_scores.append(train_score)

    # Evaluasi pada data pengujian
    test_score = clfRFR.score(X_test, y_test)
    test_scores.append(test_score)

    trees.append(n_tree)

plt.plot(trees, train_scores, marker='o', label='Train')
plt.plot(trees, test_scores, marker='o', label='Test')


# Implementasi
# Pisahkan fitur dan target
X = floadDt[['nb_www', 'nb_com', 'length_dot', 'length_slash', 'length_digits', 'length_url', 'length_hostname', 'ratio_digits_url', 'ratio_digits_host']]
y = floadDt['status']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model pada set pelatihan
rf_classifier.fit(X_train, y_train)

# Lakukan prediksi pada set pengujian
y_pred = rf_classifier.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Tampilkan confusion matrix dan classification report
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Contoh prediksi
# url1 = "https://www.bankmandiri.co.id"
# url2 = "https://iplogger.com/teslink"
# f1 = [calculate_www(url1), calculate_www(url2)]
# f2 = [calculate_com(url1), calculate_com(url2)]
# f3 = [calculate_dot(url1), calculate_dot(url2)]
# f4 = [calculate_slash(url1), calculate_slash(url2)]
# f5 = [count_digits(url1), count_digits(url2)]
# f6 = [len(url1), len(url2)]
# f7 = [calculate_hostname_length(url1), calculate_hostname_length(url2)]
# f8 = [calculate_ratio_digits(url1), calculate_ratio_digits(url2)]
# f9 = [calculate_ratio_digits(urlparse(url1).netloc), calculate_ratio_digits(urlparse(url2).netloc)]

# new_data = pd.DataFrame({'nb_www': f1, 'nb_com': f2, 'length_dot': f3, 'length_slash': f4, 'length_digits': f5, 'length_url': f6, 'length_hostname': f7, 'ratio_digits_url': f8,
#                          'ratio_digits_host': f9})

# # Prediksi menggunakan model yang disimpan
# predictions_rfr = clfRFR.predict(new_data)

