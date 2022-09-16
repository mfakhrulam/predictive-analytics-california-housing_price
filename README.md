# Laporan Proyek Machine Learning Pertama - Muhammad Fakhrul Amin

## Domain Proyek

_Machine learning_ merupakan salah satu cabang ilmu komputer yang menggunakan algoritma untuk mengidentifikasi pola dalam suatu data dan menyelesaikan permasalahan menggunakan pola tersebut tanpa perlu secara langsung diprogram olah manusia. Proyek ini memanfaatkan _machine learning_ untuk menyelesaikan permasalahan dalam dunia bisnis/ekonomi. Bidang bisnis yang dimaksud adalah untuk memprediksi nilai atau harga suatu rumah dalam kompleks tertentu. 

Bayangkan apabila Anda seorang pengusaha _real estate_ yang ingin mengembangkan bisnisnya ke daerah yang lebih luas. Misalkan Anda ingin membeli rumah lalu menjualnya kembali kepada orang lain. Agar proses pemberian harga lebih efisien, Anda dapat menerapkan automasi dalam sistem untuk memprediksi harga atau nilai suatu rumah dengan teknik _predictive modelling_ menggunakan algoritma regresi. Prediksi yang dilakukan oleh model _machine learning_ tentu saja dipengaruhi oleh berbagai faktor, seperti total ruangan, total kamar tidur, jarak rumah dari laut, umur rumah, pendapatan pemilik rumah, dan lain sebagainya. Adapun referensi yang saya gunakan yaitu artikel yang ada pada halaman 701 dalam jurnal internasional berikut.  [[1]](#1)  



## Business Understanding

Berdasarkan kondisi yang telah diuraikan sebelumnya, akan dikembangkan suatu sistem prediksi nilai atau harga rumah. Sebelum itu, berikut adalah pernyataan masalah dan tujuan atau _goals_ yang ingin diraih.

### Problem Statements

- Fitur atau faktor apa yang paling berpengaruh terhadap nilai atau harga suatu rumah?
- Berapa harga rumah dengan fitur tertentu?

### Goals

- Mengetahui fitur atau faktor yang paling berpengaruh terhadap nilai atau harga suatu rumah.
- Membuat model _machine learning_ yang dapat memprediksi harga rumah berdasarkan fitur-fitur yang ada.

### Solution statements

- Menggunakan _pairplot_ dan matriks korelasi (_corr_) yang disediakan oleh _library_ _seaborn_ dan _pandas_ untuk menampilkan korelasi fitur-fitur yang ada.
- Menggunakan tiga algoritma regresi untuk mencari algoritma mana yang lebih sesuai untuk digunakan sebagai model akhir.
- Menggunakan _gridsearch_ dari _scikit-learn_ untuk mencari hyperparameter yang paling sesuai untuk digunakan oleh model _machine learning_.
- Menggunakan metrik MAE untuk mengukur tingkat error masing-masing model.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data sensus di California pada tahun 1990. Data sensus ini berkaitan dengan rumah-rumah yang ada di suatu distrik di California dan beberapa ringkasan statistiknya. Dataset ini memiliki 10 kolom dan 20640 baris. Tidak semua datanya valid, terdapat _missing value_ dalam salah satu kolomnya (pada kolom `total_bedrooms`), sehingga diperlukan proses lanjut untuk menanganinya. Adapun sumber data ini diambil dari _kaggle_, sumber data dapat diakses melalui tautan berikut: [_California Housing Prices_](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

### Variabel-variabel pada _California Housing Prices_ dataset adalah sebagai berikut:
| Nama variabel | Keterangan |
| --- | --- |
| `longitude` | Garis bujur bumi, sebagai penunjuk lokasi timur dan barat, semakin tinggi nilainya semakin ke barat |
| `latitude` | Garis lintang bumi, sebagai penunjuk lokasi utara dan selatan, semakin tinggi nilainya semakin ke utara |
| `housing_median_age` | Nilai median umur rumah dalam suatu blok. Semakin kecil nilainya maka semakin baru suatu rumah |
| `total_rooms` | Total ruangan dalam suatu blok |
| `total_bedrooms` | Total kamar tidur dalam suatu blok |
| `population` | Jumlah orang yang tinggal dalam suatu blok |
| `households` | Total rumah tangga yang tinggal dalam suatu blok |
| `median_income` | Nilai median pendapatan rumah tangga dalam suatu blok perumahan |
| `median_house_value ` | Nilai median harga rumah dalam suatu blok perumahan **(variabel target)** |
| `ocean_proximity` | Lokasi rumah mengacu pada laut |

### Exploratory Data Analysis (EDA)
Hal yang pertama perlu dilakukan setelah memuat dataset yaitu mengetahui bentuk, info, dan deskripsi dari data. Ini bisa dilakukan dengan perintah berikut secara berurutan dalam _cell_ yang berbeda:
```py
housing.shape
housing.info()
housing.describe()
```
Kemudian cari jumlah _missing value_ yang ada dalam data.  
```py
housing.isnull().sum()
```
```py
sns.heatmap(housing.isnull(), cbar = False, cmap='flare')
```
![Heatmap Mising Value](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/01-heatmap-missing-value.png)  
Nilai yang kosong itu dapat diisi menggunakan median dari data terkait.  
```py
housing['total_bedrooms'].fillna(value=housing['total_bedrooms'].median(), inplace=True)
```
Selanjutnya yaitu melihat _outliers_ dari data.  
![Outliers](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/02-outliers.png)  

Tampak bahwa banyak fitur memiliki _outliers_.  
**Kemudian akan ditampilkan analisis univariate.** 
**Categorical**  
![Univariate-Categorical](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/03-univariate-categorical.png)  
**Numerical**  
![Univariate-Numerical](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/04-univariate-numerical.png)  
**Kemudian akan ditampilkan analisis multivariate**.   
**Categorical**  
![Multivariate-Categorical](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/05-multivariate-categorical.png)  
**Numerical**  
![Multivariate-Numerical](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/06-multivariate-numerical.png)  
**Correlation Matrix untuk Fitur Numerical**  
![Correlation Matrix](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/07-correlation-matrix.png)  
**Kesimpulan EDA**   
- Terdapat _outliers_ di banyak fitur yang perlu dihilangkan  
- Terdapat korelasi yang cukup tinggi antara `median_house_value` (target) dengan `median_income`  
- Rumah paling banyak terletak pada `<1H OCEAN`  
- Terdapat korelasi yang tinggi antara `total_room`, `total_beedroom`, `population`, dan `households`.  

## Data Preparation  

Ada beberapa teknik _data preparation_ yang digunakan dalam proyek ini, antara lain:
- Menghilangkan _Outliers_ (_Outliers Removal_) adalah teknik untuk menghilangkan data yang nilainya sangat jauh daripada data lain. _Outliers_ perlu dihilangkan karena akan menyebabkan bias pada model _machine learning_ yang akan dibuat. Fitur yang mengalami _outliers_ yaitu `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, dan `meidan_house_value`.  
- _Encoding_ fitur kategori adalah teknik untuk mengubah fitur _categorical_ ke dalam bentuk vektor biner yang bernilai integer 0 dan 1. Teknik ini dilakukan karena banyak algoritma _machine learning_ yang tidak dapat mengolah data berbentuk _categorical_. Fitur yang perlu diterapkan encoding yaitu `ocean_proximity' karena merupakan fitur kategori.  
- _Reduction dimensions_ adalah teknik untuk mengurangi dimensi fitur dengan menggabungkan dua fitur atau lebih yang memiliki tingkat korelasi tinggi dan mengandung informasi yang sama. Teknik ini perlu untuk dilakukan agar model tidak menjadi lebih kompleks karena fitur yang banyak dapat mengakibatkan _data point_ tidak representatif, sehingga mengurangi performa model. Fitur yang diterapkan teknik ini yaitu fitur `longitude` dan `latitude` karena kedua fitur ini memiliki korelasi tinggi dan memiliki informasi yang sama sebagai letak rumah. Kedua fitur itu disatukan menjadi fitur `coordinate`.
- _Train test split_ adalah proses untuk membagi data menjadi dua bagian, yaitu data _train_ dan data _test_. Ini diperlukan agar kita dapat mengetes model apakah sudah layak untuk memprediksi data yang belum pernah ditemui sebelumnya, sebelum masuk ke tahap _deployment_. Dalam proyek ini, data yang telah melewati proses sebelumnya dibagi menjadi dua dengan proporsi 80% data _train_ dan 20% data _test_.  
- Standarisasi adalah proses untuk menormalisasi data agar lebih mudah diterima oleh model _machine learning_. Ini juga diperlukan karena data _numerical_ yang ada di dataset memiliki nilai maksimum dan minimum yang berbeda beda. Hal ini dapat menjadi bias dalam model karena model menganggap data yang memiliki nilai yang besar merupakan fitur yang terpenting, padahal belum tentu seperti itu. Karena alasan itulah, standarisasi perlu dilakukan. Standarisasi dilakukan menggunakan `StandarScaler` dari _library scikit-learn_ karena data terdistribusi normal jika dilihat dari grafiknya. Fitur yang dikenai standarisasi yaitu fitur numerik antara lain `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, dan `coordinate`.

## Modeling

Ada tiga model _machine learning_ yang digunakan dalam proyek ini, yaitu model _KNN Regressor_, _Random Forest Regressor_, dan _Ada Boosting Regressor_. Setiap model _machine learning_ akan melalui proses _improvement_ dengan _hyperparameter tuning_ menggunakan metode _Grid Search_. Sebelum membuat model _machine learning_, siapkan dulu _dataframe_ untuk menyimpan hasil _training_ ketiga model.  
```py
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mae', 'test_mae'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])
```
Hal ini juga akan digunakan pada tahap selanjutnya (_evaluation_). Kemudian siapkan tiga model yang akan digunakan.
- _KNN Regressor_, parameter yang digunakan yaitu `n_neighbor`. Sebelum model dibuat, parameter dalam model ini dimasukkan ke dalam Grid Search terlebih dahulu untuk menemukan parameter terbaik.
    ```py
  # Gunakan gridsearch untuk mencari hasil terbaik
  neighbors = list(range(1,15))
  parameters = {
      'n_neighbors':neighbors
  }
  knn = KNeighborsRegressor()
  grid_search = GridSearchCV(knn, parameters, verbose=2)
  grid_search.fit(X_train, y_train)
  print("KNN GridSearch score: "+str(grid_search.best_score_))
  print("KNN GridSearch params: "+str(grid_search.best_params_))
  # Output
  # KNN GridSearch score: 0.6715451895544013
  # KNN GridSearch params: {'n_neighbors': 11}
  ```  
  Setelah didapat `n_neighbor` terbaik yaitu 11, buat modelnya menggunakan perintah berikut.
  ```py
  # buat model prediksi
  knn = KNeighborsRegressor(n_neighbors=11)
  knn.fit(X_train, y_train)
  ```
  Model ini memiliki kelebihan yaitu tidak ada waktu tunggu untuk proses pelatihan dan juga mudah untuk diimplementasikan. Kelemahannya yaitu model ini tidak cocok digunakan untuk data yang besar dan juga sensitif pada noise dan missing value.
- _Random Forest Regressor_, parameter yang digunakan yaitu n_estimator, max_depth, random_state, dan n_jobs. Sama seperti model KNN, parameter dalam model ini juga dimasukkan ke dalam _Grid Search_ terlebih dahulu. Bedanya ada pada parameter dan model yang digunakan.
  ``` py
  n_estimators = list(range(40,101,20))
  max_depth = list(range(10,20,2))
  parameters = {
      'n_estimators':n_estimators,
      'max_depth':max_depth
  }
  RF = RandomForestRegressor()
  ```
  Setelah melalui _Grid Search_, didapat parameter terbaik yaitu.  
  ```
  RF GridSearch score: 0.7324702141251758
  RF GridSearch params: {'max_depth': 18, 'n_estimators': 100}
  ```
  Kemudian buat modelnya berdasarkan parameter tersebut (`max_depth=18` dan `n_estimators=100`) dan tambahkan `random_state=42` untuk mengacak sampel yang digunakan serta `n_jobs=-1` untuk menggunakan semua prosesor secara paralel.
  ```py
  # buat model prediksi
  RF = RandomForestRegressor(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1)
  RF.fit(X_train, y_train)
  ```
  Model ini memiliki kelebihan yaitu dapat bekerja baik pada fitur _categorical_ dan _numerical_, stabil, tidak sensitif terhadap _noise_. Kelemahannya yaitu memiliki algoritma yang kompleks, sehingga memiliki waktu _training_ yang lebih lama.
- _Ada Boosting Regressor_, parameter yang digunakan yaitu learning_rate dan random_state. Proses pencarian dengan _Grid Search_ masih sama seperti sebelumnya, tetapi ada yang berbeda pada model dan parameternya. Berikut potongan kode yang berbeda.
  ``` py
  parameters = {
      'learning_rate':[0.1,0.3,0.5,0.05,0.005]
  }
  boosting = AdaBoostRegressor()
  ```
  Setelah melalui _Grid Search_, didapat parameter terbaik yaitu.  
  ```
  Boosting GridSearch score: 0.5273297505316186
  Boosting GridSearch params: {'learning_rate': 0.05}
  ```
  Kemudian buat model berdasarkan parameter terbaik tersebut (`learning_rate=0.05` dan tambahkan `random_state=42` untuk mengacak sampel yang digunakan.
  ```py
  # buat model prediksi
  boosting = AdaBoostRegressor(learning_rate=0.05, random_state=42)                             
  boosting.fit(X_train, y_train)
  ```
  Model ini memeliki kelebihan yaitu kurang rentan terhadap _overfitting_. Kelemahannya yaitu model ini membutuhkan dataset yang berkualitas karena senstif terhadap _noise_ dan _outliers_.

Kesimpulan model:  
Dari ketiga model di atas, menurut saya, _random forest_ lah yang merupakan solusi terbaik untuk dataset proyek ini. _Random forest_ lebih stabil dan bekerja baik pada fitur _categorical_ dan _numerical_, di mana kedua fitur ini memiliki pengaruh terhadap prediksi yang dilakukan. Selain itu, model ini juga lebih aman jika bertemu _noise_. Meski algoritma ini perlu waktu yang lebih lama untuk proses _training_ dibanding kedua algoritma lainnya, tapi waktu itu tidak terlalu lama menurut hitungan manusia

## Evaluation

Dalam kasus regresi, ada beberapa metrik evaluasi yang dapat digunakan seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE). Metrik evaluasi yang digunakan yaitu Mean Absolute Error (MAE). MAE adalah rata-rata selisih mutlak nilai prediksi (predict value) dengan nilai sebenarnya (true value), artinya setiap selisih dari nilai prediksi dengan nilai sebenarnya akan dijumlahkan kemudian hasil itu akan dibagi dengan banyaknya data. Formula metrik MAE dapat dituliskan sebagai berikut:   
![Formula MAE](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/08-MAE.png)   
Metrik MAE digunakan dalam proyek ini karena hasil prediksi yang diinginkan berupa harga rumah atau nilai selisih absolut sebenarnya, bukan selisih yang dikuadratkan. Selisih yang dikuadratkan dalam konteks ini tidaklah cocok. Karena itu dipilihlah metrik MAE untuk mengukur tingkat _error_ model.  

Kesimpulan hasil proyek:
- Model _machine learning_ KNN dan RF sudah memenuhi batas 10% MAE yang telah disesuakan dengan skala data, artinya model sudah cukup bagus. Sedangkan model _Boosting_ masih berada di atas batas tersebut. Batas MAE itu didapat dari:
  ```py
  mae_scale = abs(housing['median_house_value'].max()-housing['median_house_value'].min())*0.1
  mae_scale
  # 46720.100000000006 (output)
  ```
  ```
	              mae_train       mae_test
  KNN	        34077.380931	38032.507924
  RF	        13907.016915	33080.75042
  Boosting	48981.361864	50111.97536
  ```
- Ketiga model masih _overfit_, terutama model _Random Forest_.
- Model _Random Forest_ memiliki hasil _error_ MAE paling kecil yang artinya model ini merupakan model yang terbaik dibanding dua model lain.  
  ![Metrik evaluasi](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/09-metrik-evaluasi.png)  
  Selain itu, model ini juga yang paling presisi untuk memprediksi harga. Berikut grafik perbedaan prediksi ketiga model.  
  ![Perbandingan prediksi](https://raw.githubusercontent.com/mfakhrulam/predictive-analytics-california-housing_price/main/images/10-prediksi.png)  
  

## Referensi
<a id="1">[1]</a> 
[M. Kovářík, and R. Benda, “Applied machine learning predictive modelling in regional spatial data analysis problem,” Financ. Perform. Firms Sci. Educ. Pract. 2015, pp. 701–715, 2015.](https://web.archive.org/web/20180722041033/http://www.ufu.utb.cz/konference/sbornik2015.pdf)  


