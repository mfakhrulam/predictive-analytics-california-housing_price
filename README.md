# Laporan Proyek Machine Learning Pertama - Muhammad Fakhrul Amin

## Domain Proyek

Machine learning merupakan salah satu cabang ilmu komputer yang menggunakan algoritma untuk mengidentifikasi pola dalam suatu data dan menyelesaikan permasalahan menggunakan pola tersebut tanpa perlu secara langsung diprogram olah manusia. Proyek ini memanfaatkan machine learning untuk menyelesaikan permasalahan dalam dunia bisnis/ekonomi. Bidang bisnis yang dimaksud adalah untuk memprediksi nilai atau harga suatu rumah dalam kompleks tertentu. 

Bayangkan apabila Anda seorang pengusaha real estate yang ingin mengembangkan bisnisnya ke daerah yang lebih luas. Misalkan Anda ingin membeli rumah lalu menjualnya kembali kepada orang lain. Agar proses pemberian harga lebih efisien, Anda dapat menerapkan automasi dalam sistem untuk memprediksi harga atau nilai suatu rumah dengan teknik predictive modelling menggunakan algoritma regresi. Prediksi yang dilakukan oleh model machine learning tentu saja dipengaruhi oleh berbagai faktor, seperti total ruangan, total kamar tidur, jarak rumah dari laut, umur rumah, pendapatan pemilik rumah, dan lain sebagainya. Adapun artikel yang saya jadikan referensi yaitu jurnal internasional [FINANCE AND PERFORMANCE OF FIRMS IN SCIENCE,EDUCATION AND PRACTICE](https://web.archive.org/web/20180722041033/http://www.ufu.utb.cz/konference/sbornik2015.pdf) pada halaman 701.


## Business Understanding

Berdasarkan kondisi yang telah diuraikan sebelumnya, akan dikembangkan suatu sistem prediksi nilai atau harga rumah. Sebelum itu, berikut adalah pernyataan masalah dan tujuan atau goals yang ingin diraih.

### Problem Statements

- Fitur atau faktor apa yang paling berpengaruh terhadap nilai atau harga suatu rumah?
- Berapa harga rumah dengan fitur tertentu?

### Goals

- Mengetahui fitur atau faktor yang paling berpengaruh terhadap nilai atau harga suatu rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah berdasarkan fitur-fitur yang ada.

### Solution statements

- Menggunakan pairplot dan matriks korelasi (corr) yang disediakan oleh library seaborn dan pandas untuk menampilkan korelasi fitur-fitur yang ada.
- Menggunakan tiga algoritma regresi untuk mencari algoritma mana yang lebih sesuai untuk digunakan sebagai model akhir.
- Menggunakan gridsearch dari scikit-learn untuk mencari hyperparameter yang paling sesuai untuk digunakan oleh model machine learning.
- Menggunakan metrik MAE untuk mengukur tingkat error masing-masing model.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data sensus di California pada tahun 1990. Data sensus ini berkaitan dengan rumah-rumah yang ada di suatu distrik di California dan beberapa ringkasan statistiknya. Dataset ini memiliki 10 kolom dan 20640 baris. Tidak semua datanya valid, terdapat *missing value* dalam salah satu kolomnya (pada kolom `total_bedrooms`), sehingga diperlukan proses lanjut untuk menanganinya. Adapun sumber data ini diambil dari kaggle, sumber data dapat diakses melalui tautan berikut: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

### Variabel-variabel pada California Housing Prices dataset adalah sebagai berikut:
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
Hal yang pertama perlu dilakukan setelah memuat dataset yaitu mengetahui bentuk, info, dan deskripsi dari data. Ini bisa dilakukan dengan perintah berikut secara berurutan dalam cell yang berbeda:
```py
housing.shape
housing.info()
housing.describe()
```
Kemudian cari jumlah missing value yang ada dalam data.
```py
housing.isnull().sum()
```
```py
sns.heatmap(housing.isnull(), cbar = False, cmap='flare')
```
![Heatmap Mising Value](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/01-heatmap-missing-value.png)  
Nilai yang kosong itu dapat diisi menggunakan median dari data terkait.
```py
housing['total_bedrooms'].fillna(value=housing['total_bedrooms'].median(), inplace=True)
```
Selanjutnya yaitu melihat outliers dari data.
![Outliers](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/02-outliers.png)  

Tampak bahwa banyak fitur memiliki outliers.  
**Kemudian akan ditampilkan analisis univariate.** 
**Categorical**  
![Univariate-Categorical](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/03-univariate-categorical.png)  
**Numerical**  
![Univariate-Numerical](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/04-univariate-numerical.png)  
**Kemudian akan ditampilkan analisis multivariate**.   
**Categorical**  
![Multivariate-Categorical](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/05-multivariate-categorical.png)  
**Numerical**  
![Multivariate-Numerical](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/06-multivariate-numerical.png)  
**Correlation Matrix untuk Fitur Numerical**  
![Correlation Matrix](https://github.com/mfakhrulam/predictive-analytics-california-housing_price/blob/main/images/07-correlation-matrix.png)  
**Kesimpulan EDA**   
- Terdapat outlier di banyak fitur yang perlu dihilangkan  
- Terdapat korelasi yang cukup tinggi antara median_house_value (target) dengan median_income  
- Rumah paling banyak terletak pada <1H OCEAN  
- Terdapat korelasi yang tinggi antara total_room, total_beedroom, population, dan households.  

## Data Preparation  

Ada beberapa teknik data preparation yang digunakan dalam proyek ini, antara lain:
- Menghilangkan Outliers (Outliers Removal) adalah teknik untuk menghilangkan data yang nilainya sangat jauh daripada data lain. Outliers perlu dihilangkan karena akan menyebabkan bias pada model machine learning yang akan dibuat.
- Encoding fitur kategori adalah teknik untuk mengubah fitur categorical ke dalam bentuk vektor biner yang bernilai integer 0 dan 1. Teknik ini dilakukan karena banyak algoritma machine learning yang tidak dapat mengolah data berbentuk categorical.
- Reduction dimensions adalah teknik untuk mengurangi dimensi fitur dengan menggabungkan dua fitur atau lebih yang memiliki tingkat korelasi tinggi dan mengandung informasi yang sama. Teknik ini perlu untuk dilakukan agar model tidak menjadi lebih kompleks karena fitur yang banyak dapat mengakibatkan data point tidak representatif, sehingga mengurangi performa model.
- Train test split adalah proses untuk membagi data menjadi dua bagian, yaitu data train dan data test. Ini diperlukan agar kita dapat mengetes model apakah sudah layak untuk memprediksi data yang belum pernah ditemui sebelumnya, sebelum masuk ke tahap deployment.
- Standarisasi adalah proses untuk menormalisasi data agar lebih mudah diterima oleh model machine learning. Ini juga diperlukan karena data numerical yang ada di dataset memiliki nilai maksimum dan minimum yang berbeda beda. Hal ini dapat menjadi bias dalam model karena model menganggap data yang memiliki nilai yang besar merupakan fitur yang terpenting, padahal belum tentu seperti itu. Karena alasan itulah, standarisasi perlu dilakukan. 

## Modeling

Ada tiga model machine learning yang digunakan dalam proyek ini, yaitu model KNN regressor, Random Forest regressor, dan Ada Boosting regressor. Setiap model machine learning akan melalui proses improvement dengan hyperparameter tuning menggunakan metode Grid Search. Sebelum membuat model machine learning, siapkan dulu dataframe untuk menyimpan hasil training ketiga model. Hal ini juga akan digunakan pada tahap selanjutnya (evaluation). Kemudian siapkan tiga model yang akan digunakan.
- KNN Regressor, parameter yang digunakan yaitu n_neighbor. Model ini memiliki kelebihan yaitu tidak ada waktu tunggu untuk proses pelatihan dan juga mudah untuk diimplementasikan. Kelemahannya yaitu model ini tidak cocok digunakan untuk data yang besar dan juga sensitif pada noise dan missing value.
- Random Forest Regressor, parameter yang digunakan yaitu n_estimator, max_depth, random_state, dan n_jobs. Model ini memiliki kelebihan yaitu dapat bekerja baik pada fitur categorical dan numerical, stabil, tidak sensitif terhadap noise. Kelemahannya yaitu memiliki algoritma yang kompleks, sehingga memiliki waktu training yang lebih lama.
- Ada Boosting Regressor, parameter yang digunakan yaitu learning_rate dan random_state. Model ini memeliki kelebihan yaitu kurang rentan terhadap overfitting. Kelemahannya yaitu model ini membutuhkan dataset yang berkualitas karena senstif terhadap noise dan outliers.

Dari ketiga model di atas, menurut saya, random forest lah yang merupakan solusi terbaik untuk dataset proyek ini. Random forest lebih stabil dan bekerja baik pada fitur categorical dan numerical, di mana kedua fitur ini memiliki pengaruh terhadap prediksi yang dilakukan. Selain itu, model ini juga lebih aman jika bertemu noise. Meski algoritma ini perlu waktu yang lebih lama untuk proses training dibanding kedua algoritma lainnya, tapi waktu itu tidak terlalu lama menurut hitungan manusia

## Evaluation

Dalam kasus regresi, ada beberapa metrik evaluasi yang dapat digunakan seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE). Metrik evaluasi yang digunakan yaitu Mean Absolute Error (MAE). MAE adalah rata-rata selisih mutlak nilai prediksi (predict value) dengan nilai sebenarnya (true value), artinya setiap selisih dari nilai prediksi dengan nilai sebenarnya akan dijumlahkan kemudian hasil itu akan dibagi dengan banyaknya data. Formula metrik MAE dapat dituliskan sebagai berikut:
$$ MSE = \frac{1}{n} \Sigma_{i=1}^n|{y}-\hat{y}| $$ Metrik MAE digunakan dalam proyek ini karena hasil prediksi yang diinginkan berupa harga rumah atau nilai selisih absolut sebenarnya, bukan selisih yang dikuadratkan. Selisih yang dikuadratkan dalam konteks ini tidaklah cocok. Karena itu dipilihlah metrik MAE untuk mengukur tingkat error model.

Kesimpulan hasil proyek:
- Model machine learning KNN dan RF sudah memenuhi batas 10% MAE yang telah disesuakan dengan skala data, artinya model sudah cukup bagus. Sedangkan model Boosting masih berada di atas batas tersebut. Batas MAE itu didapat dari:
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
- Ketiga model masih overfit, terutama model Random Forest.

