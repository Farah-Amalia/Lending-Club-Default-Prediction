# Lending Club Default Prediction


## Domain Proyek

LendingClub adalah platform _peer-to-peer lending_ terbesar di dunia yang berlokasi di San Francisco, California, Amerika Serikat.

Terdapat dua pihak yang berperan dalam P2P Lending, yakni Peminjam dan Pendana/Investor. Cara kerja P2P Lending secara singkat adalah sebagai berikut.
1. Peminjam mengajukan permohonon pinjaman dengan mengunggah semua dokumen yang dibutuhkan, yang di antaranya merupakan dokumen berisi laporan keuangan dalam jangka waktu tertentu dan juga tujuan peminjam dalam pinjaman tersebut.
2. Sebagai investor, pendana memiliki akses untuk menelusuri data-data pengajuan pinjaman di dashboard yang telah disediakan, terutama data relevan mengenai si peminjam seperti pendapatan, riwayat keuangan, tujuan peminjaman (bisnis, kesehatan, atau pendidikan) beserta alasannya, dan sebagainya. Jika pendana memutuskan untuk menginvestasikan pinjaman tersebut, pendana bisa langsung menginvestasikan sejumlah dana setelah melakukan deposit sesuai tujuan investasi pendana. Peminjam akan mencicil dana pinjamannya setiap bulan dan pendana akan mendapatkan keuntungan berupa pokok dan bunga. Besaran bunga akan tergantung pada suku bunga pinjaman yang diinvestasikan.

Salah satu permasalahan yang sering muncul dalam model bisnis P2P Lending adalah terjadinya gagal bayar oleh peminjam. Oleh karena itu, penyedia platform harus mampu mendeteksi adanya peminjam yang kemungkinan akan gagal bayar, untuk meminimalisir risiko dan kerugian yang mungkin dialami. Dengan adanya data historis peminjam, platform dapat memanfaatkan teknologi machine learning untuk mempelajari pola atau perilaku peminjam yang kemungkinan akan gagal bayar. Terdapat beberapa riset yang dilakukan berdasarkan data pinjaman dari LendingClub, di antaranya adalah riset oleh [Jagtiani, J., & Lemieux, C. (2019)](https://onlinelibrary.wiley.com/doi/abs/10.1111/fima.12295) yang membahas tentang alternatif data oleh LendingClub yang bermanfaat untuk memprediksi gagal bayar.  

## Business Understanding
Berdasarkan uraian singkat tentang bisnis P2P Lending, muncul problem berikut.
### Problem Statement: 
Bagaimana cara mendeteksi peminjam yang gagal bayar?
Dengan adanya problem statement di atas, maka tujuan dilakukannya proyek ini adalah sebagai berikut.
### Goals: 
Membangun model yang dapat memprediksi apakah peminjam mampu membayar kembali pinjamannya atau gagal bayar.
Selanjutnya, solusi yang ditawarkan untuk mencapai tujuan tersebut adalah sebagai berikut.
### Solution: 
1. Logistic Regression, yaitu model logit untuk mengklasifikasikan dua kategori
2. Tensorflow / Deep Learning, yaitu model neural network yang akan didesain untuk dapat mengklasifikasikan dua kategori (menggunakan fungsi aktivasi sigmoid)

## Data Understanding
Data yang digunakan pada proyek ini dapat diakses di Kaggle pada link berikut https://www.kaggle.com/jeandedieunyandwi/lending-club-dataset

Data yang digunakan terdiri dari satu target variabel (loan_status) yaitu skenario atau status pada loan tersebut:
1. Fully paid: Peminjam telah melunasi pinjaman (pokok dan bunga)
2. Charged-off: Peminjam tidak membayar angsuran pada waktunya untuk jangka waktu yang tertentu, atau dengan kata lain gagal bayar

Untuk memahami target variabel, dilakukan visualisasi untuk membandingkan jumlah pengamatan pada kedua kategori tersebut.
![Loan Status](https://github.com/Farah-Amalia/Lending-Club-Default-Prediction/blob/main/download%20(1).png)

Visualisasi di atas menggambarkan bahwa terdapat sekitar 20% dari keseluruhan pinjaman yang gagal bayar.

Sedangkan untuk feature yang ada pada dataset dijelaskan pada tabel berikut.
| No. | Feature | Deskripsi |
| ------ | ------ | ------ |
| 1. | loan_amnt | Jumlah pinjaman yang terdaftar yang diajukan oleh peminjam. |
| 2. | term | Jumlah pembayaran pinjaman, 36 bulan atau 60 bulan. |
| 3. | int_rate | Interest rate atau bunga. |
| 4. | installment | Pembayaran bulanan terutang. |
| 5. | grade | Grade yang ditentukan oleh LC. |
| 6. | sub_grade | Sub-grade yang ditentukan oleh LC |
| 7. | emp_title | Jabatan yang peminjam saat mengajukan pinjaman |
| 8. | emp_lenth | Lama kerja dalam satuan tahun. |
| 9. | home_ownership | Status kepemilikan rumah yang diberikan oleh peminjam pada saat pendaftaran atau diperoleh dari laporan kredit. |
| 10. | annual_inc | Pendapatan tahunan yang dilaporkan oleh peminjam selama pendaftaran. |
| 11. | verification_status | Menunjukkan jika pendapatan diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi. |
| 12. | issue_date | Bulan di mana pinjaman didanai. |
| 13. | purpose | Kategori yang diisi oleh peminjam untuk permintaan pinjaman. |
| 14. | title | Judul pinjaman yang diisi oleh peminjam. |
| 15. | address | Alamat yang diberikan oleh peminjam dalam aplikasi pinjaman. |
| 16. | dti | Rasio yang dihitung menggunakan total pembayaran hutang bulanan peminjam atas total kewajiban hutang, dibagi dengan pendapatan bulanan peminjam yang dilaporkan sendiri. |
| 17. | earliest_cr_lina | Bulan di mana batas kredit paling awal yang dilaporkan peminjam dibuka. |
| 18. | open_acc | Jumlah credit line terbuka dalam file kredit peminjam. |
| 19. | pub_rec | Jumlah derogatory public record |
| 20. | revol_bal | Total credit revolving balance |
| 21. | revol_util | Tingkat penggunaan revolving balance, atau jumlah kredit yang digunakan peminjam relatif terhadap semua kredit yang tersedia. |
| 22. | total_acc | Jumlah total credit line saat ini dalam file kredit peminjam. |
| 23. | initial_list_status | Status pencatatan awal pinjaman. Nilai yang mungkin adalah W atau F |
| 24. | application_type | Menunjukkan apakah pinjaman tersebut merupakan aplikasi individu atau aplikasi bersama dengan dua peminjam bersama. |
| 25. | mort_acc | Jumlah rekening hipotek/mortgage. |
| 26. | pub_rec_bankruptcies | Jumlah public record bankruptcies. |

## Data Preparation
Terdapat beberapa metode yang digunakan untuk melakukan pre-processing pada data, di antaranya adalah:
- Imputasi missing value, menggunakan rata-rata berdasarkan variabel lain yang terkait
- Menghapus pengamatan yang memuat outlier
- Melakukan one hot encoding pada kolom kategorikal
- Standarisasi data, hal ini diperlukan untuk memberikan output yang lebih baik terutama pada model tensorflow

## Modeling
Terdapat dua macam algoritma machine learning yang digunakan, yaitu Logistic Regression dan Tensorflow Deep Learning/ Neural Network.
Tahapan yang dilakukan adalah melakukan splitting data menjadi tiga bagian yaitu train, validation, dan test. Tahapan berikutnya adalah fitting pada masing-masing model. Untuk model tensorflow, terlebih dahulu dibuat model architecture-nya.
Hasil yang diperoleh adalah tidak ada perbedaan yang signifikan antara model Logistic Regression dan Tensorflow, sehingga untuk alasan efisiensi maka akan digunakan model Logistic Regression karena memiliki waktu pemrosesan yang lebih cepat.

## Evaluation
Metriks evaluasi yang digunakan adalah AUC dan recall karena data yang digunakan adalah imbalanced, serta accuracy untuk perbandingan. Berikut adalah metriks evaluasi pada data test setelah dilakukan modeling.
1. **AUC Score**: AUC adalah singkatan dari "Area under ROC curve", artinya AUC mengukur seluruh area dua dimensi di bawah seluruh kurva ROC dari (0,0) hingga (1,1)
2. **Recall**: Recall merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. Dalam konteks ini, recall menggambarkan berapa persen pinjaman yang diprediksi gagal bayar dibandingkan keseluruhan pinjaman yang sebenarnya gagal bayar. Formula dari recall adalah **tp / (tp + fn)** di mana tp adalah jumlah true positives dan fn adalah jumlah false negatives.
3. **Accuracy**: Accuracy adalah salah satu metrik untuk mengevaluasi model klasifikasi, dengan formula **jumlah prediksi yang benar / jumlah total prediksi**. Kelemahan metrik ini adalah tidak relevan jika digunakan pada data yang imbalance.

Berikut adalah metrik evaluasi model yang diperoleh.

| Metrics | Logistic regression | Tensorflow |
| ------ | ------ | ------ |
| Accuracy | 0.888751 | 0.888103 |
| Recall | 0.994631 | 0.989376 |
| AUC Score | 0.907278 | 0.906522 |

Berdasarkan tabel di atas, performa logistic regression dan tensorflow tidak berbeda jauh dan sedikit lebih tinggi, sehingga model yang dipilih sebagai solusi untuk penyelesaian masalah prediksi peminjam gagal bayar adalah logistic regression.
