
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##################################################
#Hypothesis Testing (Hiptez Testleri)
#################################################

# Bir inanışı, bir savı test etmek için kullanılan istatistiksel yöntemlerdir.

# Hipotez testleri kapsamında grup karşılaştırmaları olacaktır.

# Grup karşılaştırmalarında temel amaç olası farklılıkların şans eseri ortaya çıkıp çıkmadığını göstermeye çalışmaktır.

# örneğin:
# Mobil uygulamada yapılan arayüz değişikliği sonrasında kullanıcıların uygulamada geçirdikleri günlük ortalama süre
# arttı mı ?
# Buradan elde ettiğimiz sonuç, aldığımız örneklem üzerinden elde ettiğimiz cevap, şans eseri mi ortaya çıktı yoksa
# gerçekten böyle bir durum söz konusu mu ?
# Bu konuda bunu anlamaya çalışacağız. Bunu istatistiki bir şekilde hesaplayıp anlamaya çalışacağız. İspat edeceğiz.

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi) (İki grup ortalamasını karşılaştırma)
######################################################

# Bağımsız iki örneklem t-testi olarak da bilinen A/B testi, iki bağımsız grubun ortalamalarını karşılaştırmak için
# kullanılan istatistiksel bir yöntemdir.
#
# A/B testinde, iki farklı gruptaki belirli bir metrik veya sonuç değişkeninin ortalama değerleri arasında anlamlı bir
# fark olup olmadığını belirlemeyi amaçlarız. Bu yöntem pazarlama, ürün geliştirme ve kullanıcı deneyimi araştırması
# gibi çeşitli alanlarda yaygın olarak kullanılmaktadır.
#
# Katılımcıları veya denekleri iki gruba ayırarak, bir grubu belirli bir uygulamaya veya koşula (A grubu), diğer grubu
# ise farklı bir uygulamaya veya koşula (B grubu) maruz bırakırız. Daha sonra her iki grup için de istenen sonucu veya
# ilgilenilen ölçütü ölçeriz. Bağımsız iki örneklemli t-testi, iki grup arasında gözlemlenen ortalama farkının
# istatistiksel olarak anlamlı olup olmadığını veya şans eseri oluşup oluşmadığını değerlendirmemizi sağlar.
#
# A/B testi sayesinde, dönüşüm oranlarında, etkileşim seviyelerinde veya diğer ilgili ölçütlerde bir artış olup
# olmadığına bakılmaksızın, hangi tedavinin veya koşulun daha iyi sonuçlara yol açtığını belirleyerek veriye dayalı
# kararlar alabiliriz.
#
# Genel olarak A/B testi, farklı yaklaşımların etkinliği hakkında değerli bilgiler sağlayarak stratejileri optimize
# etmemize ve istatistiksel kanıtlara dayalı bilinçli seçimler yapmamıza yardımcı olur.

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# İki grup karşılaştırma testidir.

# H0:M1=M2
# H0: yokluk hipotezidir. Sınayacak olduğumuz durum H0 hipotezindedir. yani iki grup ortalaması arasında fark yoktur der
# Bir diğer ifadeyle iki grup ortalamsı birbirine eşittir der.

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# ML modeli anlamlı farklılık oluşturabildi mi ?
# H0: M1 = M2
# H0: M1 != M2

#  A (Eski)          B (ML)
# Gün   Gelir       Gün   Gelir
#  1     12          1      14
#  2     15          2      8
#  3     20          3      30
#  ...   ...         ...    ...
#  ...   ...         ...    ...
#  40    21           40     28

# n: gözlem sayısı
# Aort, Bort: ortalama
# s1,s2: standart sapma
# n1 = 40, n2 = 40
# Aort = 18K, Bort = 20K
# s1 = 5, s2 = 10

# th = 1,1314<1,99=tt
# th>tt olduğundan
# Sonuç: H0 REDDEDİLEMEZ.

# Biz test istatiği kullanmak yerine p value değerini kullancaağız.
# p-value<0.05 durumuna göre yorum yapacağız. Eğer p-value < 0.05 ise H0 red edilir.

############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
############################


df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

# shapiro testi bir değişkenin dağılımının normal dağılıp dağılmadığını test eder.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

# Eğer normallik varsayımı sağlanıyorsa ttest kullanılabilir.
#
# Eğer Normallik varsayımı sağlanıyor, varyans homjenliği varsayımı sağlanıyorsa ttest kullanılabilir.
#
# Eğer normallik varsayımı sağlanıyor, varyans homejenliği sağlanmıyorsa da ttest kullanılabilir. Bu durumda varyans
# homejenliği varsayımı sağlanmıyorsa equal_var=False olması lazım.

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# P-değeri 0,05'ten büyük olduğu için H0 reddedilemez.
# Sigara içenlerin ve içmeyenlerin ortalama hesaplamaları arasında istatistiksel olarak anlamlı bir fark yoktur.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))





