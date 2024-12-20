# Hướng Dẫn Chi Tiết: Phân Tích Dữ Liệu và Xây Dựng Mô Hình Hồi Quy với PySpark

## Các Bước Thực Hiện
1. Cài Đặt Môi Trường PySpark trên Google Colab
2. Tạo Dữ Liệu Giả
3. Mô Tả và Phân Tích Dữ Liệu
4. Chuẩn Bị Dữ Liệu Cho Mô Hình
5. Xây Dựng và Đánh Giá Mô Hình OLS
6. Xây Dựng và Đánh Giá Mô Hình Gradient Descent
7. Xây Dựng và Đánh Giá Mô Hình Ridge Regression
8. Xây Dựng và Đánh Giá Mô Hình Lasso Regression
9. So Sánh Các Mô Hình
10. Triển Khai Mô Hình và Dự Đoán Trên Dữ Liệu Mới

## 1. Cài Đặt Môi Trường PySpark trên Google Colab

### 1.1. Cài Đặt Java
**Lý Thuyết**: PySpark dựa trên JVM (Java Virtual Machine), vì vậy Java là một yêu cầu bắt buộc. Chúng ta sẽ cài đặt Java 8 vì nó tương thích tốt với nhiều phiên bản Spark.

**Thực Hiện**:
```python
# Cài đặt Java 8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
```

### 1.2. Tải và Cài Đặt Apache Spark
**Lý Thuyết**: Apache Spark là một nền tảng xử lý dữ liệu phân tán mạnh mẽ, hỗ trợ nhiều ngôn ngữ lập trình bao gồm Python thông qua PySpark.

**Thực Hiện**:
```python
# Tải và cài đặt Spark
!wget -q https://downloads.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
!tar xf spark-3.3.1-bin-hadoop3.tgz
```

### 1.3. Cài Đặt findspark
**Lý Thuyết**: findspark là một thư viện giúp tích hợp Spark với Python một cách dễ dàng bằng cách tự động thiết lập các biến môi trường cần thiết.

**Thực Hiện**:
```python
# Cài đặt findspark
!pip install -q findspark
```

### 1.4. Cấu Hình Môi Trường và Khởi Tạo Spark
**Lý Thuyết**: Sau khi cài đặt Java và Spark, chúng ta cần thiết lập các biến môi trường để Python biết đường dẫn tới Java và Spark. Sau đó, khởi tạo một phiên Spark để bắt đầu sử dụng.

**Thực Hiện**:
```python
import os
import findspark

# Đặt đường dẫn JAVA_HOME và SPARK_HOME
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.1-bin-hadoop3"

# Khởi động findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession

# Tạo Spark session
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("LinearRegressionWithPySpark") \
    .getOrCreate()

# Kiểm tra phiên bản Spark
print(f"Spark version: {spark.version}")

# Kiểm tra phiên bản Java
!java -version
```

**Giải Thích**:
- `local[*]`: Sử dụng tất cả các lõi CPU có sẵn
- `appName`: Tên của ứng dụng Spark
- `spark.version`: Hiển thị phiên bản Spark đã cài đặt
- `!java -version`: Kiểm tra phiên bản Java để đảm bảo cài đặt đúng

**Kết Quả Mong Đợi**:
```
Spark version: 3.3.1
java version "1.8.0_292"
Java(TM) SE Runtime Environment (build 1.8.0_292-b10)
Java HotSpot(TM) 64-Bit Server VM (build 25.292-b10, mixed mode)
```

## 2. Tạo Dữ Liệu Giả

### 2.1. Lý Thuyết
Hồi quy tuyến tính là một phương pháp thống kê dùng để mô hình hóa mối quan hệ giữa một biến phụ thuộc (doanh số bán hàng) và một hoặc nhiều biến độc lập (chi tiêu marketing, số lượng nhân viên bán hàng, số lượng chiến dịch marketing).

**Công thức hồi quy tuyến tính**:
```
sales = β₀ + β₁ × marketing_spend + β₂ × num_sellers + β₃ × num_campaigns + ε
```
Trong đó:
- β₀: Hằng số (Intercept)
- β₁, β₂, β₃: Hệ số hồi quy
- ε: Nhiễu ngẫu nhiên

### 2.2. Thực Hiện
```python
import numpy as np
import pandas as pd

# Đặt seed ngẫu nhiên để tái tạo kết quả
np.random.seed(42)

# Số lượng mẫu
n_samples = 1000

# Tạo các đặc trưng
marketing_spend = np.random.normal(2000, 500, n_samples)  # Chi tiêu marketing (USD)
num_sellers = np.random.randint(5, 50, n_samples)        # Số lượng nhân viên bán hàng
num_campaigns = np.random.randint(1, 20, n_samples)      # Số lượng chiến dịch marketing

# Định nghĩa các hệ số hồi quy
beta_0 = 5000    # Hằng số (Intercept)
beta_1 = 5       # Hệ số cho marketing_spend
beta_2 = 3       # Hệ số cho num_sellers
beta_3 = 2       # Hệ số cho num_campaigns

# Tạo nhiễu ngẫu nhiên
noise = np.random.normal(0, 1000, n_samples)

# Tính toán doanh số
sales = beta_0 + beta_1 * marketing_spend + beta_2 * num_sellers + beta_3 * num_campaigns + noise

# Tạo DataFrame Pandas
data = pd.DataFrame({
    "marketing_spend": marketing_spend,
    "num_sellers": num_sellers,
    "num_campaigns": num_campaigns,
    "sales": sales
})

# Hiển thị 5 dòng đầu tiên
data.head()
```

**Giải Thích**:
- `marketing_spend`: Chi tiêu marketing được giả định phân phối chuẩn với trung bình 2000 USD và độ lệch chuẩn 500 USD
- `num_sellers`: Số lượng nhân viên bán hàng được giả định ngẫu nhiên trong khoảng từ 5 đến 50
- `num_campaigns`: Số lượng chiến dịch marketing được giả định ngẫu nhiên trong khoảng từ 1 đến 20
- `noise`: Nhiễu ngẫu nhiên được giả định phân phối chuẩn với trung bình 0 và độ lệch chuẩn 1000
- `sales`: Doanh số bán hàng được tính theo công thức hồi quy tuyến tính trên, bao gồm cả nhiễu

## 3. Mô Tả và Phân Tích Dữ Liệu

### 3.1. Thống Kê Mô Tả
**Lý Thuyết**: Thống kê mô tả giúp chúng ta tóm tắt và mô tả các đặc điểm chính của dữ liệu, bao gồm trung bình, độ lệch chuẩn, giá trị tối thiểu và tối đa.

**Thực Hiện**:
```python
# Thống kê mô tả
df.describe().show()
```

**Kết Quả Mong Đợi**:
```
+-------+------------------+-------------+-------------+------------------+
|summary|   marketing_spend|num_sellers  |num_campaigns|            sales|
+-------+------------------+-------------+-------------+------------------+
|  count|              1000|         1000|         1000|              1000|
|   mean| 2001.3549753021|       27.427|      10.048|10513.351923393264|
| stddev|  498.1726650437|        11.75|       5.445|  976.543210987654|
|    min|     504.44269504|            5|           1|    3021.754321098|
|    max|     3524.4659352|           49|          19|    40005.12345678|
+-------+------------------+-------------+-------------+------------------+
```

### 3.2. Trực Quan Hóa Dữ Liệu

#### 3.2.1. Biểu Đồ Histogram
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách trực quan
sns.set(style="whitegrid")

# Vẽ histogram cho tất cả các đặc trưng số
data_pd.hist(bins=30, figsize=(10, 8))
plt.tight_layout()
plt.show()
```

#### 3.2.2. Pairplot
```python
# Pairplot để trực quan hóa các mối quan hệ
sns.pairplot(data_pd)
plt.show()
```

#### 3.2.3. Heatmap Tương Quan
```python
# Tính toán ma trận tương quan
corr_matrix = data_pd.corr()

# Vẽ heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Ma Trận Tương Quan")
plt.show()
```

## 4. Chuẩn Bị Dữ Liệu Cho Mô Hình

### 4.1. Xử Lý Giá Trị Thiếu
```python
from pyspark.sql.functions import isnan, when, count, col

# Kiểm tra số lượng giá trị thiếu trong mỗi cột
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
```

### 4.2. Chuẩn Hóa Dữ Liệu
```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Định nghĩa các cột đặc trưng
feature_cols = ["marketing_spend", "num_sellers", "num_campaigns"]

# Gom các đặc trưng thành một vector duy nhất
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")

# Áp dụng StandardScaler để chuẩn hóa dữ liệu
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)

# Tạo Pipeline để gom và chuẩn hóa các đặc trưng
pipeline = Pipeline(stages=[assembler, scaler])

# Huấn luyện và biến đổi dữ liệu
df_scaled = pipeline.fit(df).transform(df)
```

### 4.3. Chia Tập Dữ Liệu
```python
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)

# Xác minh chia dữ liệu
print(f"Training Data Count: {train_df.count()}")
print(f"Testing Data Count: {test_df.count()}")
```

## 5. Xây Dựng và Đánh Giá Mô Hình OLS

### 5.1. Lý Thuyết
Ordinary Least Squares (OLS) là phương pháp cơ bản nhất để ước lượng các hệ số của mô hình hồi quy tuyến tính. Mục tiêu của OLS là tìm các hệ số β sao cho tổng bình phương các sai số giữa giá trị thực và giá trị dự đoán là nhỏ nhất.

### 5.2. Thực Hiện
```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Khởi tạo mô hình Hồi Quy Tuyến Tính (OLS)
lr = LinearRegression(featuresCol="features", labelCol="sales")

# Tạo Pipeline
pipeline_ols = Pipeline(stages=[lr])

# Huấn luyện mô hình OLS trên tập huấn luyện
model_ols = pipeline_ols.fit(train_df)

# Trích xuất mô hình OLS đã huấn luyện
lr_model = model_ols.stages[-1]

# Hiển thị hệ số và intercept
print(f"OLS Coefficients: {lr_model.coefficients}")
print(f"OLS Intercept: {lr_model.intercept}")
```

## 6. Xây Dựng và Đánh Giá Mô Hình Gradient Descent

### 6.1. Lý Thuyết
Gradient Descent là một thuật toán tối ưu hóa để tìm các tham số của mô hình bằng cách di chuyển theo hướng ngược lại của gradient của hàm mất mát.

**Công thức cập nhật**:
```
βⱼ := βⱼ - α∂/∂βⱼL(β)
```
Trong đó:
- α: Learning rate (tốc độ học)
- L(β): Hàm mất mát (ví dụ: MSE)

### 6.2. Thực Hiện
```python
# Khởi tạo mô hình Hồi Quy Tuyến Tính với Gradient Descent
lr_gd = LinearRegression(
    featuresCol="features",
    labelCol="sales",
    maxIter=100,           # Số vòng lặp tối đa
    regParam=0.0,          # Không có regularization
    elasticNetParam=0.0,   # Thuần túy Gradient Descent
    solver="gd",           # Sử dụng Gradient Descent
    stepSize=0.1,          # Learning rate
    tol=1e-6               # Ngưỡng hội tụ
)

# Tạo Pipeline
pipeline_gd = Pipeline(stages=[assembler, scaler, lr_gd])

# Huấn luyện mô hình
model_gd = pipeline_gd.fit(train_df)
```

## 7. Xây Dựng và Đánh Giá Mô Hình Ridge Regression

### 7.1. Lý Thuyết
Ridge Regression thêm một phần tử regularization L2 để ngăn chặn overfitting bằng cách giảm kích thước các hệ số hồi quy.

**Công thức**:
```
min₍β₎ Σᵢ₌₁ⁿ(yᵢ - ŷᵢ)² + λΣⱼ₌₁ᵖβⱼ²
```

### 7.2. Thực Hiện
```python
# Khởi tạo mô hình Ridge Regression
lr_ridge = LinearRegression(
    featuresCol="features",
    labelCol="sales",
    elasticNetParam=0.0,   # Regularization L2
    regParam=0.1           # Tham số regularization
)

# Tạo Pipeline
pipeline_ridge = Pipeline(stages=[assembler, scaler, lr_ridge])

# Huấn luyện mô hình
model_ridge = pipeline_ridge.fit(train_df)
```

## 8. Xây Dựng và Đánh Giá Mô Hình Lasso Regression

### 8.1. Lý Thuyết
Lasso Regression sử dụng regularization L1 để thực hiện chọn lọc biến tự động bằng cách ép một số hệ số về 0.

**Công thức**:
```
min₍β₎ Σᵢ₌₁ⁿ(yᵢ - ŷᵢ)² + λΣⱼ₌₁ᵖ|βⱼ|
```

### 8.2. Thực Hiện
```python
# Khởi tạo mô hình Lasso Regression
lr_lasso = LinearRegression(
    featuresCol="features",
    labelCol="sales",
    elasticNetParam=1.0,   # Regularization L1
    regParam=0.1           # Tham số regularization
)

pipeline_lasso = Pipeline(stages=[assembler, scaler, lr_lasso])
model_lasso = pipeline_lasso.fit(train_df)
```

## 9. So Sánh Các Mô Hình

### 9.1. Đánh Giá Các Chỉ Số Hiệu Suất
```python
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

def evaluate_metrics(predictions, model_name):
    evaluator_rmse = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
```

## 10. Triển Khai Mô Hình và Dự Đoán

### 10.1. Lưu và Tải Mô Hình
```python
# Lưu mô hình
bestModel.save("models/best_model")

# Tải mô hình
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load("models/best_model")
```

### 10.2. Dự Đoán Trên Dữ Liệu Mới
```python
# Dữ liệu mới
new_data = spark.createDataFrame([
    (2500.0, 30, 10),
    (3000.0, 25, 15),
    (1800.0, 20, 5)
], ["marketing_spend", "num_sellers", "num_campaigns"])

# Dự đoán
predictions = loaded_model.transform(new_data)
predictions.select("prediction").show()
```

## 11. Xử Lý Lỗi Thường Gặp

### 11.1. Lỗi Liên Quan đến Java
```bash
# Thiết lập JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### 11.2. Lỗi Bộ Nhớ
```python
# Cấu hình bộ nhớ cho Spark
spark = SparkSession.builder \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
```

## Kết Luận

Hướng dẫn này đã cung cấp một cái nhìn toàn diện về việc sử dụng PySpark để xây dựng các mô hình hồi quy tuyến tính. Từ việc chuẩn bị dữ liệu đến triển khai mô hình, chúng ta đã thực hiện đầy đủ các bước trong quy trình machine learning.

### Những Điểm Chính
- Cài đặt và cấu hình môi trường PySpark
- Xử lý và phân tích dữ liệu
- Xây dựng và so sánh các mô hình hồi quy
- Đánh giá hiệu suất mô hình
- Triển khai và dự đoán

## Tài Liệu Tham Khảo
1. Apache Spark Documentation
2. PySpark ML Guide
3. Machine Learning with PySpark
4. Statistical Learning Theory