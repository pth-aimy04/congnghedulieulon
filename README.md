# Techcombank BigData Pipeline — VSCode Setup

## 1. Cài đặt môi trường

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Tải dataset (bắt buộc)

Tạo thư mục `data/` trong cùng folder với file .py

### Online Retail (bán lẻ)
- Link: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- Đặt tên file: `data/online_retail.csv`
- Schema: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### Bank Marketing (ngân hàng)
- Link: https://archive.ics.uci.edu/dataset/222/bank+marketing
- File gốc tên `bank-full.csv`, dấu phân cách là `;`
- Đặt tên file: `data/bank-full.csv`

## 3. Cấu trúc thư mục

```
project/
├── techcombank_bigdata.py   ← file chính
├── requirements.txt
├── data/
│   ├── online_retail.csv
│   └── bank-full.csv
├── delta_warehouse/         ← tự tạo khi chạy (Delta tables)
├── output_charts/           ← tự tạo khi chạy (biểu đồ)
└── stream_input/            ← tự tạo khi chạy (streaming)
```

## 4. Chạy

```bash
python techcombank_bigdata.py
```

Thời gian chạy ước tính: **5–15 phút** tùy cấu hình máy.

## 5. Lưu ý Java

PySpark yêu cầu **Java 8 hoặc 11**. Kiểm tra bằng:
```bash
java -version
```
Nếu chưa có: tải tại https://adoptium.net/

## 6. Output

- `output_charts/rfm_distribution.png` — phân phối RFM
- `output_charts/kmeans_clusters.png` — phân cụm K-Means
- `output_charts/ml_results.png` — AUC + Feature Importance
- `delta_warehouse/` — tất cả bảng Delta (Bronze/Silver/Gold)

## 7. Các thay đổi so với bản Databricks

| Hạng mục | Databricks | VSCode (bản này) |
|---|---|---|
| ML Engine | scikit-learn + .toPandas() | **PySpark MLlib 100%** |
| CrossValidation | Không có | **3-fold CV cho LR** |
| Bài toán ML | 2 model (Marketing) | **3 model: LR + RF + GBT Credit** |
| Window function | No partition (OOM risk) | **partitionBy("job")** |
| Delta partition | Không | **partitionBy Country/job** |
| Streaming source | Databricks Volume | **CSV local** |
| Explain plan | Không có | **df.explain() ở 2 chỗ** |

## 8. CI/CD với GitHub Actions

Dự án sử dụng **GitHub Actions** để tự động hóa quy trình phát triển phần mềm.

###  CI (Continuous Integration)
Tự động chạy mỗi khi có `push` hoặc `pull request`.

Thực hiện các bước:
- Cài đặt môi trường Python 3.10
- Cài đặt các thư viện từ `requirements.txt`
- Cài đặt Java để chạy PySpark
- Kiểm tra syntax của file Python
- Kiểm tra import các thư viện (PySpark, Delta, Pandas, v.v.)

 Mục tiêu: đảm bảo code luôn chạy được và không bị lỗi môi trường.

---

###  CD (Continuous Delivery)
Tự động chạy sau khi CI thành công.

Thực hiện:
- Đóng gói toàn bộ project thành file `.zip`
- Upload artifact lên GitHub

 Mục tiêu: sẵn sàng deploy hoặc bàn giao project.

---

### Artifact
Sau mỗi lần chạy CD, có thể tải file đóng gói tại:

**GitHub → Actions → Artifacts**

---

### Workflow files

- `.github/workflows/ci.yml` — kiểm tra và build project (CI)
- `.github/workflows/cd.yml` — đóng gói và phát hành (CD)