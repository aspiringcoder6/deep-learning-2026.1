# ASAP-AES · chấm điểm bài luận

Repo này đọc bài luận có nhãn của ASAP-AES, cố định train/validation/test cho từng mã đề, và cung cấp metric QWK/MSE, mô hình Ridge/HAN cùng giao diện mẫu chạy local. Đề đang khảo sát là **essay_set 1**. Giao diện hiện chỉ cho **điểm giả lập**; chưa kết nối mô hình.

## Cấu trúc

| Đường dẫn | Nội dung |
| --- | --- |
| `src/essay_scoring/data.py` | Đọc TSV, làm sạch, tạo và kiểm manifest |
| `src/essay_scoring/metrics.py` | QWK và MSE theo thang điểm gốc |
| `src/essay_scoring/models/` | Ridge và Hierarchical Attention Network |
| `data/splits/set-1-v1.json` | Manifest cố định theo `essay_id`, seed 42 |
| `data/analysis/set-1-summary.json` | Thống kê đề 1 |
| `outputs/asap-aes/experiments.xlsx` | Tab **Thí nghiệm**, lý do chọn đề và phân phối điểm |
| `demo/index.html` | Demo local, điểm giả lập có nhãn |

## Cài môi trường

Dùng Python 3.11+ có `pip`. Trong PowerShell tại thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
```

Lệnh `prepare`, metric và demo HTML chỉ cần thư viện chuẩn Python. Huấn luyện Ridge/HAN cần các gói trong `pyproject.toml`. Xem [hướng dẫn cài PyTorch](https://pytorch.org/get-started/locally/) nếu dùng GPU.

## Dữ liệu có nhãn

Tải `training_set_rel3.tsv` từ [ASAP-AES trên Kaggle](https://www.kaggle.com/competitions/asap-aes/data) vào `data/raw/`. Kaggle có thể yêu cầu đăng nhập và chấp nhận điều khoản. Bản sao công khai đã dùng để kiểm tra repo là [TasfiaS/ASAP-AES trên Hugging Face](https://huggingface.co/datasets/TasfiaS/ASAP-AES/blob/main/training_set_rel3.tsv), SHA-256 `cf04127eaff3224ce4aab64654e62e5738d3d672c61bb998546dd1674c60d312` (16.337.165 byte). File gốc được bỏ qua bởi Git vì chứa bài viết của học sinh.

Loader đọc bốn trường bắt buộc `essay_id`, `essay_set`, `essay`, `domain1_score`. Dữ liệu tải về có 12.976 hàng; cả bốn cột đều hiện diện và `domain1_score` có nhãn ở mọi hàng. File dùng mã hóa Latin-1, nên loader tự thử UTF-8 rồi Latin-1. Lệnh dưới đây xác nhận dữ liệu và tái tạo các báo cáo đề 1:

```powershell
python -m essay_scoring prepare --essay-set 1 --seed 42
```

Lệnh sẽ không ghi đè manifest đã chốt nếu kết quả khác. Đề 1 có 1.783 bài sau làm sạch (không có bài trống hoặc trùng nội dung), [thang rubric **2–12**](https://aclanthology.org/2022.coling-1.240/). Thống kê độ dài và phân phối chỉ tính trên **train**: trung vị 367 từ; điểm 8 có 537/1.427 bài (37,6%). Đề này có số bài đủ lớn, độ dài phù hợp với cả TF-IDF và HAN, và thang điểm có nhiều mức để so sánh QWK/MSE. Chi tiết ở tab **Thí nghiệm** và `data/analysis/set-1-summary.json`.

## Quy ước dự đoán

Đầu vào thống nhất là mã đề và nội dung bài luận:

```json
{"essay_set": 1, "essay": "..."}
```

Đầu ra thống nhất là điểm dự đoán trên thang gốc và giới hạn thang điểm; `rounded_score` dùng để tính QWK:

```json
{"essay_set": 1, "predicted_score": 8.4, "rounded_score": 8, "score_range": {"min": 2, "max": 12}}
```

Lệnh `predict` cũng trả thêm tên mô hình. Giá trị điểm được chặn vào thang rubric; làm tròn `.5` lên mức kế tiếp. MSE dùng giá trị liên tục đã chặn, QWK dùng giá trị đã làm tròn. QWK trả `null` khi cả hai dãy cùng là một điểm cố định.

## Chia tập và giữ test riêng

`prepare` loại bài trống, chuẩn hóa chữ hoa/thường và khoảng trắng để bỏ bản trùng (giữ `essay_id` thấp nhất), rồi xáo trộn với seed 42. Manifest phiên bản 1 ghi checksum dữ liệu, seed, quy tắc làm sạch và danh sách `essay_id` từng tập. Với đề 1: train **1.427**, validation **178**, test **178**. Hàm kiểm tra xác nhận mọi ID xuất hiện đúng một lần, đúng seed, và không có nội dung trùng giữa các tập.

Huấn luyện chỉ dùng train và validation từ manifest. Thống kê khảo sát về độ dài và điểm chỉ dùng train. Sau khi so sánh các run trên validation, lệnh `finalize` ghi một lựa chọn cố định cùng lý do và checksum run/checkpoint. `evaluate --final-model` chỉ mở test cho đúng run đó; không dùng kết quả test để chọn siêu tham số.

```powershell
python -m essay_scoring train --essay-set 1 --model ridge --run-dir outputs/set-1/ridge
python -m essay_scoring train --essay-set 1 --model han --epochs 5 --run-dir outputs/set-1/han
python -m essay_scoring predict --essay-set 1 --run-dir outputs/set-1/ridge --essay-file path/to/essay.txt
# Chỉ sau khi chọn run bằng validation:
python -m essay_scoring finalize --run-dir outputs/set-1/ridge --note "Ridge được chọn theo QWK/MSE validation"
python -m essay_scoring evaluate --final-model --run-dir outputs/set-1/ridge
```

Run ghi checksum dữ liệu, checksum manifest, seed, ID chia tập và metric validation trong `run.json`. Mỗi run cần thư mục mới. Chỉ mở checkpoint đáng tin cậy: Ridge dùng Python pickle.

## Checkpoint đề 1

Hai run `ridge-v2` và `han-v2` được train lại vì bản bàn giao v1 thiếu checkpoint. Các run v2 dùng manifest đề 1 đã chốt, seed 42, với 1.427 bài train và 178 bài validation. HAN train 5 epoch và chọn epoch 3 theo MSE validation. Gói phát hành có cả hai model, `run.json`, log HAN, manifest, checksum từng file và phiên bản thư viện đã dùng. [Bản phát hành ASAP set 1 v2](https://github.com/aspiringcoder6/deep-learning-2026.1/releases/tag/asap-set1-v2)

| Run | QWK validation | MSE validation |
| --- | ---: | ---: |
| `ridge-v2` | 0.722835 | 0.952226 |
| `han-v2` | 0.715822 | 0.987638 |

Tải và mở gói trong PowerShell tại thư mục repo. Môi trường đã kiểm tra là Windows, Python 3.12 và PyTorch CPU. Bước kiểm tra SHA-256 xác nhận gói tải về đúng bản phát hành.

```powershell
$modelsZip = Join-Path $env:TEMP 'asap-set1-v2.zip'
curl.exe --fail --location 'https://github.com/aspiringcoder6/deep-learning-2026.1/releases/download/asap-set1-v2/asap-set1-v2.zip' --output $modelsZip
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
$expectedHash = '8858723b2400e91728dd289deeadcc57305e2aff1e3fa80cbc41838b936a963c'
if ((Get-FileHash -LiteralPath $modelsZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) { throw 'Archive checksum mismatch.' }
Expand-Archive -LiteralPath $modelsZip -DestinationPath outputs/set-1 -Force
python -m pip install -r outputs/set-1/requirements-cpu.txt
python -m essay_scoring predict --essay-set 1 --run-dir outputs/set-1/ridge-v2 --essay-file path/to/essay.txt
python -m essay_scoring predict --essay-set 1 --run-dir outputs/set-1/han-v2 --essay-file path/to/essay.txt
```

Metric validation của v2 được tính lại từ checkpoint đã nạp. Kiểm tra reload dùng batch size đã lưu và so sánh điểm liên tục với `rel_tol=1e-5`, `abs_tol=1e-6`. Checksum JSON tương thích với LF và CRLF, kể cả checksum cũ. TV3 dùng validation để chọn run trước khi gọi `finalize` và mở test.

## Demo local

```powershell
python -m http.server 8000 --directory demo
```

Mở `http://localhost:8000`. Giao diện có mã đề, ô đề bài, ô dán bài luận, nút chấm và vùng kết quả. Điểm giả lập là giá trị cố định theo thang điểm, không phản ánh chất lượng bài; nội dung đề bài trong demo chỉ để hiển thị.
