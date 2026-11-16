# Dữ liệu ASAP-AES

Đặt file có nhãn `training_set_rel3.tsv` tại `data/raw/`. Nguồn gốc: [Kaggle ASAP-AES](https://www.kaggle.com/competitions/asap-aes/data). Bản sao dùng ở đây: [TasfiaS/ASAP-AES](https://huggingface.co/datasets/TasfiaS/ASAP-AES/blob/main/training_set_rel3.tsv), SHA-256 `cf04127eaff3224ce4aab64654e62e5738d3d672c61bb998546dd1674c60d312`.

File TSV cần `essay_id`, `essay_set`, `essay` và nhãn `domain1_score`. `data/raw/` không được commit. Chạy `python -m essay_scoring prepare --essay-set 1 --seed 42` để xác minh và tái tạo manifest cùng thống kê. Manifest `data/splits/set-1-v1.json` và báo cáo `data/analysis/set-1-summary.json` được lưu trong repo để các thành viên dùng cùng tập chia. Dùng đúng file có checksum trên khi chạy train/evaluate.
