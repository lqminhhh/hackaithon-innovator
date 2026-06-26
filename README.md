# VietMind MCQ - HackAIThon 2026

![alt text](/assets/image.png)

#### _Đọc bản tiếng Anh tại_ <kbd><a href="docs/translations/README_en.md">English</a></kbd>

Một AI Agent giải câu hỏi trắc nghiệm tiếng Việt, được xây dựng dựa trên cơ chế
suy luận thích ứng cho **[HackAIthon 2026](https://hackaithon.vsds.vn/) -
Track C (Innovator)**.

VietMind MCQ là bản dự thi cuối cùng của nhóm Cow 🐄. Hệ thống chạy hoàn
toàn trong container, sử dụng một LLM chạy trên máy, đọc file kiểm tra từ
`/code/private_test.json`, và ghi kết quả ra `/code/submission.csv` cùng
`/code/submission_time.csv`.

## I. Kiến Trúc

![Kiến trúc VietMind MCQ](assets/architecture/vietmind_architecture.png)

## II. Thông Tin Bài Nộp

| Yêu cầu | Bài nộp của nhóm |
| --- | --- |
| Đội thi | `Cow` 🐄 |
| Thành viên | `Lê Quang Minh, Nguyễn Ngô Thảo Uyên, Nguyễn Minh Việt` |
| Tổ chức | `Denison University, The Ohio State University` |
| Cuộc thi | [HackAIthon 2026](https://hackaithon.vsds.vn/) |
| Mô hình | `Qwen/Qwen3.5-4B` |
| Giới hạn mô hình | Một LLM mở, dưới 5B tham số |
| Suy luận | Offline, chỉ dùng một mô hình |
| Docker image | `powato/hackaithon-cow:latest` |
| Kích thước image | khoảng 16.2 GB |
| Runner cuối | `src/v03_gamma.py` |
| Input chính thức | `/code/private_test.json` |
| Output chính thức | `/code/submission.csv` và `/code/submission_time.csv` |
| Cột output | `submission.csv`: `qid,answer`; `submission_time.csv`: `qid,answer,time` |
| GPU mục tiêu | Chính thức hỗ trợ NVIDIA Ampere trở lên, ít nhất 32 GB VRAM |

## III. Mục Đích

Pipeline phân tích câu hỏi trắc nghiệm tiếng Việt, xác định câu hỏi theo dạng
bài, chạy suy luận theo từng route, trích xuất đáp án bằng cơ chế ràng buộc lựa
chọn, và ghi một đáp án hợp lệ cho mỗi `qid`.

Nói ngắn gọn, hệ thống cố gắng dành thêm thời gian suy nghĩ cho những câu khó
giải quyết hơn, nhưng vẫn giữ kỷ luật để luôn tạo ra một file nộp bài sạch.

Nhánh cuối cùng là `v03_gamma`. Chúng tôi chọn nhánh này vì đây là điểm cân
bằng thực tế nhất giữa độ chính xác trên public set, tốc độ chạy, và độ ổn định
trên GPU 32 GB VRAM.

## IV. Ý Tưởng Chính

VietMind MCQ được xây dựng quanh ý tưởng suy luận thích ứng: không phải câu hỏi
nào cũng nên nhận cùng một lượng tính toán. Một số câu là kiểm tra kiến thức
trực tiếp, một số cần đọc kỹ đoạn văn, một số là bài khoa học cần tính toán, và
một số có dạng an toàn hoặc lựa chọn từ chối.

Ý tưởng này cũng đến từ trải nghiệm của chính chúng tôi với các kỳ thi đại học
ở Việt Nam. Khi còn là học sinh phổ thông, chúng tôi học được rằng không thể xử
lý mọi câu trắc nghiệm theo cùng một cách. Câu kiến thức đơn giản có thể trả lời
nhanh. Câu toán thường cần nháp. Câu đọc hiểu thường cần quay lại đọc đoạn văn.
Các đáp án dễ nhầm lẫn cần so sánh kỹ giữa các lựa chọn.

VietMind MCQ làm theo bản năng làm bài đó: trả lời nhanh khi câu hỏi đơn giản,
và chậm lại khi cấu trúc câu hỏi cho thấy có rủi ro.

Mỗi route được xử lý khác nhau:

- Câu `READING` có thể dùng self consistency kiểu đọc lại khi chi tiết trong
  bài là quan trọng.
- Câu `STEM` nhận suy luận kỹ hơn vì lỗi tính toán nhỏ có thể làm đổi đáp án.
- Câu `KNOWLEDGE` được cấp thêm compute khi có nhiều lựa chọn, lựa chọn mơ hồ,
  hoặc cấu trúc đáp án khó.
- Câu `SAFETY` có thể dùng luật chọn đáp án từ chối khi câu hỏi yêu cầu hành vi
  không an toàn.

## V. Tóm Tắt Kết Quả

| Phiên bản | Điểm public | Thời gian trên GPU của nhóm |
| --- | --- | --- |
| `v02_gamma` | 85.31% | 12.77 giây/câu |
| `v03_alpha` | 84.23% | 3.87 giây/câu |
| `v03_gamma` | **85.96%** | 7.98 giây/câu |
| `v03_delta` | 87.04% | 27.53 giây/câu |

`v03_delta` có điểm public cao hơn, nhưng nặng hơn nhiều và vẫn có rủi ro OOM
trên các lần chạy dài với bộ nhớ hạn chế. Vì vậy, nhánh nộp cuối cùng là
`v03_gamma`.

Các số liệu runtime ở trên được đo trên GPU RTX 24 GB của nhóm. Chúng có
ý nghĩa để so sánh giữa các phiên bản, không phải để dự đoán chính xác thời gian
chạy trên máy chấm. Chúng tôi không khuyến nghị dùng NVIDIA T4 cho bài chấm
chính thức, vì T4 có biên bộ nhớ rất chặt, inference chậm hơn nhiều, và có thể
dẫn đến khác biệt runtime hoặc lỗi khi chạy Docker/vLLM.

Chi tiết lịch sử phiên bản: [docs/version_results.md](docs/version_results.md)

## VI. Báo Cáo

- Báo cáo tiếng Việt: [docs/report/report_vi.md](docs/report/report_vi.md)
- Báo cáo tiếng Anh: [docs/report/report_en.md](docs/report/report_en.md)
- Slide thuyết trình: [docs/report/presentation_slide.pdf](docs/report/presentation_slide.pdf)

## VII. Hướng Dẫn Chạy Cho Ban Giám Khảo

### Yêu Cầu

- **NVIDIA CUDA GPU** có ít nhất **32 GB** VRAM
- Chính thức hỗ trợ NVIDIA Ampere trở lên, ví dụ RTX 3090/4090,
  RTX A5000/A6000, A100, L40/L40S, hoặc GPU CUDA tương đương
- Hỗ trợ kỹ thuật nhưng không khuyến nghị: Tesla T4 16 GB. Vui lòng không dùng
  T4 cho bài chấm chính thức vì T4 không đạt mục tiêu 32 GB VRAM, có thể quá
  chậm, biên bộ nhớ quá sát, hoặc gây khác biệt/lỗi khi chạy Docker với vLLM
- Docker
- Gói `nvidia-container-toolkit`, để `docker run --gpus all` hoạt động
- Nên có ít nhất 25 GB dung lượng đĩa trống cho Docker image khoảng 16.2 GB,
  extracted layers, cache, và output

**Lưu ý:** Cấu hình nộp cuối đã được điều chỉnh cho GPU 32 GB VRAM. Với private
set khoảng 2000 câu, thời gian chạy vẫn có thể kéo dài nhiều giờ vì Wave 2 dùng
self-consistency cho các câu khó. Vui lòng cấp đủ thời gian chạy và tránh dùng
GPU dưới 32 GB VRAM cho bài chấm chính thức.

### Pull Image

```bash
docker pull powato/hackaithon-cow:latest
```

### Kiểm Tra Trước Khi Chạy

Trước khi chạy bài nộp, các lệnh sau nên chạy thành công:

```bash
docker version
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
df -h .
```

Kết quả mong đợi:

- `docker version` hiển thị cả `Client` và `Server`
- `nvidia-smi` chạy được bên trong CUDA container
- ổ đĩa hiện tại có đủ dung lượng cho Docker image và file output

### Chạy Container

Đặt file test chính thức vào đường dẫn `/code/private_test.json` trong
container. Không mount cả thư mục vào `/code`, vì điều đó sẽ che mất source code
có sẵn trong image.

```bash
docker run --name cow-vietmind-run --gpus all --ipc=host \
  -v "$PWD/private_test.json:/code/private_test.json" \
  powato/hackaithon-cow:latest
```

Sau khi chạy xong, hai file sau cần tồn tại trong container theo yêu cầu BTC:

```text
/code/submission.csv
/code/submission_time.csv
```

Khi chạy local, có thể copy hai file này ra ngoài rồi xóa container:

```bash
docker cp cow-vietmind-run:/code/submission.csv ./submission.csv
docker cp cow-vietmind-run:/code/submission_time.csv ./submission_time.csv
docker rm cow-vietmind-run
```

Các file này cần có định dạng:

```csv
qid,answer
```

```csv
qid,answer,time
```

### Tên Input Được Hỗ Trợ

Container kiểm tra input theo thứ tự:

1. `/code/private_test.json`
2. `/code/private_test.csv`
3. `/data/private_test.csv`
4. `/data/public_test.csv`
5. `/data/private_test.json`
6. `/data/public_test.json`

BTC path chính thức là `/code/private_test.json`. Các đường dẫn `/data/...`
được giữ lại để tương thích với các lần chạy local cũ. Input CSV có thể dùng
các cột đáp án như `A,B,C,D,...`; câu hỏi có nhiều hơn bốn lựa chọn vẫn được hỗ
trợ.

## VIII. Hướng Dẫn Chạy Cho Developer

Cài dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Chạy pipeline cuối trên máy local:

```bash
python src/v03_gamma.py \
  --input data/public-test_1780368312.json \
  --output data/submissions/submission_v03_gamma.csv \
  --trace-output data/traces/trace_v03_gamma.jsonl \
  --safe-mode
```

Chạy theo cùng kiểu entrypoint như Docker:

```bash
./run.sh data/private_test.csv output/submission.csv output/trace.jsonl output/submission_time.csv
```

Chạy test:

```bash
python3.11 -m pytest
```

## IX. Lỗi Thường Gặp

Xem [docs/faq.md](docs/faq.md) để biết cách xử lý các lỗi thực tế, bao gồm:

- Docker daemon chưa chạy
- cần khởi động Docker thủ công trong một số môi trường notebook hoặc cloud
- `docker run --gpus all` không hoạt động
- không đủ dung lượng đĩa cho Docker image
- `vLLM unavailable`
- thiếu file input trong `/data`

## X. Ghi Chú

- Đường chạy cuối hoạt động offline khi inference.
- Đường chạy cuối chỉ dùng một LLM mở: `Qwen/Qwen3.5-4B`, dưới giới hạn 5B
  tham số.
- Không dùng RAG, embedding model, reranker, semantic router model, hoặc LLM thứ
  hai.
- Thiết lập runtime nằm trong
  [configs/pipeline_config.yaml](configs/pipeline_config.yaml).

## XI. Lời Cảm Ơn

Xin cảm ơn HackAIthon 2026, Hội Sinh Viên Việt Nam, VSDS, Vietcombank, VNPT AI,
ban tổ chức, các nhà tài trợ, đội ngũ hỗ trợ kỹ thuật, mentor, và ban giám khảo
đã tạo ra một sân chơi để sinh viên được xây dựng, thử nghiệm, và học hỏi về AI
Agent.
