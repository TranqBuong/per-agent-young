# Multi-Agent Quality Engineer — QE Agent

Hệ thống tự động phân tích requirement và sinh test case, test data, automation code qua 3 AI agent.

## Kiến trúc

```
Requirement Input
      │
      ▼
Agent 1 — Requirement Analyzer
   Phân tích requirement, trích xuất scenarios, quality score
      │
      ▼
Agent 2 — Test Case & Payload Generator
   Sinh test cases theo kỹ thuật EP/BVA/DT/ST/EG...
      │
      ▼
Agent 3 — Automation Code Writer
   Sinh code tự động (pytest / playwright / k6 / selenium / postman)
      │
      ▼
   RTM + Export
```

## Yêu cầu

- Python 3.11+
- Groq API key (đăng ký tại [console.groq.com](https://console.groq.com))

## Cài đặt

```bash
# 1. Clone / giải nén project
cd per-agent-young

# 2. Tạo virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình API key
cp .env.example .env
# Mở .env và điền GROQ_API_KEY của bạn
```

## Chạy server

```bash
# Cách 1 — load .env tự động
export $(cat .env | xargs) && uvicorn backend.main:app --reload --port 8080

# Cách 2 — set thủ công
export GROQ_API_KEY=your-key-here
uvicorn backend.main:app --reload --port 8080
```

Mở trình duyệt tại: **http://localhost:8080**

## Sử dụng

1. **Nhập requirement** — dán text, upload file (.txt, .md, .pdf, .docx, .yaml, .json), hoặc fetch từ URL
2. **Analyze** — Agent 1 phân tích và hiển thị Quality Score + Scenarios
3. **Confirm** — Xác nhận scenarios (có thể chỉnh sửa inline hoặc thêm thủ công)
4. **Generate Test Cases** — Agent 2 sinh test cases theo kỹ thuật phù hợp
5. **Generate Code** — Agent 3 sinh automation code theo framework đã chọn
6. **Export** — Tải RTM (Excel) hoặc Test Cases (Excel)

## Input được hỗ trợ

| Loại | Định dạng |
|---|---|
| Text | Dán trực tiếp vào textarea |
| File | .txt, .md, .pdf, .docx |
| Swagger / OpenAPI | .yaml, .json |
| URL | HTTP/HTTPS |

## Frameworks

| Framework | Ngôn ngữ | Use case |
|---|---|---|
| pytest | Python | API testing |
| playwright | Python | Web E2E |
| k6 | JavaScript | Performance |
| selenium | Python | Web browser |
| postman | JSON | API collection |

## Chạy tests

```bash
python -m pytest tests/ -v
```

## Biến môi trường

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key |
| `GROQ_MODEL` | ❌ | Model chính (default: `llama-3.1-8b-instant`) |
| `GROQ_MODEL_LIGHT` | ❌ | Model nhẹ cho preview (default: `llama-3.1-8b-instant`) |

## Cache

Kết quả LLM được cache trên disk (thư mục `storage/`), TTL 2 ngày, hash key SHA256. Cache có **similarity fallback**: nếu requirement text mới > 85% giống text đã cache, hệ thống trả về kết quả cũ thay vì gọi LLM lại.
