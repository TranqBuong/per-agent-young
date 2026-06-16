# Mô tả Use Case — Test Pilot: Multi-Agent QA Automation

## Bài toán

Tester tại Zalopay mất nhiều thời gian đọc tài liệu đặc tả, suy luận test case và viết automation code thủ công. Quy trình này dễ bỏ sót negative case, thiếu traceability giữa requirement và code, và lặp lại với mỗi tính năng mới.

## Giải pháp

**Test Pilot** tự động hóa toàn bộ quy trình qua 3 AI Agent phối hợp tuần tự:

**Agent 1 — Requirement Analyzer**
Nhận đầu vào là tài liệu requirement (text, PDF, DOCX, Markdown, Swagger/OpenAPI), trích xuất test scenario đầy đủ các loại: positive, negative, boundary, security, edge case. Đánh giá chất lượng đặc tả qua Quality Score (completeness, testability, clarity).

**Agent 2 — Test Case & Payload Generator**
Tự động chọn kỹ thuật kiểm thử phù hợp (EP, BVA, Decision Table, Error Guessing…) và sinh test case có cấu trúc đầy đủ: preconditions, steps, test data, expected result. Payload template tạo tự động từ test data.

**Agent 3 — Automation Code Writer**
Chuyển test case thành code automation sẵn sàng chạy, hỗ trợ 5 framework: pytest, Playwright, k6, Selenium, Postman Collection.

## Kết quả

- Test case có ID traceable: SCN-NNN → TC-TECHNIQUE-NNN
- Requirement Traceability Matrix (RTM) xuất Excel
- Automation code tích hợp được CI/CD ngay
- Tester chỉnh sửa scenario inline, thêm scenario thủ công trước khi generate test case

## Giá trị

Rút ngắn thời gian từ requirement đến automation code từ vài ngày xuống vài phút, đảm bảo coverage đồng đều và traceability rõ ràng.
