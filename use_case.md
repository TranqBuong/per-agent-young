# QE Agent — AI-Powered QE Automation

**Vấn đề giải quyết:**
QE thường mất nhiều giờ đọc requirement, phân tích edge case và viết test case thủ công — dễ bỏ sót, khó đảm bảo coverage đồng đều.

**Người dùng:**
QE Engineer, Developer, BA — những người cần biến requirement thành test case có cấu trúc mà không mất công phân tích từ đầu.

**Cách hoạt động:**
- ** 1 ** đọc requirement, đánh giá chất lượng (completeness, testability, clarity) và trích xuất scenarios theo từng loại: positive, negative, boundary, security, edge case.
- ** 2 ** áp dụng đồng thời nhiều kỹ thuật kiểm thử (EP, BVA, Decision Table, Error Guessing...) để sinh test cases chi tiết, sau đó AI tự review và loại bỏ các test case trùng nhau về mặt ngữ nghĩa.
- ** 3 ** generate automation code theo framework tùy chọn (pytest, Playwright, k6), có thể chạy trực tiếp trên giao diện.

**Giá trị mang lại:**
Từ một đoạn requirement, tạo ra bộ test case đa kỹ thuật, có cấu trúc, không trùng lặp và sẵn sàng tự động hóa — trong vài giây thay vì vài giờ.
