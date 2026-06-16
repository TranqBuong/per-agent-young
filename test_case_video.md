API: Chuyển tiền Zalopay (POST /v1/transfer)

Mô tả:
Người dùng chuyển tiền từ ví Zalopay của mình sang ví Zalopay của người khác
thông qua số điện thoại.

Đầu vào:
- sender_id (string, required): ID người gửi
- receiver_phone (string, required): Số điện thoại người nhận (10 số, bắt đầu 0)
- amount (integer, required): Số tiền chuyển, đơn vị VND
  - Tối thiểu: 1.000 VND
  - Tối đa: 100.000.000 VND / giao dịch
  - Tối đa: 200.000.000 VND / ngày
- note (string, optional): Lời nhắn, tối đa 50 ký tự
- otp (string, required): Mã OTP 6 số, hết hạn sau 60 giây

Đầu ra thành công (200):
- transaction_id: mã giao dịch duy nhất
- status: "SUCCESS"
- new_balance: số dư ví người gửi sau giao dịch

Lỗi cần xử lý:
- Số dư không đủ → 400 INSUFFICIENT_BALANCE
- Số điện thoại không tồn tại trên Zalopay → 404 RECEIVER_NOT_FOUND
- OTP sai hoặc hết hạn → 401 INVALID_OTP
- Vượt hạn mức ngày → 400 DAILY_LIMIT_EXCEEDED
- Tài khoản bị khóa → 403 ACCOUNT_SUSPENDED


Tại sao requirement này demo tốt:

Agent 1 sẽ sinh ~8-10 scenario đa dạng (positive, negative, boundary, security)
Agent 2 sẽ dùng nhiều kỹ thuật: EP (phân vùng amount), BVA (1.000 / 100M / 200M), Error Guessing (OTP expired)
Agent 3 sinh code pytest với test data cụ thể, trông chuyên nghiệp
Toàn bộ pipeline chạy khoảng 30-60 giây — vừa đủ cho video
