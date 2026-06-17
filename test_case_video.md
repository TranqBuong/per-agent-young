API: Chuyển tiền ZaloPay (POST /v1/transfer)

sender_id (string)
receiver_phone (string, 10 số bắt đầu 0)
amount (integer, 1.000–100.000.000 VND, tối đa 200.000.000/ngày)
note (string tùy chọn, ≤50 ký tự)
otp (string, 6 số, hết hạn 60s)

Thành công (200): transaction_id, status "SUCCESS", new_balance

Lỗi: 400 INSUFFICIENT_BALANCE, 
404 RECEIVER_NOT_FOUND, 
401 INVALID_OTP, 
400 DAILY_LIMIT_EXCEEDED, 
403 ACCOUNT_SUSPENDED
