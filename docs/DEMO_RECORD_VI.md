# KỊCH BẢN QUAY VIDEO DEMO WEBSITE (Tiếng Việt) — ~75 giây

## Chuẩn bị (trước khi quay)
```bash
# 1. Bật model nền (daemon) + web — đợi ~1 phút cho daemon tóm tắt xong
pgrep -f scripts.live_daemon || nohup .venv/bin/python -m scripts.live_daemon --backend 7b --poll 60 > data/live/daemon.log 2>&1 &
pgrep -f "streamlit run" || nohup .venv/bin/streamlit run webapp/app.py > /tmp/streamlit.log 2>&1 &
curl -sI http://localhost:8501 | head -1     # mong đợi 200
```
- Mở **http://localhost:8501** ở trình duyệt, **F11** toàn màn hình, phóng to 100–110%.
- **Công cụ quay:** Windows **Win + Alt + R** (Game Bar), hoặc OBS / Loom. Bật mic nếu thuyết minh.

---

## Kịch bản theo mốc thời gian (hành động · lời thoại)

**0:00–0:08 — Mở đầu** *(ở tab 🔴 Live & Advisory)*
> "Đây là hệ thống dự đoán rủi ro crash cổ phiếu: một LLM đọc tin tài chính và cảnh báo. Trang web có 3 phần."

**0:08–0:30 — Daily advisory** *(trỏ chuột vào gauge + mức Risk)*
> "Đây là dự đoán chính thức: mô hình Qwen 32B chạy mỗi ngày, đưa ra **xác suất crash trong 3 ngày tới**, mức rủi ro, và các cổ phiếu rủi ro nhất kèm lý do."

**0:30–0:48 — Live monitor + feed** *(kéo chuột xuống chậm)*
> "Phần này tóm tắt **tin trực tiếp bằng LLM 7B** chạy nền — cập nhật theo dòng tin. Bên dưới là feed 50 tin mới nhất; bấm **Show 50 more** để xem thêm."
*(bấm thử nút Show 50 more 1 lần)*

**0:48–1:05 — Research & Backtest** *(click tab 📊)*
> "Đây là số liệu nghiêm túc trên dữ liệu lịch sử có nhãn: **AUROC cửa sổ COVID 0.785, có RAG 0.847**, kèm biểu đồ. Bấm **Play** xem lại diễn biến rủi ro theo ngày."
*(bấm Play cho đường crash chạy vài giây)*

**1:05–1:15 — How it works + chốt** *(click tab ℹ️ rồi dừng)*
> "Phương pháp 4 pha TRR + RAG, mô hình zero-shot. Tóm lại: dữ liệu lớn → RAG → LLM → triển khai trực tiếp và backtest có nhãn. Cảm ơn thầy/cô."

---

## Mẹo quay
- Di chuột **chậm**, dừng 1–2 giây ở mỗi con số quan trọng.
- Quay **1 lần liền mạch**; nếu vấp, dừng 3 giây rồi nói lại câu đó (cắt sau).
- Nếu mạng yếu: số liệu vẫn hiện vì web đọc từ file daemon (không chờ mạng).
- Độ dài lý tưởng: **60–90 giây**. Xuất 1080p.
```
