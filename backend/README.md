# Blog Notebook - Backend (FastAPI + Supabase)

Backend REST API được xây dựng bằng Python **FastAPI**, tích hợp database **Supabase (PostgreSQL)** để quản lý bài viết và dữ liệu blog.

---

## 🚀 Cách chạy Backend ở Local

### 1. Kích hoạt môi trường Conda
Mở terminal và kích hoạt môi trường conda `blog`:
```powershell
conda activate blog
```

### 2. Di chuyển vào thư mục backend và chạy server
```powershell
cd c:\project\blog\blog-notebook\backend
python run.py
```
Hoặc dùng trực tiếp lệnh `uvicorn`:
```powershell
uvicorn app.main:app --reload --port 8000
```

---

## 🌐 Các Endpoint & Tài liệu API

Khi server đang chạy:
- **Swagger UI (Interactive API Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Danh sách API chính:
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/api/posts` | Lấy danh sách bài viết (hỗ trợ filter `tag`, `category`, `search`) |
| `GET` | `/api/posts/{slug}` | Lấy chi tiết bài viết (bao gồm nội dung Markdown) |
| `POST` | `/api/posts` | Tạo bài viết mới (yêu cầu header `X-Api-Key`) |
| `PUT` | `/api/posts/{slug}` | Cập nhật bài viết (yêu cầu header `X-Api-Key`) |
| `DELETE` | `/api/posts/{slug}` | Xoá bài viết (yêu cầu header `X-Api-Key`) |

---

## 🗄️ Kết nối Supabase Cloud

Mặc định, server sẽ tự chạy ở chế độ **Mock In-Memory** để bạn có thể test ngay mà không bị crash nếu chưa có tài khoản Supabase.

Khi bạn đã tạo Project trên [Supabase](https://supabase.com):
1. Mở **SQL Editor** trên Supabase Dashboard và dán toàn bộ nội dung file [schema.sql](file:///c:/project/blog/backend/schema.sql) vào để tạo bảng `posts`.
2. Lấy **Project URL** và **anon/public API key** trong phần *Project Settings -> API*.
3. Điền vào file [backend/.env](file:///c:/project/blog/backend/.env):
   ```ini
   SUPABASE_URL=https://xxxxxxxx.supabase.co
   SUPABASE_KEY=your-supabase-anon-key
   ADMIN_API_KEY=dev_secret_key_123
   ```
4. Khởi động lại backend, server sẽ tự động chuyển sang đọc/ghi dữ liệu thật từ Supabase.

