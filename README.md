# Setlog Notebook (Fullstack Personal Blog)

Hệ thống blog cá nhân với kiến trúc phân tách rõ ràng (**Fullstack Decoupled**):
- **Frontend (`/frontend`)**: React + Vite + Tailwind CSS + TypeScript + KaTeX + Mermaid.
- **Backend (`/backend`)**: Python FastAPI REST API.
- **Database**: PostgreSQL trên **Supabase Cloud**.
- **CI/CD**: GitHub Actions tự động deploy Frontend lên **GitHub Pages** và đồng bộ bài viết Markdown lên **Supabase**.

---

## 📁 Cấu trúc dự án

```text
blog-notebook/
├── frontend/                   # Ứng dụng giao diện blog (React + Vite)
│   ├── posts/                  # Các bài viết Markdown (.md)
│   ├── src/                    # Mã nguồn React
│   └── package.json
│
├── backend/                    # API Backend (FastAPI)
│   ├── app/                    # Mã nguồn FastAPI
│   ├── sync_posts.py           # Script tự động đồng bộ bài viết lên Supabase
│   ├── schema.sql              # SQL Schema khởi tạo bảng trên Supabase
│   ├── requirements.txt        # Thư viện Python
│   └── run.py                  # Entrypoint chạy server local
│
├── .github/
│   └── workflows/
│       ├── deploy-frontend.yml # Tự động deploy Frontend lên GitHub Pages
│       └── sync-posts.yml      # Tự động sync bài viết lên Supabase khi push
│
└── .gitignore                  # Chặn file rác và chặn lộ key bí mật
```

---

## 💻 Hướng dẫn chạy ở Local

### 1. Khởi động Backend (FastAPI)
```powershell
conda activate blog
cd backend
python run.py
```
> API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Khởi động Frontend (React)
```powershell
cd frontend
npm install
npm run dev
```
> Web UI: [http://localhost:5173](http://localhost:5173)

### 3. Đồng bộ bài viết Markdown vào Database
```powershell
conda activate blog
cd backend
python sync_posts.py
```

