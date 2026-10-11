import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.core.config import settings

# Global client
_supabase_client = None

# Sample fallback posts when Supabase credentials are not yet set
MOCK_POSTS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "slug": "welcome-to-fastapi-supabase",
        "title": "Khởi động Backend FastAPI kết nối Supabase",
        "excerpt": "Hướng dẫn thiết lập FastAPI và kết nối Database Supabase cho Blog.",
        "content": "# Chào mừng bạn đến với Blog API!\n\nBackend FastAPI đang hoạt động ở chế độ Local Mock (hoặc kết nối Supabase).\n\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI(title='Blog API')\n```\n\n### Tính năng:\n- [x] RESTful API cho Posts\n- [x] Hỗ trợ Markdown, Katex, Mermaid\n- [x] Tích hợp Supabase Database\n- [x] CORS sẵn sàng cho Frontend",
        "date": datetime.now().isoformat(),
        "tags": ["fastapi", "python", "supabase", "react"],
        "category": "Backend",
        "series": "Setup Fullstack Blog",
        "series_index": 1,
        "pinned": True,
        "published": True,
        "author": "Phi Trieu Dai",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
]


def get_supabase_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = settings.SUPABASE_URL.strip()
    key = settings.SUPABASE_KEY.strip()

    if url and key and not url.startswith("https://your-project-ref"):
        # Auto clean URL if copied with /rest/v1 or trailing slash
        clean_url = url.split("/rest/v1")[0].rstrip("/")
        try:
            from supabase import create_client, Client
            _supabase_client = create_client(clean_url, key)
            print(" Connected to Supabase Cloud successfully!")
            return _supabase_client
        except Exception as e:
            print(f"⚠️ Failed to initialize Supabase client: {e}")
            return None
    return None

