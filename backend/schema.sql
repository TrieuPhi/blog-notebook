-- SQL Schema for Blog posts in Supabase
-- Run this in your Supabase SQL Editor: https://supabase.com/dashboard/project/_/sql

CREATE TABLE IF NOT EXISTS public.posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    excerpt TEXT DEFAULT '',
    content TEXT NOT NULL,
    date TIMESTAMPTZ DEFAULT NOW(),
    author TEXT DEFAULT 'Phi Trieu Dai',
    tags TEXT[] DEFAULT '{}',
    category TEXT DEFAULT 'general',
    series TEXT,
    series_index INT,
    pinned BOOLEAN DEFAULT FALSE,
    published BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying
CREATE INDEX IF NOT EXISTS idx_posts_slug ON public.posts(slug);
CREATE INDEX IF NOT EXISTS idx_posts_published ON public.posts(published);
CREATE INDEX IF NOT EXISTS idx_posts_date ON public.posts(date DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE public.posts ENABLE ROW LEVEL SECURITY;

-- Allow public read access to published posts
CREATE POLICY "Public can view published posts"
ON public.posts
FOR SELECT
USING (published = true);

-- Insert sample post
INSERT INTO public.posts (slug, title, excerpt, content, tags, category, pinned)
VALUES (
    'welcome-to-my-blog',
    'Chào mừng đến với Blog cá nhân!',
    'Bài viết đầu tiên được lưu trữ và phân phối qua FastAPI & Supabase.',
    '# Chào mừng!\n\nĐây là bài viết mẫu được tải trực tiếp từ **Supabase** thông qua backend **FastAPI**.\n\n```python\nprint("Hello World from FastAPI + Supabase!")\n```\n\n- [x] Backend FastAPI\n- [x] Database Supabase\n- [x] Frontend React + Vite',
    ARRAY['welcome', 'fastapi', 'supabase'],
    'Tech',
    TRUE
)
ON CONFLICT (slug) DO NOTHING;

