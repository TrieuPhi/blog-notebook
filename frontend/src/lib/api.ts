/**
 * API Client for interacting with the FastAPI Backend and Supabase
 */

export interface ApiPost {
  id?: string;
  slug: string;
  title: string;
  excerpt?: string;
  content?: string;
  date?: string;
  author?: string;
  tags?: string[];
  category?: string;
  series?: string;
  series_index?: number;
  pinned?: boolean;
  published?: boolean;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export async function fetchPosts(params?: { tag?: string; category?: string; search?: string }): Promise<ApiPost[]> {
  try {
    const query = new URLSearchParams();
    if (params?.tag) query.append("tag", params.tag);
    if (params?.category) query.append("category", params.category);
    if (params?.search) query.append("search", params.search);

    const res = await fetch(`${API_BASE_URL}/posts?${query.toString()}`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Failed to fetch posts from FastAPI backend, falling back to local posts.", err);
    return [];
  }
}

export async function fetchPostBySlug(slug: string): Promise<ApiPost | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/posts/${slug}`);
    if (!res.ok) {
      if (res.status === 404) return null;
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn(`Failed to fetch post '${slug}' from API:`, err);
    return null;
  }
}

