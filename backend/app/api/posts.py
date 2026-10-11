from fastapi import APIRouter, HTTPException, Depends, Query, status
from typing import List, Optional
from datetime import datetime
import uuid

from app.schemas.post import PostCreate, PostUpdate, PostResponse, PostListItem
from app.db.supabase import get_supabase_client, MOCK_POSTS
from app.api.deps import verify_admin_key

router = APIRouter(prefix="/posts", tags=["Posts"])


@router.get("", response_model=List[PostListItem])
def list_posts(
    tag: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Retrieve list of published posts with filtering options."""
    client = get_supabase_client()
    if client:
        try:
            query = client.table("posts").select("id, slug, title, excerpt, date, tags, category, series, pinned, author").eq("published", True)
            if tag:
                query = query.contains("tags", [tag])
            if category:
                query = query.eq("category", category)
            if search:
                query = query.or_(f"title.ilike.%{search}%,excerpt.ilike.%{search}%")
            
            res = query.order("pinned", desc=True).order("date", desc=True).range(offset, offset + limit - 1).execute()
            return res.data
        except Exception as e:
            print(f"Supabase query error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback to local memory mock
    results = [p for p in MOCK_POSTS if p.get("published", True)]
    if tag:
        results = [p for p in results if tag in p.get("tags", [])]
    if category:
        results = [p for p in results if p.get("category") == category]
    if search:
        s = search.lower()
        results = [p for p in results if s in p.get("title", "").lower() or s in p.get("excerpt", "").lower()]

    results.sort(key=lambda x: (not x.get("pinned", False), x.get("date", "")), reverse=True)
    return results[offset : offset + limit]


@router.get("/{slug}", response_model=PostResponse)
def get_post(slug: str):
    """Get full post details including Markdown content by slug."""
    client = get_supabase_client()
    if client:
        try:
            res = client.table("posts").select("*").eq("slug", slug).eq("published", True).execute()
            if not res.data or len(res.data) == 0:
                raise HTTPException(status_code=404, detail="Post not found")
            return res.data[0]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    for p in MOCK_POSTS:
        if p.get("slug") == slug:
            return p
    raise HTTPException(status_code=404, detail="Post not found")


@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreate, _=Depends(verify_admin_key)):
    """Create a new post in Supabase (Protected by Admin API Key)."""
    client = get_supabase_client()
    post_data = post.dict()
    now_iso = datetime.now().isoformat()
    post_data["created_at"] = now_iso
    post_data["updated_at"] = now_iso
    post_data["date"] = now_iso

    if client:
        try:
            res = client.table("posts").insert(post_data).execute()
            if res.data:
                return res.data[0]
            raise HTTPException(status_code=400, detail="Failed to create post")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Mock mode
    post_data["id"] = str(uuid.uuid4())
    MOCK_POSTS.insert(0, post_data)
    return post_data


@router.put("/{slug}", response_model=PostResponse)
def update_post(slug: str, update_data: PostUpdate, _=Depends(verify_admin_key)):
    """Update an existing post by slug (Protected by Admin API Key)."""
    payload = {k: v for k, v in update_data.dict().items() if v is not None}
    payload["updated_at"] = datetime.now().isoformat()

    client = get_supabase_client()
    if client:
        try:
            res = client.table("posts").update(payload).eq("slug", slug).execute()
            if not res.data:
                raise HTTPException(status_code=404, detail="Post not found")
            return res.data[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    for idx, p in enumerate(MOCK_POSTS):
        if p.get("slug") == slug:
            MOCK_POSTS[idx].update(payload)
            return MOCK_POSTS[idx]
    raise HTTPException(status_code=404, detail="Post not found")


@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(slug: str, _=Depends(verify_admin_key)):
    """Delete a post by slug (Protected by Admin API Key)."""
    client = get_supabase_client()
    if client:
        try:
            client.table("posts").delete().eq("slug", slug).execute()
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    global MOCK_POSTS
    MOCK_POSTS = [p for p in MOCK_POSTS if p.get("slug") != slug]
    return None

