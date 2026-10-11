from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PostBase(BaseModel):
    title: str = Field(..., example="Giới thiệu về FastAPI")
    slug: str = Field(..., example="gioi-thieu-ve-fastapi")
    excerpt: Optional[str] = Field("", example="Tổng quan về framework FastAPI trong Python")
    content: str = Field(..., example="# FastAPI\n\nNội dung bài viết...")
    tags: List[str] = Field(default_factory=list, example=["python", "fastapi"])
    category: Optional[str] = Field("general", example="Tech")
    series: Optional[str] = None
    series_index: Optional[int] = None
    pinned: bool = False
    published: bool = True
    author: Optional[str] = "Phi Trieu Dai"


class PostCreate(PostBase):
    pass


class PostUpdate(BaseModel):
    title: Optional[str] = None
    excerpt: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    series: Optional[str] = None
    series_index: Optional[int] = None
    pinned: Optional[bool] = None
    published: Optional[bool] = None
    author: Optional[str] = None


class PostResponse(PostBase):
    id: Optional[str] = None
    date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PostListItem(BaseModel):
    id: Optional[str] = None
    slug: str
    title: str
    excerpt: Optional[str] = ""
    date: Optional[datetime] = None
    tags: List[str] = []
    category: Optional[str] = "general"
    series: Optional[str] = None
    pinned: bool = False
    author: Optional[str] = "Phi Trieu Dai"

