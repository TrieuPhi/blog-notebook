from fastapi import Header, HTTPException, status
from app.core.config import settings
from typing import Optional


def verify_admin_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verifies that the request comes from an authorized admin for mutating operations."""
    if not settings.ADMIN_API_KEY:
        return True
    if x_api_key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Admin API Key (X-Api-Key header required)"
        )
    return True

