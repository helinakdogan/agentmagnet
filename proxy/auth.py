import os
import hashlib
from fastapi import Security, HTTPException, status
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from typing import Union
from supabase import create_client, Client

# auto_error=False allows us to return a custom 401 instead of FastAPI's default 403
security = HTTPBearer(auto_error=False)

_supabase_client: Union[Client, None] = None

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials are not configured on the server.")
        _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected 'Bearer mg_sk_...'"
        )
        
    supabase = get_supabase_client()
    
    raw_key = credentials.credentials

    if not raw_key.startswith("mg_sk_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format. Must start with 'mg_sk_'"
        )

    # SHA-256 Hash the raw key to match the database stored key_hash
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    try:
        response = supabase.table("api_keys").select("id, project_id, project_uuid").eq("key_hash", key_hash).eq("is_active", True).execute()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")

    if not response.data or len(response.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key."
        )

    row = response.data[0]
    # Use project_uuid (UUID type) as the canonical identifier; fall back to text project_id.
    canonical_project_id = row.get("project_uuid") or row.get("project_id")

    return {
        "api_key_id": row["id"],
        "project_id": canonical_project_id,
        "openai_api_key": None,  # Reserved for future BYOK: add openai_api_key column to api_keys table
    }