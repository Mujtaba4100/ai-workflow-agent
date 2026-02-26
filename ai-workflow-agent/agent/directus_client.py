"""
Milestone 3: Directus Integration
Auth, users, API, and storage for dashboard layouts
"""

import asyncio
import json
import logging
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


# ============================================================
# Enums and Data Classes
# ============================================================

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class CollectionType(str, Enum):
    """Directus collection types"""
    DASHBOARD_LAYOUTS = "dashboard_layouts"
    WORKFLOW_METADATA = "workflow_metadata"
    PROJECT_DATA = "project_data"
    USER_SETTINGS = "user_settings"
    ANALYSIS_RESULTS = "analysis_results"


@dataclass
class User:
    """A Directus user"""
    user_id: str
    email: str
    username: str
    role: UserRole
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.user_id,
            "email": self.email,
            "username": self.username,
            "role": self.role.value,
            "createdAt": self.created_at.isoformat(),
            "lastLogin": self.last_login.isoformat() if self.last_login else None,
            "isActive": self.is_active,
            "metadata": self.metadata
        }


@dataclass
class AuthToken:
    """Authentication token"""
    token: str
    user_id: str
    expires_at: datetime
    refresh_token: Optional[str] = None
    
    def is_valid(self) -> bool:
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accessToken": self.token,
            "userId": self.user_id,
            "expiresAt": self.expires_at.isoformat(),
            "refreshToken": self.refresh_token
        }


@dataclass
class StoredItem:
    """An item stored in Directus"""
    item_id: str
    collection: CollectionType
    data: Dict[str, Any]
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item_id,
            "collection": self.collection.value,
            "data": self.data,
            "createdBy": self.created_by,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


# ============================================================
# User Manager
# ============================================================

class UserManager:
    """Manage Directus users"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self._password_hashes: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin = User(
            user_id="admin_001",
            email="admin@localhost",
            username="admin",
            role=UserRole.ADMIN
        )
        self.users[admin.user_id] = admin
        self._password_hashes[admin.user_id] = self._hash_password("admin123")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.VIEWER
    ) -> User:
        """Create a new user"""
        # Check for duplicate email
        for user in self.users.values():
            if user.email == email:
                raise ValueError(f"Email already exists: {email}")
        
        user = User(
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            email=email,
            username=username,
            role=role
        )
        
        self.users[user.user_id] = user
        self._password_hashes[user.user_id] = self._hash_password(password)
        
        logger.info(f"User created: {user.user_id} ({email})")
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password"""
        for user in self.users.values():
            if user.email == email:
                if not user.is_active:
                    return None
                
                password_hash = self._hash_password(password)
                if self._password_hashes.get(user.user_id) == password_hash:
                    user.last_login = datetime.now()
                    return user
                return None
        return None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def list_users(self, role: Optional[UserRole] = None) -> List[User]:
        """List all users, optionally filtered by role"""
        users = list(self.users.values())
        if role:
            users = [u for u in users if u.role == role]
        return users
    
    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> Optional[User]:
        """Update user properties"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        if username:
            user.username = username
        if role:
            user.role = role
        if is_active is not None:
            user.is_active = is_active
        
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        if user_id in self.users:
            del self.users[user_id]
            if user_id in self._password_hashes:
                del self._password_hashes[user_id]
            return True
        return False


# ============================================================
# Auth Manager
# ============================================================

class AuthManager:
    """Manage authentication tokens"""
    
    TOKEN_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 7
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.tokens: Dict[str, AuthToken] = {}
        self.refresh_tokens: Dict[str, str] = {}  # refresh_token -> user_id
    
    def login(self, email: str, password: str) -> Optional[AuthToken]:
        """Login and get auth token"""
        user = self.user_manager.authenticate(email, password)
        if not user:
            return None
        
        # Generate tokens
        token = self._generate_token()
        refresh_token = self._generate_token()
        
        auth_token = AuthToken(
            token=token,
            user_id=user.user_id,
            expires_at=datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS),
            refresh_token=refresh_token
        )
        
        self.tokens[token] = auth_token
        self.refresh_tokens[refresh_token] = user.user_id
        
        logger.info(f"User logged in: {user.email}")
        return auth_token
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate a token and return the user"""
        auth_token = self.tokens.get(token)
        if not auth_token or not auth_token.is_valid():
            return None
        
        return self.user_manager.get_user(auth_token.user_id)
    
    def refresh(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh an auth token"""
        user_id = self.refresh_tokens.get(refresh_token)
        if not user_id:
            return None
        
        user = self.user_manager.get_user(user_id)
        if not user or not user.is_active:
            return None
        
        # Generate new tokens
        token = self._generate_token()
        new_refresh_token = self._generate_token()
        
        auth_token = AuthToken(
            token=token,
            user_id=user.user_id,
            expires_at=datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS),
            refresh_token=new_refresh_token
        )
        
        # Remove old refresh token
        del self.refresh_tokens[refresh_token]
        
        self.tokens[token] = auth_token
        self.refresh_tokens[new_refresh_token] = user_id
        
        return auth_token
    
    def logout(self, token: str) -> bool:
        """Logout and invalidate token"""
        auth_token = self.tokens.get(token)
        if auth_token:
            del self.tokens[token]
            if auth_token.refresh_token in self.refresh_tokens:
                del self.refresh_tokens[auth_token.refresh_token]
            return True
        return False
    
    def _generate_token(self) -> str:
        """Generate a random token"""
        return uuid.uuid4().hex + uuid.uuid4().hex


# ============================================================
# Storage Manager
# ============================================================

class StorageManager:
    """Manage Directus collections and items"""
    
    def __init__(self):
        self.collections: Dict[CollectionType, Dict[str, StoredItem]] = {
            ct: {} for ct in CollectionType
        }
    
    def create_item(
        self,
        collection: CollectionType,
        data: Dict[str, Any],
        created_by: str
    ) -> StoredItem:
        """Create a new item in a collection"""
        item = StoredItem(
            item_id=f"item_{uuid.uuid4().hex[:8]}",
            collection=collection,
            data=data,
            created_by=created_by
        )
        
        self.collections[collection][item.item_id] = item
        logger.info(f"Item created: {item.item_id} in {collection.value}")
        return item
    
    def get_item(self, collection: CollectionType, item_id: str) -> Optional[StoredItem]:
        """Get an item by ID"""
        return self.collections[collection].get(item_id)
    
    def update_item(
        self,
        collection: CollectionType,
        item_id: str,
        data: Dict[str, Any]
    ) -> Optional[StoredItem]:
        """Update an item's data"""
        item = self.collections[collection].get(item_id)
        if item:
            item.data.update(data)
            item.updated_at = datetime.now()
            return item
        return None
    
    def delete_item(self, collection: CollectionType, item_id: str) -> bool:
        """Delete an item"""
        if item_id in self.collections[collection]:
            del self.collections[collection][item_id]
            return True
        return False
    
    def list_items(
        self,
        collection: CollectionType,
        created_by: Optional[str] = None,
        limit: int = 100
    ) -> List[StoredItem]:
        """List items in a collection"""
        items = list(self.collections[collection].values())
        
        if created_by:
            items = [i for i in items if i.created_by == created_by]
        
        # Sort by created_at descending
        items.sort(key=lambda x: x.created_at, reverse=True)
        
        return items[:limit]
    
    def search_items(
        self,
        collection: CollectionType,
        field: str,
        value: Any
    ) -> List[StoredItem]:
        """Search items by field value"""
        results = []
        for item in self.collections[collection].values():
            if field in item.data and item.data[field] == value:
                results.append(item)
        return results
    
    def get_collection_stats(self, collection: CollectionType) -> Dict[str, Any]:
        """Get statistics for a collection"""
        items = list(self.collections[collection].values())
        
        return {
            "collection": collection.value,
            "itemCount": len(items),
            "latestUpdate": max((i.updated_at for i in items), default=None)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        total = 0
        
        for collection in CollectionType:
            count = len(self.collections[collection])
            stats[collection.value] = count
            total += count
        
        return {
            "collections": stats,
            "totalItems": total
        }


# ============================================================
# Directus Client
# ============================================================

class DirectusClient:
    """
    Client for interacting with Directus.
    Provides auth, users, API, and storage functionality.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8055",
        admin_email: Optional[str] = None,
        admin_password: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.user_manager = UserManager()
        self.auth_manager = AuthManager(self.user_manager)
        self.storage = StorageManager()
        self._connected = False
        self._current_token: Optional[AuthToken] = None
    
    async def connect(self) -> bool:
        """Test connection to Directus server"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/server/ping",
                    timeout=5.0
                )
                self._connected = response.status_code == 200
                return self._connected
        except Exception as e:
            logger.warning(f"Directus connection failed: {e}")
            # Use local storage mode
            self._connected = False
            return True  # Still functional with local storage
    
    def is_connected(self) -> bool:
        """Check if connected to Directus"""
        return self._connected
    
    # --- Auth Methods ---
    
    def login(self, email: str, password: str) -> Optional[AuthToken]:
        """Login user"""
        token = self.auth_manager.login(email, password)
        if token:
            self._current_token = token
        return token
    
    def logout(self) -> bool:
        """Logout current user"""
        if self._current_token:
            result = self.auth_manager.logout(self._current_token.token)
            self._current_token = None
            return result
        return False
    
    def get_current_user(self) -> Optional[User]:
        """Get currently logged in user"""
        if self._current_token:
            return self.auth_manager.validate_token(self._current_token.token)
        return None
    
    def refresh_token(self) -> Optional[AuthToken]:
        """Refresh current token"""
        if self._current_token and self._current_token.refresh_token:
            new_token = self.auth_manager.refresh(self._current_token.refresh_token)
            if new_token:
                self._current_token = new_token
            return new_token
        return None
    
    # --- User Methods ---
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.VIEWER
    ) -> User:
        """Create a new user"""
        return self.user_manager.create_user(email, username, password, role)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.user_manager.get_user(user_id)
    
    def list_users(self, role: Optional[UserRole] = None) -> List[User]:
        """List all users"""
        return self.user_manager.list_users(role)
    
    # --- Storage Methods ---
    
    def save_dashboard_layout(
        self,
        name: str,
        layout_json: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> StoredItem:
        """Save a dashboard layout"""
        creator = user_id or (self._current_token.user_id if self._current_token else "anonymous")
        
        data = {
            "name": name,
            "layout": layout_json,
            "savedAt": datetime.now().isoformat()
        }
        
        return self.storage.create_item(
            CollectionType.DASHBOARD_LAYOUTS,
            data,
            creator
        )
    
    def get_dashboard_layout(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a dashboard layout by ID"""
        item = self.storage.get_item(CollectionType.DASHBOARD_LAYOUTS, item_id)
        return item.data.get("layout") if item else None
    
    def list_dashboard_layouts(self, user_id: Optional[str] = None) -> List[StoredItem]:
        """List all dashboard layouts"""
        return self.storage.list_items(CollectionType.DASHBOARD_LAYOUTS, user_id)
    
    def save_workflow_metadata(
        self,
        workflow_id: str,
        metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> StoredItem:
        """Save workflow metadata"""
        creator = user_id or (self._current_token.user_id if self._current_token else "anonymous")
        
        data = {
            "workflowId": workflow_id,
            "metadata": metadata,
            "savedAt": datetime.now().isoformat()
        }
        
        return self.storage.create_item(
            CollectionType.WORKFLOW_METADATA,
            data,
            creator
        )
    
    def save_project_data(
        self,
        project_name: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> StoredItem:
        """Save project data"""
        creator = user_id or (self._current_token.user_id if self._current_token else "anonymous")
        
        project_data = {
            "projectName": project_name,
            "data": data,
            "savedAt": datetime.now().isoformat()
        }
        
        return self.storage.create_item(
            CollectionType.PROJECT_DATA,
            project_data,
            creator
        )
    
    def save_analysis_result(
        self,
        url: str,
        analysis: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> StoredItem:
        """Save a page analysis result"""
        creator = user_id or (self._current_token.user_id if self._current_token else "anonymous")
        
        data = {
            "url": url,
            "analysis": analysis,
            "analyzedAt": datetime.now().isoformat()
        }
        
        return self.storage.create_item(
            CollectionType.ANALYSIS_RESULTS,
            data,
            creator
        )
    
    def get_analysis_results(self, url: Optional[str] = None) -> List[StoredItem]:
        """Get analysis results, optionally filtered by URL"""
        if url:
            return self.storage.search_items(
                CollectionType.ANALYSIS_RESULTS,
                "url",
                url
            )
        return self.storage.list_items(CollectionType.ANALYSIS_RESULTS)
    
    # --- API Methods ---
    
    def get_collections(self) -> List[str]:
        """Get list of available collections"""
        return [ct.value for ct in CollectionType]
    
    def get_collection_items(
        self,
        collection: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get items from a collection"""
        try:
            ct = CollectionType(collection)
            items = self.storage.list_items(ct, limit=limit)
            return [i.to_dict() for i in items]
        except ValueError:
            return []
    
    def create_collection_item(
        self,
        collection: str,
        data: Dict[str, Any]
    ) -> Optional[StoredItem]:
        """Create an item in any collection"""
        try:
            ct = CollectionType(collection)
            user_id = self._current_token.user_id if self._current_token else "anonymous"
            return self.storage.create_item(ct, data, user_id)
        except ValueError:
            return None
    
    def delete_collection_item(self, collection: str, item_id: str) -> bool:
        """Delete an item from a collection"""
        try:
            ct = CollectionType(collection)
            return self.storage.delete_item(ct, item_id)
        except ValueError:
            return False
    
    # --- Statistics ---
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        storage_stats = self.storage.get_all_stats()
        
        return {
            "connected": self._connected,
            "baseUrl": self.base_url,
            "users": {
                "total": len(self.user_manager.users),
                "admins": len([u for u in self.user_manager.users.values() 
                              if u.role == UserRole.ADMIN]),
                "active": len([u for u in self.user_manager.users.values() 
                              if u.is_active])
            },
            "storage": storage_stats,
            "activeSessions": len(self.auth_manager.tokens)
        }


# ============================================================
# Singleton Instance
# ============================================================

_directus_client: Optional[DirectusClient] = None


def get_directus_client(
    base_url: str = "http://localhost:8055"
) -> DirectusClient:
    """Get or create the Directus client singleton"""
    global _directus_client
    if _directus_client is None:
        _directus_client = DirectusClient(base_url)
    return _directus_client
