from dataclasses import Field
from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
# schemas.py
from datetime import datetime
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List




Role = Literal["user", "admin", "responder"]


# Conversations
class ConversationCreate(BaseModel):
    subject: Optional[str] = None

class ConversationOut(BaseModel):
    id: int
    user_id: int
    admin_id: Optional[int]
    subject: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None
    unread_count: int = 0

# Messages
class MessageCreate(BaseModel):
    content: str

class MessageOut(BaseModel):
    id: int
    conversation_id: int
    sender_id: int
    sender_name: str
    sender_role: str
    content: str
    is_read: bool
    created_at: datetime
class RegisterIn(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    role: Role = "user"
    contact_number: str | None = None
    barangay: str | None = None
    address: str | None = None
    emergency_contact_name: str | None = None
    emergency_contact_number: str | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: str
    status: Optional[str] = "active"  # Add this
    contact_number: str | None
    barangay: str | None
    address: str | None
    profile_photo: str | None = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None   # <-- add this
    is_online: Optional[bool] = False  

    class Config:
        from_attributes = True

class UserProfileOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    contact_number: Optional[str]
    barangay: Optional[str]
    address: Optional[str]
    emergency_contact_name: Optional[str]
    emergency_contact_number: Optional[str]
    profile_photo: str | None = None

    last_active: Optional[datetime] = None   # <-- add this
    is_online: Optional[bool] = False  

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    full_name: Optional[str]
    contact_number: Optional[str]
    barangay: Optional[str]
    address: Optional[str]
    emergency_contact_name: Optional[str]
    emergency_contact_number: Optional[str]

# Add these classes to your schemas.py
class ChatMessageBase(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessageBase]] = None

class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    timestamp: str
    quick_replies: Optional[List[str]] = None

class ChatHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: str

# Incident Reporting Schemas
class IncidentReportCreate(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact: Optional[str] = None

class IncidentReportUpdate(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[int] = None
    notes: Optional[str] = None

from pydantic import validator
import json

class IncidentReportResponse(BaseModel):
    id: str
    description: str
    incident_type: str
    severity: str
    priority: int
    latitude: float
    longitude: float
    barangay: Optional[str]
    address: Optional[str]
    status: str
    ml_confidence: float
    keywords: List[str]               # ← remains List[str]
    assigned_to: Optional[int] = None   # <-- ADD THIS
    created_at: datetime
    updated_at: Optional[datetime]
    image_paths: Optional[List[str]] = []  # Add this
    video_paths: Optional[List[str]] = []  # Add this
    text_analysis: Optional[Dict] = None   # Add this
    resolved_at: Optional[datetime] = None

    @validator('keywords', pre=True)
    def parse_keywords(cls, v):
        """Convert a JSON string to a list if necessary."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []              # fallback to empty list
        return v
    @validator('image_paths', pre=True)
    def parse_image_paths(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v or []

    @validator('video_paths', pre=True)
    def parse_video_paths(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v or []
    
    @validator('text_analysis', pre=True)
    def parse_text_analysis(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return None
        return v
    class Config:
        from_attributes = True

class MLTextAnalysisResponse(BaseModel):
    type: str
    confidence: float
    keywords: List[str]
    all_predictions: List[float]

class MLReportAnalysisResponse(BaseModel):
    success: bool
    report_id: str
    incident_type: str
    severity: str
    priority: int
    confidence: float
    analysis: Dict[str, Any]
    location: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

class IncidentUpdateCreate(BaseModel):
    update_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class IncidentUpdateResponse(BaseModel):
    id: int
    update_type: str
    content: str
    user_name: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class StatisticsResponse(BaseModel):
    total: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_barangay: Dict[str, int]
    time_period: str

    class UserRoleUpdate(BaseModel):
        role: str  # admin, responder, citizen

class UserStatusUpdate(BaseModel):
    status: str  # active, inactive, suspended

class UserAdminUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None

class AdminDashboardStats(BaseModel):
    total_incidents: int
    pending: int
    in_progress: int
    resolved: int
    total_users: int
    citizens: int
    responders: int
    administrators: int
    recent_incidents: list

class UserListResponse(BaseModel):
    users: List[UserOut]
    total: int
    page: int
    limit: int
    total_pages: int

# Add Status enum
class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

# Add TMO role to Role
Role = Literal["user", "admin", "responder", "tmo"]


# Update UserProfileOut to include status
class UserProfileOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    status: Optional[str] = "active"  # Add this
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None
    profile_photo: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Admin-specific schemas
class UserCreateAdmin(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Role = "user"
    status: Optional[str] = "active"
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None

class UserRoleUpdate(BaseModel):
    role: Role

class UserStatusUpdate(BaseModel):
    status: str

class UserAdminUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None

class AdminDashboardStats(BaseModel):
    total_incidents: int
    pending: int
    in_progress: int
    resolved: int
    total_users: int
    citizens: int
    responders: int
    tmoOfficers: int
    administrators: int
    recent_incidents: List[Dict[str, Any]]

class UserListResponse(BaseModel):
    users: List[UserOut]
    total: int
    page: int
    limit: int
    total_pages: int

class UserAdminCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Role = "user"
    status: str = "active"
    contact_number: Optional[str] = None
    barangay: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None

    class Config:
        from_attributes = True

class AnonymousEmergencyResponse(BaseModel):
    id: int
    latitude: float
    longitude: float
    audio_url: str          # URL to access the file
    timestamp: datetime

    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    message: str
    severity: str = "medium"
    target_zone: Optional[str] = Field(None, alias="targetZone")
    target_roles: Optional[List[str]] = Field([], alias="targetRoles")
    geometry: Optional[Any] = None
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")
    image_url: Optional[str] = None
    
    class Config:
        populate_by_name = True
    
    

class AlertResponse(BaseModel):
    id: int
    message: str
    severity: str
    target_zone: Optional[str]
    target_roles: List[str]
    geometry: Optional[Dict[str, Any]]
    image_url: Optional[str] = None   # <-- Add this line
    created_at: datetime
    expires_at: Optional[datetime] = None

# ========== ML Analytics Schemas ==========

class MLAnalyticsResponse(BaseModel):
    user_id: Optional[int] = None
    total_predictions: int = 0
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    avg_confidence: float = 0.0
    trend_data: List[tuple] = []
    days_analyzed: int = 30
    top_predictions: Dict[str, int] = {}

class ModelPerformanceResponse(BaseModel):
    latest_accuracy: float = 0.0
    best_accuracy: float = 0.0
    training_runs: int = 0
    avg_accuracy: float = 0.0
    text_model: str = "basic"
    image_model: str = "yolo"
    dataset_size: Dict[str, Any] = {}

class DatasetStatusResponse(BaseModel):
    incident_images: Dict[str, Any]
    training_data: Dict[str, Any]
    total_size_mb: float
    storage_warning: bool

class MLStatsSummaryResponse(BaseModel):
    user_stats: Optional[MLAnalyticsResponse] = None
    model_performance: Optional[ModelPerformanceResponse] = None
    dataset_status: Optional[DatasetStatusResponse] = None

class Config:
        orm_mode = True
   
class ResponderLocationUpdate(BaseModel):
    lat: float
    lng: float
    accuracy: Optional[float] = None
    timestamp: Optional[float] = None   # Unix timestamp in milliseconds

class ResponderLocationResponse(BaseModel):
    responder_id: int
    name: str
    lat: float
    lng: float
    accuracy: Optional[float] = None
    last_update: datetime

class LegalComplianceCreate(BaseModel):
    title: str
    law_number: Optional[str] = None
    category: str
    description: str
    official_statement: str
    effective_date: Optional[str] = None
    is_active: bool = True

class LegalComplianceUpdate(BaseModel):
    title: Optional[str] = None
    law_number: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    official_statement: Optional[str] = None
    effective_date: Optional[str] = None
    is_active: Optional[bool] = None

class LegalComplianceResponse(BaseModel):
    id: int
    title: str
    law_number: Optional[str]
    category: str
    description: str
    official_statement: str
    effective_date: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True