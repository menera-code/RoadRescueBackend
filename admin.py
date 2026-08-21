from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from models import ChatHistory
from sqlalchemy import func, and_
from database import SessionLocal
from datetime import datetime, timezone

from models import IncidentReport, ResponderResolvedIncident
from datetime import datetime

import crud_incidents

from models import ResponderLocation, User
from schemas import ResponderLocationUpdate, ResponderLocationResponse

# ================= RESPONDER LOCATION TRACKING =================
from models import ResponderLocation  # you'll need to create this model
from pydantic import BaseModel
from datetime import datetime

from models import IncidentReport, User, IncidentAssignmentLog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import Alert, User
from schemas import AlertCreate
from deps import get_current_admin  # ensure only admin can post

import os
from models import AnonymousEmergency
from schemas import AnonymousEmergencyResponse

from deps import get_db, get_current_user
from deps import get_current_user
from models import User
from schemas import (
    UserOut, UserProfileOut, AdminDashboardStats, 
    UserListResponse, UserRoleUpdate, UserStatusUpdate,
    UserAdminUpdate, UserAdminCreate  # Add UserAdminCreate here
)
import crud_users
from security import hash_password  # Add this import
from fastapi import BackgroundTasks, HTTPException, Depends

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from deps import get_current_user, get_db
from models import User
from sqlalchemy.orm import Session
from ml_analytics import ml_analytics



from models import LegalCompliance
from schemas import LegalComplianceCreate, LegalComplianceUpdate, LegalComplianceResponse

import json
from collections import Counter
from datetime import datetime
from fastapi import Query, HTTPException
from sqlalchemy.orm import Session
# ❌ Removed: import numpy as np
# ❌ Removed: from sklearn.neighbors import KernelDensity
import math
from typing import Optional

router = APIRouter(prefix="/admin", tags=["admin"])

# ================= USER MANAGEMENT ENDPOINTS =================

@router.get("/users", response_model=UserListResponse)
async def get_all_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = None,
    status: Optional[str] = None,
    barangay: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all users with filters (Admin only)
    """
    # Check if user is admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get users with filters
    result = crud_users.get_all_users(
        db, skip=skip, limit=limit, 
        role=role, status=status, barangay=barangay, search=search
    )
    
    return result

@router.get("/users/{user_id}", response_model=UserProfileOut)
async def get_user_details(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed user information (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = crud_users.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role_data: UserRoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user role (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Validate role
    valid_roles = ["user", "admin", "responder", "tmo"]
    if role_data.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    # Check if trying to change own role
    if user_id == current_user.id and role_data.role != "admin":
        raise HTTPException(status_code=400, detail="Cannot change your own admin role")
    
    user = crud_users.update_user_role(db, user_id, role_data.role)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"User role updated to {role_data.role}", "user": UserOut.from_orm(user)}

@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    status_data: UserStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user status (active/inactive) (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Validate status
    valid_statuses = ["active", "inactive", "suspended"]
    if status_data.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    # Check if trying to change own status
    if user_id == current_user.id and status_data.status != "active":
        raise HTTPException(status_code=400, detail="Cannot deactivate/suspend your own account")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.status = status_data.status
    db.commit()
    db.refresh(user)
    
    return {"message": f"User status updated to {status_data.status}", "user": UserOut.from_orm(user)}

@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user_data: UserAdminUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user information (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only provided fields
    update_data = user_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    
    return {"message": "User updated successfully", "user": UserProfileOut.from_orm(user)}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a user (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Check if trying to delete own account
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted successfully"}

# ================= DASHBOARD STATISTICS ENDPOINTS =================

@router.get("/dashboard/stats", response_model=AdminDashboardStats)
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard statistics (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get user statistics
    user_stats = crud_users.get_user_statistics(db)
    
    # Get incident statistics (you'll need to implement this in crud_incidents.py)
    # For now, using placeholder values
    from models import IncidentReport
    from sqlalchemy import func
    
    total_incidents = db.query(func.count(IncidentReport.id)).scalar() or 0
    pending = db.query(func.count(IncidentReport.id)).filter(IncidentReport.status == "pending").scalar() or 0
    in_progress = db.query(func.count(IncidentReport.id)).filter(IncidentReport.status == "in-progress").scalar() or 0
    resolved = db.query(func.count(IncidentReport.id)).filter(IncidentReport.status == "resolved").scalar() or 0
    
    # Get recent incidents (last 10)
    recent_incidents = db.query(IncidentReport).order_by(IncidentReport.created_at.desc()).limit(10).all()
    
    # Format recent incidents for response
    recent_incidents_list = []
    for incident in recent_incidents:
        recent_incidents_list.append({
            "id": incident.id,
            "type": incident.incident_type or "Unknown",
            "severity": incident.severity or "medium",
            "status": incident.status,
            "barangay": incident.barangay or "Unknown",
            "created_at": incident.created_at
        })
    
    return AdminDashboardStats(
        total_incidents=total_incidents,
        pending=pending,
        in_progress=in_progress,
        resolved=resolved,
        total_users=user_stats["total_users"],
        citizens=user_stats.get("citizens", 0),
        responders=user_stats.get("responders", 0),
        tmoOfficers=user_stats.get("tmoOfficers", 0),  # You need to add this to user_stats
        administrators=user_stats.get("administrators", 0),
        recent_incidents=recent_incidents_list
    )

@router.get("/responders")
async def get_responders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all responders (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    responders = db.query(User).filter(User.role == "responder", User.status == "active").all()
    
    return [
        {
            "id": user.id,
            "name": user.full_name,
            "contact_number": user.contact_number,
            "barangay": user.barangay
        }
        for user in responders
    ]

# ================= INCIDENT MANAGEMENT ENDPOINTS =================
from sqlalchemy.orm import Session
from models import IncidentReport

@router.get("/incidents")
async def get_incidents(
    assigned_to: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(IncidentReport)
    if assigned_to is not None:    
        query = query.filter(IncidentReport.assigned_to == assigned_to)
    if status:
        query = query.filter(IncidentReport.status == status)
    
    incidents = query.all()
    
    # Convert to list of dicts including media fields
    result = []
    for inc in incidents:
        result.append({
            "id": inc.id,
            "type": inc.incident_type,
            "incident_type": inc.incident_type,
            "severity": inc.severity,
            "status": inc.status,
            "description": inc.description,
            "barangay": inc.barangay,
            "address": inc.address,
            "contact_number": inc.contact_number,
            "emergency_contact": inc.emergency_contact,
            "latitude": inc.latitude,
            "longitude": inc.longitude,
            "image_paths": inc.image_paths,   # ← JSON string
            "video_paths": inc.video_paths,   # ← JSON string
            "text_analysis": inc.text_analysis,
            "created_at": inc.created_at,
            "resolved_at": inc.resolved_at,
            "assigned_to": inc.assigned_to,   # ✅ Ensure this field is included
            "user": {
                "full_name": inc.reporter.full_name if inc.reporter else None
            } if inc.reporter else None
        })
    return result

# ================= HEATMAP ENDPOINT (must come before /incidents/{incident_id}) =================
@router.get("/incidents/heatmap")
async def get_heatmap_data(
    days: int = Query(30, ge=1, le=365),
    status: Optional[str] = Query("in-progress"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Admin or responder access required")

    from datetime import datetime, timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(IncidentReport).filter(
        IncidentReport.created_at >= cutoff_date,
        IncidentReport.latitude.isnot(None),
        IncidentReport.longitude.isnot(None)
    )
    if status and status != "all":
        query = query.filter(IncidentReport.status == status)
    
    incidents = query.all()
    
    # Return empty list if none, but still 200 OK
    return [
        {
            "id": inc.id,
            "type": inc.incident_type or "Unknown",
            "severity": inc.severity or "medium",
            "status": inc.status,
            "latitude": inc.latitude,
            "longitude": inc.longitude,
            "barangay": inc.barangay or "Unknown",
            "created_at": inc.created_at.isoformat() if inc.created_at else None
        }
        for inc in incidents
    ]



def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_closest_responder(incident_lat: float, incident_lon: float, db: Session) -> Optional[int]:
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(minutes=60)

    # 1. Fresh locations (last 60 min)
    fresh = db.query(ResponderLocation).join(
        User, ResponderLocation.responder_id == User.id
    ).filter(
        User.role == "responder",
        User.status == "active",
        ResponderLocation.updated_at >= cutoff
    ).all()

    if fresh:
        closest = min(fresh, key=lambda r: haversine(incident_lat, incident_lon, r.latitude, r.longitude))
        return closest.responder_id

    # 2. Any location (stale)
    stale = db.query(ResponderLocation).join(
        User, ResponderLocation.responder_id == User.id
    ).filter(
        User.role == "responder",
        User.status == "active"
    ).all()

    if stale:
        closest = min(stale, key=lambda r: haversine(incident_lat, incident_lon, r.latitude, r.longitude))
        return closest.responder_id

    # 3. Fallback: any active responder
    any_responder = db.query(User).filter(
        User.role == "responder",
        User.status == "active"
    ).first()
    return any_responder.id if any_responder else None


def assign_closest_responder(incident: IncidentReport, admin_user: User, db: Session) -> str:
    if incident.latitude is None or incident.longitude is None:
        return "No location to determine closest responder."
    responder_id = get_closest_responder(incident.latitude, incident.longitude, db)
    if responder_id is None:
        return "No active responder available."
    # Use the existing assign function
    return assign_incident_to_responder(incident, responder_id, admin_user, db)


def assign_incident_to_responder(
    incident: IncidentReport,
    responder_id: Optional[int],
    admin_user: User,
    db: Session
) -> str:
    """Assign or unassign an incident. Returns a message."""
    from models import IncidentAssignmentLog
    previous = incident.assigned_to
    if responder_id is not None:
        responder = db.query(User).filter(User.id == responder_id, User.role == "responder").first()
        if not responder:
            raise HTTPException(404, "Responder not found")
        incident.assigned_to = responder_id
        action = "assign"
        msg = f"Assigned to {responder.full_name}"
    else:
        incident.assigned_to = None
        action = "unassign"
        msg = "Unassigned"

    incident.updated_at = datetime.utcnow()
    log = IncidentAssignmentLog(
        incident_id=incident.id,
        assigned_by=admin_user.id,
        assigned_to=responder_id if responder_id else previous,
        action=action
    )
    db.add(log)
    db.commit()
    return msg

# ================= DYNAMIC INCIDENT DETAILS (after static routes) =================
@router.get("/incidents/{incident_id}")
async def get_incident_details(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get incident details (Admin only)
    """
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Admin/responder access required")
    
    from models import IncidentReport
    from sqlalchemy.orm import joinedload
    
    incident = db.query(IncidentReport).options(
        joinedload(IncidentReport.reporter)
    ).filter(IncidentReport.id == incident_id).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return {
        "id": incident.id,
        "description": incident.description,
        "type": incident.incident_type,
        "severity": incident.severity,
        "priority": incident.priority,
        "latitude": incident.latitude,
        "longitude": incident.longitude,
        "barangay": incident.barangay,
        "address": incident.address,
        "status": incident.status,
        "contact_number": incident.contact_number,
        "emergency_contact": incident.emergency_contact,
        "created_at": incident.created_at,
        "updated_at": incident.updated_at,
        "ml_confidence": incident.ml_confidence,
        "keywords": incident.keywords or [],
        "reporter": {
            "id": incident.reporter.id if incident.reporter else None,
            "name": incident.reporter.full_name if incident.reporter else "Unknown",
            "contact": incident.reporter.contact_number if incident.reporter else None
        },
        "assigned_to": incident.assigned_to,
        "verified_by": incident.verified_by,
        "image_paths": incident.image_paths,
        "video_paths": incident.video_paths
    }

@router.put("/incidents/{incident_id}/status")
async def update_incident_status(
    incident_id: str,
    status: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Your existing permission checks...
    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Only admin or assigned responder can update
    if current_user.role != "admin" and incident.assigned_to != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    old_status = incident.status
    incident.status = status
    if status == "resolved" and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()
    db.add(incident)
    
    # ** ADD THIS BLOCK **
    if status == "resolved" and current_user.role == "responder" and old_status != "resolved":
        resolved_log = ResponderResolvedIncident(
            incident_id=incident_id,
            responder_id=current_user.id,
            resolved_at=datetime.utcnow(),
            notes=f"Resolved by {current_user.full_name}"
        )
        db.add(resolved_log)
    
    db.commit()
    return {"success": True, "message": f"Status updated to {status}"}

# ================= ASSIGN INCIDENT TO RESPONDER =================
@router.post("/incidents/{incident_id}/assign")
async def assign_incident_to_responder(
    incident_id: str,
    responder_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Use the helper that does the actual assignment
    assign_incident_to_responder_helper(incident, responder_id, current_user, db)
    
    # Ensure the incident is refreshed
    db.refresh(incident)
    return {"message": "Assignment updated", "incident_id": incident_id, "assigned_to": incident.assigned_to}
# ================= ANALYTICS ENDPOINTS =================
def assign_incident_to_responder_helper(
    incident: IncidentReport,
    responder_id: Optional[int],
    admin_user: User,
    db: Session
) -> str:
    """Assign or unassign an incident. Returns a message."""
    from models import IncidentAssignmentLog

    previous = incident.assigned_to
    if responder_id is not None:
        responder = db.query(User).filter(User.id == responder_id, User.role == "responder").first()
        if not responder:
            raise HTTPException(404, "Responder not found")
        incident.assigned_to = responder_id
        action = "assign"
        msg = f"Assigned to {responder.full_name}"
    else:
        incident.assigned_to = None
        action = "unassign"
        msg = "Unassigned"

    incident.updated_at = datetime.utcnow()
    log = IncidentAssignmentLog(
        incident_id=incident.id,
        assigned_by=admin_user.id,
        assigned_to=responder_id if responder_id else previous,
        action=action
    )
    db.add(log)
    db.commit()
    return msg
# ================= ML ANALYTICS ENDPOINTS =================

from ml_analytics import ml_analytics
from schemas import MLAnalyticsResponse, ModelPerformanceResponse, DatasetStatusResponse, MLStatsSummaryResponse
from deps import get_current_admin
from fastapi import Query
from typing import Optional

@router.get("/ml-analytics/{user_id}", response_model=MLAnalyticsResponse)
async def get_ml_analytics(
    user_id: int,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Get ML analytics for specific user (Admin only)
    """
    stats = ml_analytics.get_user_ml_stats(user_id, days, db)
    return MLAnalyticsResponse(**stats)

# ── Unified ML performance endpoints (only one set) ──

@router.get("/ml-performance")
async def admin_ml_performance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return model performance metrics (accuracy, version, etc.)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ml_analytics.get_model_performance(db)

@router.get("/dataset-status")
async def admin_dataset_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return storage info and training dataset counts"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ml_analytics.get_dataset_status(db)

@router.get("/ml-stats-summary")
async def admin_ml_stats_summary(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return summary stats (total predictions, avg confidence, by type, by severity)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ml_analytics.get_training_stats(days, db)

@router.get("/analytics")
async def get_analytics_data(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enhanced analytics with date filtering, hourly, weekly, barangay, resolution time, vehicle types, and barangay trends."""
    try:
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        # Parse dates, default to last 30 days if missing
        if start_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        else:
            start = datetime.utcnow() - timedelta(days=30)

        if end_date:
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        else:
            end = datetime.utcnow()

        # Ensure end is inclusive of the full day
        end = end + timedelta(days=1)

        # Base query for filtered period
        base = db.query(IncidentReport).filter(
            IncidentReport.created_at >= start,
            IncidentReport.created_at < end
        )

        # Incidents by type
        type_rows = base.filter(IncidentReport.incident_type.isnot(None))\
            .with_entities(IncidentReport.incident_type, func.count(IncidentReport.id))\
            .group_by(IncidentReport.incident_type).all()
        incidentsByType = [{"name": t, "count": c, "percentage": 0} for t, c in type_rows]

        # Severity distribution
        sev_rows = base.filter(IncidentReport.severity.isnot(None))\
            .with_entities(IncidentReport.severity, func.count(IncidentReport.id))\
            .group_by(IncidentReport.severity).all()
        severityDistribution = [{"level": s, "count": c} for s, c in sev_rows]

        # Daily activity
        daily_rows = base.with_entities(
            func.date(IncidentReport.created_at).label('date'),
            func.count(IncidentReport.id)
        ).group_by('date').order_by('date').all()
        daily = [{"date": d.strftime("%Y-%m-%d"), "activity": cnt} for d, cnt in daily_rows]

        # Weekly trend (last 4 weeks) – we compute from the start date
        week_start = start - timedelta(days=start.weekday())  # align to Monday
        weeks = []
        for i in range(4):
            w_start = week_start + timedelta(weeks=i)
            w_end = w_start + timedelta(weeks=1)
            cnt = db.query(func.count(IncidentReport.id)).filter(
                IncidentReport.created_at >= w_start,
                IncidentReport.created_at < w_end
            ).scalar() or 0
            weeks.append({
                "week": w_start.strftime("%Y-%m-%d"),
                "count": cnt
            })
        weeklyTrend = weeks

        # Barangay distribution (top 10) – for the filtered period
        barangay_rows = base.filter(IncidentReport.barangay.isnot(None))\
            .with_entities(IncidentReport.barangay, func.count(IncidentReport.id))\
            .group_by(IncidentReport.barangay).order_by(func.count(IncidentReport.id).desc()).limit(10).all()
        barangayDistribution = [{"barangay": b, "count": c} for b, c in barangay_rows]

        # Hourly distribution
        hourly_rows = base.filter(IncidentReport.created_at.isnot(None))\
            .with_entities(
                func.extract('hour', IncidentReport.created_at).label('hour'),
                func.count(IncidentReport.id)
            ).group_by('hour').order_by('hour').all()
        hourlyDistribution = [{"hour": int(h), "count": c} for h, c in hourly_rows]

        # Average resolution time (in hours) for resolved incidents
        resolved = base.filter(IncidentReport.status == "resolved",
                               IncidentReport.resolved_at.isnot(None),
                               IncidentReport.created_at.isnot(None))\
            .with_entities(IncidentReport.created_at, IncidentReport.resolved_at).all()
        if resolved:
            total_hours = sum(
                (r.resolved_at - r.created_at).total_seconds() / 3600 for r in resolved
            )
            avg_resolution = round(total_hours / len(resolved), 2)
        else:
            avg_resolution = 0

        # ========== SAFE VEHICLE TYPE DISTRIBUTION ==========
        from collections import defaultdict
        import json

        vehicle_counts = defaultdict(int)
        incidents_in_period = base.all()

        for inc in incidents_in_period:
            # Image analysis
            if inc.image_analysis:
                try:
                    img_data = json.loads(inc.image_analysis)
                    if isinstance(img_data, dict):
                        vehicles = img_data.get("vehicles", {})
                        if isinstance(vehicles, dict):
                            for vehicle, count in vehicles.items():
                                if isinstance(count, (int, float)):
                                    vehicle_counts[vehicle] += int(count)
                except Exception as e:
                    print(f"⚠️ Image analysis error for incident {inc.id}: {e}")

            # Text analysis
            if inc.text_analysis:
                try:
                    txt_data = json.loads(inc.text_analysis)
                    if isinstance(txt_data, dict):
                        vehicles = txt_data.get("mentioned_vehicles")
                        if isinstance(vehicles, list):
                            for vehicle in vehicles:
                                vehicle_counts[vehicle] += 1
                except Exception as e:
                    print(f"⚠️ Text analysis error for incident {inc.id}: {e}")

        vehicle_types = [{"type": k, "count": v} for k, v in vehicle_counts.items() if v > 0]
        vehicle_types.sort(key=lambda x: -x["count"])

        # ========== SAFE BARANGAY TRENDS (Today, Week, Month) ==========
        
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)

        barangay_periods = {}
        for period, start_dt in [("today", today_start), ("week", week_start), ("month", month_start)]:
            results = db.query(IncidentReport.barangay, func.count(IncidentReport.id))\
                        .filter(IncidentReport.created_at >= start_dt)\
                        .filter(IncidentReport.barangay.isnot(None))\
                        .group_by(IncidentReport.barangay)\
                        .all()
            for barangay, cnt in results:
                if barangay not in barangay_periods:
                    barangay_periods[barangay] = {"today": 0, "week": 0, "month": 0}
                barangay_periods[barangay][period] = cnt

        barangay_trends = []
        for barangay, periods in barangay_periods.items():
            barangay_trends.append({
                "barangay": barangay,
                "today": periods["today"],
                "week": periods["week"],
                "month": periods["month"],
                "total": periods["today"] + periods["week"] + periods["month"]
            })
        barangay_trends.sort(key=lambda x: -x["total"])
        barangay_trends = barangay_trends[:10]

        return {
            "incidentsByType": incidentsByType,
            "severityDistribution": severityDistribution,
            "activitySummary": {
                "daily": daily,
                "weekly": weeklyTrend,
                "monthly": []  # optional
            },
            "barangayDistribution": barangayDistribution,
            "hourlyDistribution": hourlyDistribution,
            "weeklyTrend": weeklyTrend,
            "avgResolutionHours": avg_resolution,
            "vehicleTypes": vehicle_types,
            "barangayTrends": barangay_trends,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
# ================= USER CREATION ENDPOINT =================

@router.post("/users", response_model=UserProfileOut)
async def create_user_admin(
    user_data: UserAdminCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new user (Admin only)
    """
    # Check if user is admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Check if email already exists
    existing_user = crud_users.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user using crud function
    user_dict = user_data.dict(exclude={'send_welcome_email'})
    
    # Create user
    user = crud_users.create_user_admin(db, user_dict)
    
    return user

@router.get("/barangays")
async def get_barangays(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of barangays for dropdown (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get unique barangays from users
    from sqlalchemy import distinct
    barangays = db.query(distinct(User.barangay)).filter(
        User.barangay.isnot(None),
        User.barangay != ''
    ).order_by(User.barangay).all()
    
    # Extract barangay names
    barangay_list = [barangay[0] for barangay in barangays]
    
    # If no barangays found, return default list
    if not barangay_list:
        barangay_list = [
            "Bayanan I", "Bayanan II", "Calero", "Camilmil",
            "Camilmil", "Canubing I", "Canubing II", "Comunal",
            "Guinobatan", "Gutad", "Ibaba East", "Ibaba West",
            "Ilaya", "Lalud", "Lazareto", "Maidlang",
            "Malidong", "Pachoca", "Palhi", "Panggalaan",
            "Parang", "Patas", "Puting Tubig", "San Antonio",
            "San Vicente Central", "San Vicente East", "San Vicente North",
            "San Vicente South", "San Vicente West", "Santa Cruz",
            "Santa Isabel", "Santa Maria", "Santo Niño",
            "Sapul", "Silonay", "Suqui", "Tawagan",
            "Tawiran", "Tibag", "Wawa"
        ]
    
    return {"barangays": barangay_list}

# ================= ADMIN CHAT ENDPOINTS =================

@router.get("/chats")
async def get_chat_list(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of users with chat history, including last message and timestamp.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from models import ChatHistory
    from sqlalchemy import func, and_
    
    # Subquery: latest message timestamp per user
    subq = db.query(
        ChatHistory.user_id,
        func.max(ChatHistory.created_at).label('last_activity')
    ).group_by(ChatHistory.user_id).subquery()
    
    # Get the actual last message for each user
    last_msgs = db.query(
        ChatHistory.user_id,
        ChatHistory.message,
        ChatHistory.created_at
    ).join(
        subq,
        and_(
            ChatHistory.user_id == subq.c.user_id,
            ChatHistory.created_at == subq.c.last_activity
        )
    ).all()
    
    # Fetch user details
    user_ids = [msg.user_id for msg in last_msgs]
    users = db.query(User).filter(User.id.in_(user_ids)).all()
    user_dict = {u.id: u for u in users}
    
    result = []
    for msg in last_msgs:
        user = user_dict.get(msg.user_id)
        if user:
            result.append({
                "id": user.id,
                "user_name": user.full_name,
                "last_message": msg.message,
                "last_activity": msg.created_at.isoformat(),
                "avatar": user.profile_photo
            })
    
    # Optional: sort by last_activity descending
    result.sort(key=lambda x: x["last_activity"], reverse=True)
    return result

@router.get("/chats/{user_id}/messages")
async def get_user_chat_messages(
    user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all messages (user + assistant) for a specific user.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from models import ChatHistory
    
    messages = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id
    ).order_by(ChatHistory.created_at.asc()).limit(limit).all()
    
    return [
        {
            "role": msg.role,
            "message": msg.message,
            "timestamp": msg.created_at.isoformat(),
            "metadata": msg.chat_metadata
        }
        for msg in messages
    ]

@router.post("/chats/{user_id}/message")
async def send_admin_message(
    user_id: int,
    message_data: dict,  # expects {"message": "text"}
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Send a message as admin to a user's chat. The message is stored as an
    assistant message with metadata indicating it came from admin.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from models import ChatHistory
    import json
    
    new_msg = ChatHistory(
        user_id=user_id,
        role="assistant",
        message=message_data.get("message"),
        chat_metadata=json.dumps({"admin_id": current_user.id, "source": "admin"})
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    
    return {
        "role": new_msg.role,
        "message": new_msg.message,
        "timestamp": new_msg.created_at.isoformat(),
        "metadata": new_msg.chat_metadata
    }

@router.get("/emergencies/anonymous", response_model=list[AnonymousEmergencyResponse])
async def list_anonymous_emergencies(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """
    List all anonymous emergency alerts (no login required by user).
    Accessible only to admins.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    emergencies = db.query(AnonymousEmergency)\
                    .order_by(AnonymousEmergency.timestamp.desc())\
                    .offset(skip).limit(limit)\
                    .all()

    result = []
    for e in emergencies:
        # Extract filename from stored path and build public URL
        filename = os.path.basename(e.audio_path)
        audio_url = f"/uploads/emergencies/{filename}"
        result.append(AnonymousEmergencyResponse(
            id=e.id,
            latitude=e.latitude,
            longitude=e.longitude,
            audio_url=audio_url,
            timestamp=e.timestamp
        ))
    return result

@router.post("/broadcast-alert")
def broadcast_alert(
    alert_data: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    print("=" * 50)
    print("Received alert_data:")
    print("  message:", alert_data.message)
    print("  severity:", alert_data.severity)
    print("  geometry:", alert_data.geometry)
    print("  type(geometry):", type(alert_data.geometry))
    print("  target_zone:", alert_data.target_zone)      # ✅ snake_case
    print("  target_roles:", alert_data.target_roles)    # ✅ snake_case

    new_alert = Alert(
        message=alert_data.message,
        severity=alert_data.severity,
        target_zone=alert_data.target_zone,
        target_roles=alert_data.target_roles,
        geometry=alert_data.geometry,
        created_by=current_user.id,
        expires_at=alert_data.expires_at,
    )
    print("Before commit - new_alert.geometry:", new_alert.geometry)

    try:
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)
        print("After commit - new_alert.geometry:", new_alert.geometry)
    except Exception as e:
        print("!!! Database error:", e)
        raise

    return {"success": True, "alert_id": new_alert.id}

import json
import uuid
import os
from fastapi import Form, UploadFile, File

@router.post("/broadcast-alert-with-image")
async def broadcast_alert_with_image(
    message: str = Form(...),
    severity: str = Form("medium"),
    geometry: Optional[str] = Form(None),   # GeoJSON as JSON string
    expires_at: Optional[str] = Form(None),   # ISO string from frontend
    image: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):

    # Parse expires_at if provided
    expiration = None
    if expires_at:
        try:
            expiration = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expires_at format")

    # Parse geometry if provided
    geom_dict = None
    if geometry:
        try:
            geom_dict = json.loads(geometry)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid geometry JSON")

    # Handle image upload
    image_url = None
    if image:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create upload directory if not exists
        upload_dir = "uploads/alerts"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        ext = os.path.splitext(image.filename)[1] or '.jpg'
        filename = f"alert_{uuid.uuid4()}{ext}"
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        contents = await image.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        image_url = f"/uploads/alerts/{filename}"

    # Create alert record
    new_alert = Alert(
        message=message,
        severity=severity,
        geometry=geom_dict,
        image_url=image_url,
        created_by=current_user.id,
        expires_at=expiration

    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)

    return {"success": True, "alert_id": new_alert.id}

from pydantic import BaseModel

class AlertUpdate(BaseModel):
    message: Optional[str] = None
    severity: Optional[str] = None
    geometry: Optional[dict] = None
    image_url: Optional[str] = None   # not for direct update, but if we allow image change

@router.get("/alerts")
def get_all_alerts(
    include_expired: bool = False,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    query = db.query(Alert)
    if not include_expired:
        # Show only alerts that have no expiration date OR expiration is in the future
        query = query.filter(
            (Alert.expires_at == None) | (Alert.expires_at > datetime.utcnow())
        )
    alerts = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    return alerts

@router.get("/alerts/{alert_id}")
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@router.put("/alerts/{alert_id}")
def update_alert(
    alert_id: int,
    alert_data: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    update_data = alert_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)
    
    db.commit()
    db.refresh(alert)
    return alert

@router.delete("/alerts/{alert_id}")
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Optionally delete image file
    if alert.image_url:
        file_path = alert.image_url.lstrip('/')
        if os.path.exists(file_path):
            os.remove(file_path)
    
    db.delete(alert)
    db.commit()
    return {"message": "Alert deleted"}


# ─── The following three endpoints already exist above; they are duplicates.
# I'm removing them to avoid conflicts. Keep the ones above.

# @router.get("/ml-performance")
# ...
# @router.get("/dataset-status")
# ...
# @router.get("/ml-stats-summary")
# ...

@router.get("/training-data-status")
async def admin_training_data_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return counts of verified and used training samples"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ml_analytics.get_training_data_status(db)


@router.post("/ml/train")
async def start_training(
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return {"message": "Training endpoint disabled - using predictor.py"}

@router.put("/api/reports/{report_id}/verify")
async def verify_report(
    report_id: str,
    corrected_type: str,
    corrected_severity: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get the original incident
    incident = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Check if already in training_datasets
    existing = db.query(TrainingDataset).filter(TrainingDataset.report_id == report_id).first()
    if not existing:
        training = TrainingDataset(
            report_id=report_id,
            description=incident.description,
            incident_type=incident.incident_type,
            severity=incident.severity,
            corrected_type=corrected_type,
            corrected_severity=corrected_severity,
            image_paths=incident.image_paths,
            video_paths=incident.video_paths,
            is_verified=True,
            verified_by=current_user.id,
            used_in_training=False
        )
        db.add(training)
    else:
        existing.corrected_type = corrected_type
        existing.corrected_severity = corrected_severity
        existing.is_verified = True
        existing.verified_by = current_user.id
        existing.used_in_training = False

    # Optionally update the incident's status or type
    incident.incident_type = corrected_type
    incident.severity = corrected_severity
    incident.status = "verified"  # or whatever you prefer

    db.commit()
    return {"success": True, "message": "Report verified and added to training data"}


@router.get("/training-data-status")
async def training_data_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ml_analytics.get_training_data_status(db)

@router.post("/incidents/{incident_id}/approve")
async def approve_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, "Admin access required")

    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(404, "Incident not found")

    incident.status = "in-progress"
    incident.updated_at = datetime.utcnow()
    db.commit()  # commit status change first

    if incident.assigned_to is None:
        msg = assign_closest_responder(incident, current_user, db)
        # log or notify
    return {"message": "Incident approved and marked in-progress"}

@router.delete("/incidents/{incident_id}")
async def delete_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Permanently delete an incident (Admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from models import IncidentReport
    
    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    db.delete(incident)
    db.commit()
    
    return {"message": "Incident deleted successfully"}

@router.get("/incidents/{incident_id}/assignments")
async def get_incident_assignment_history(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from models import IncidentAssignmentLog, User
    
    logs = db.query(IncidentAssignmentLog).filter(
        IncidentAssignmentLog.incident_id == incident_id
    ).order_by(IncidentAssignmentLog.created_at.desc()).all()
    
    result = []
    for log in logs:
        assigner = db.query(User).filter(User.id == log.assigned_by).first()
        assignee = db.query(User).filter(User.id == log.assigned_to).first() if log.assigned_to else None
        result.append({
            "id": log.id,
            "action": log.action,
            "assigned_by": assigner.full_name if assigner else "Unknown",
            "assigned_to": assignee.full_name if assignee else "None",
            "timestamp": log.created_at.isoformat()
        })
    return result

class ResponderLocationUpdate(BaseModel):
    lat: float
    lng: float
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

@router.post("/responder/location")
async def update_responder_location(
    location: ResponderLocationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "responder":
        raise HTTPException(status_code=403, detail="Only responders can update location")
    
    existing = db.query(ResponderLocation).filter(ResponderLocation.responder_id == current_user.id).first()
    if existing:
        existing.latitude = location.lat
        existing.longitude = location.lng
        existing.accuracy = location.accuracy
        existing.updated_at = datetime.utcnow()   # <-- this works fine
    else:
        new_loc = ResponderLocation(
            responder_id=current_user.id,
            latitude=location.lat,
            longitude=location.lng,
            accuracy=location.accuracy,
            updated_at=datetime.utcnow()
        )
        db.add(new_loc)
    db.commit()
    return {"success": True}

@router.get("/responder-locations", response_model=list[ResponderLocationResponse])
async def get_responder_locations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Admin or responder access required")

    results = db.query(ResponderLocation, User.full_name).join(
        User, ResponderLocation.responder_id == User.id
    ).filter(User.role == "responder").all()
    
    return [
        ResponderLocationResponse(
            responder_id=loc.responder_id,
            name=name,
            lat=loc.latitude,
            lng=loc.longitude,
            accuracy=loc.accuracy,
            last_update=loc.updated_at.isoformat() + "Z"
        )
        for loc, name in results
    ]

# ================= LEGAL COMPLIANCE ENDPOINTS =================

@router.get("/legal-compliances", response_model=List[LegalComplianceResponse])
async def get_legal_compliances(
    category: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    query = db.query(LegalCompliance)
    if category:
        query = query.filter(LegalCompliance.category == category)
    if is_active is not None:
        query = query.filter(LegalCompliance.is_active == is_active)
    
    return query.order_by(LegalCompliance.created_at.desc()).all()


@router.post("/legal-compliances", response_model=LegalComplianceResponse)
async def create_legal_compliance(
    data: LegalComplianceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    entry = LegalCompliance(**data.dict(), created_by=current_user.id)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.put("/legal-compliances/{entry_id}", response_model=LegalComplianceResponse)
async def update_legal_compliance(
    entry_id: int,
    data: LegalComplianceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    entry = db.query(LegalCompliance).filter(LegalCompliance.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    for field, value in data.dict(exclude_unset=True).items():
        setattr(entry, field, value)
    
    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/legal-compliances/{entry_id}")
async def delete_legal_compliance(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    entry = db.query(LegalCompliance).filter(LegalCompliance.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    db.delete(entry)
    db.commit()
    return {"message": "Legal compliance entry deleted"}

@router.get("/incidents/{incident_id}/media-analysis")
def get_media_analysis(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)   # ← now uses current user (not admin)
):
    incident = crud_incidents.get_incident_report(db, incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    
    # ✅ Allow: admin, the reporter (user_id), or the assigned responder (assigned_to)
    if current_user.role != "admin" and \
       current_user.id != incident.user_id and \
       current_user.id != incident.assigned_to:
        raise HTTPException(403, "Not authorized to view this report's analysis")
    
    return {
        "text_analysis": json.loads(incident.text_analysis) if incident.text_analysis else None,
        "image_analysis": json.loads(incident.image_analysis) if incident.image_analysis else None,
        "video_analysis": json.loads(incident.video_analysis) if incident.video_analysis else None,
    }

@router.get("/ml/prediction-stats")
def get_prediction_stats(
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    incidents = db.query(IncidentReport).filter(IncidentReport.ml_confidence.isnot(None)).all()
    type_counts = Counter()
    severity_counts = Counter()
    confidences = []
    for inc in incidents:
        if inc.incident_type:
            type_counts[inc.incident_type] += 1
        if inc.severity:
            severity_counts[inc.severity] += 1
        if inc.ml_confidence:
            confidences.append(inc.ml_confidence)
    avg_conf = sum(confidences)/len(confidences) if confidences else 0
    return {
        "total_predictions": len(incidents),
        "avg_confidence": round(avg_conf, 2),
        "by_type": [{"type": k, "count": v} for k, v in type_counts.most_common()],
        "by_severity": [{"severity": k, "count": v} for k, v in severity_counts.most_common()],
    }

@router.get("/incidents/heatmap-predict")
def predict_hotspots(
    start_date: datetime = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: datetime = Query(..., description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    # ✅ Lazy‑load heavy libraries only when this endpoint is called
    try:
        from sklearn.neighbors import KernelDensity
        import numpy as np
    except ImportError:
        raise HTTPException(500, "scikit-learn or numpy not installed")

    incidents = db.query(IncidentReport).filter(
        IncidentReport.created_at >= start_date,
        IncidentReport.created_at <= end_date,
        IncidentReport.latitude.isnot(None),
        IncidentReport.longitude.isnot(None)
    ).all()

    if len(incidents) < 3:
        return {"type": "FeatureCollection", "features": [], "message": "Not enough data"}

    coords = np.radians([[i.latitude, i.longitude] for i in incidents])
    
    # Fallback: if only few points, just return those points as hotspots
    if len(coords) < 5:
        features = []
        for inc in incidents:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [inc.longitude, inc.latitude]},
                "properties": {"intensity": 0.8}
            })
        return {"type": "FeatureCollection", "features": features}

    # Use KDE only if enough points
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(coords)

    # Grid over Calapan area (adjust bounds as needed)
    lat_grid = np.linspace(13.35, 13.45, 30)
    lng_grid = np.linspace(121.13, 121.23, 30)
    points = np.array([[lat, lng] for lat in lat_grid for lng in lng_grid])
    points_rad = np.radians(points)
    densities = np.exp(kde.score_samples(points_rad))
    densities = densities / densities.max() if densities.max() > 0 else densities

    features = []
    for i, (lat, lng) in enumerate(points):
        if densities[i] > 0.1:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {"intensity": float(densities[i])}
            })
    return {"type": "FeatureCollection", "features": features}

@router.post("/incidents/auto-assign")
async def auto_assign_all_incidents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, "Admin access required")

    # 1. Get all active responders
    active_responders = db.query(User).filter(
        User.role == "responder",
        User.status == "active"
    ).all()
    if not active_responders:
        return {"assigned": 0, "total_unassigned": 0, "errors": ["No active responders available"]}

    # 2. Current assignment counts and locations
    responder_counts = {}
    responder_locations = {}
    for resp in active_responders:
        count = db.query(IncidentReport).filter(
            IncidentReport.assigned_to == resp.id,
            IncidentReport.status.in_(["pending", "in-progress"])
        ).count()
        responder_counts[resp.id] = count
        loc = db.query(ResponderLocation).filter(
            ResponderLocation.responder_id == resp.id
        ).order_by(ResponderLocation.updated_at.desc()).first()
        responder_locations[resp.id] = loc

    # 3. Unassigned incidents (oldest first)
    unassigned = db.query(IncidentReport).filter(
        IncidentReport.assigned_to.is_(None),
        IncidentReport.status.in_(["pending", "in-progress"]),
        IncidentReport.latitude.isnot(None),
        IncidentReport.longitude.isnot(None)
    ).order_by(IncidentReport.created_at.asc()).all()

    assigned_count = 0
    errors = []
    assignments = []

    for inc in unassigned:
        best_responder_id = None
        best_score = None

        for resp in active_responders:
            loc = responder_locations.get(resp.id)
            if loc:
                dist = haversine(inc.latitude, inc.longitude, loc.latitude, loc.longitude)
            else:
                dist = 100000
            score = responder_counts[resp.id] * 1000 + dist
            if best_score is None or score < best_score:
                best_score = score
                best_responder_id = resp.id

        if best_responder_id is None:
            errors.append(f"No suitable responder for incident {inc.id}")
            continue

        # ✅ Perform assignment with immediate commit
        try:
            inc.assigned_to = best_responder_id
            inc.updated_at = datetime.utcnow()
            
            log = IncidentAssignmentLog(
                incident_id=inc.id,
                assigned_by=current_user.id,
                assigned_to=best_responder_id,
                action="assign"
            )
            db.add(log)
            db.commit()          # <-- commit now
            db.refresh(inc)      # <-- refresh to get latest data

            assigned_count += 1
            assignments.append({"incident_id": inc.id, "assigned_to": best_responder_id})
            responder_counts[best_responder_id] += 1
        except Exception as e:
            db.rollback()
            errors.append(f"Error assigning incident {inc.id}: {str(e)}")

    return {
        "assigned": assigned_count,
        "total_unassigned": len(unassigned),
        "assignments": assignments,
        "errors": errors
    }

@router.post("/incidents/{incident_id}/auto-assign")
async def auto_assign_single_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(404, "Incident not found")
    if incident.assigned_to is not None:
        return {"message": "Already assigned", "assigned_to": incident.assigned_to}
    msg = assign_closest_responder(incident, current_user, db)
    db.commit()
    return {"message": msg, "assigned_to": incident.assigned_to}