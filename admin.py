from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from models import ChatHistory
from sqlalchemy import func, and_, or_, case, extract
from database import SessionLocal
from datetime import datetime, timezone

from models import IncidentReport, ResponderResolvedIncident
from datetime import datetime

import crud_incidents

from models import ResponderLocation, User
from schemas import ResponderLocationUpdate, ResponderLocationResponse

# ================= RESPONDER LOCATION TRACKING =================
from models import ResponderLocation
from pydantic import BaseModel
from datetime import datetime

from models import IncidentReport, User, IncidentAssignmentLog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import Alert, User
from schemas import AlertCreate
from deps import get_current_admin

import os
from models import AnonymousEmergency
from schemas import AnonymousEmergencyResponse

from deps import get_db, get_current_user
from models import User
from schemas import (
    UserOut, UserProfileOut, AdminDashboardStats, 
    UserListResponse, UserRoleUpdate, UserStatusUpdate,
    UserAdminUpdate, UserAdminCreate
)
import crud_users
from security import hash_password
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
import math
from typing import Optional

# ---------- CACHING FOR PERFORMANCE ----------
from cachetools import TTLCache
analytics_cache = TTLCache(maxsize=1, ttl=30)   # 30 seconds TTL

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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    valid_roles = ["user", "admin", "responder", "tmo"]
    if role_data.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    valid_statuses = ["active", "inactive", "suspended"]
    if status_data.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
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
    Optimized with single SQL queries.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from sqlalchemy import func, case
    
    # 1. User stats in one query
    user_stats = db.query(
        func.count(User.id).label('total'),
        func.sum(case((User.role == 'user', 1), else_=0)).label('citizens'),
        func.sum(case((User.role == 'responder', 1), else_=0)).label('responders'),
        func.sum(case((User.role == 'admin', 1), else_=0)).label('administrators'),
        func.sum(case((User.role == 'tmo', 1), else_=0)).label('tmoOfficers')
    ).first()
    
    # 2. Incident stats in one query
    incident_stats = db.query(
        func.count(IncidentReport.id).label('total'),
        func.sum(case((IncidentReport.status == 'pending', 1), else_=0)).label('pending'),
        func.sum(case((IncidentReport.status == 'in-progress', 1), else_=0)).label('in_progress'),
        func.sum(case((IncidentReport.status == 'resolved', 1), else_=0)).label('resolved')
    ).first()
    
    # 3. Recent incidents – select only needed columns
    recent = db.query(
        IncidentReport.id,
        IncidentReport.incident_type,
        IncidentReport.severity,
        IncidentReport.status,
        IncidentReport.barangay,
        IncidentReport.created_at
    ).order_by(IncidentReport.created_at.desc()).limit(10).all()
    
    recent_incidents_list = [
        {
            "id": r.id,
            "type": r.incident_type or "Unknown",
            "severity": r.severity or "medium",
            "status": r.status,
            "barangay": r.barangay or "Unknown",
            "created_at": r.created_at
        }
        for r in recent
    ]
    
    return AdminDashboardStats(
        total_incidents=incident_stats.total or 0,
        pending=incident_stats.pending or 0,
        in_progress=incident_stats.in_progress or 0,
        resolved=incident_stats.resolved or 0,
        total_users=user_stats.total or 0,
        citizens=user_stats.citizens or 0,
        responders=user_stats.responders or 0,
        tmoOfficers=user_stats.tmoOfficers or 0,
        administrators=user_stats.administrators or 0,
        recent_incidents=recent_incidents_list
    )

@router.get("/responders")
async def get_responders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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

@router.get("/incidents")
async def get_incidents(
    assigned_to: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get incidents.
    - Admin sees all.
    - Responder sees only approved incidents (verified_by not None and status != 'pending').
    """
    query = db.query(IncidentReport)
    
    # 🔐 Approval filter for responders
    if current_user.role == "responder":
        query = query.filter(
            IncidentReport.verified_by.isnot(None),
            IncidentReport.status != "pending"
        )
    
    if assigned_to is not None:
        query = query.filter(IncidentReport.assigned_to == assigned_to)
    if status:
        query = query.filter(IncidentReport.status == status)
    
    incidents = query.all()
    
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
            "image_paths": inc.image_paths,
            "video_paths": inc.video_paths,
            "text_analysis": inc.text_analysis,
            "created_at": inc.created_at,
            "resolved_at": inc.resolved_at,
            "assigned_to": inc.assigned_to,
            "user": {
                "full_name": inc.reporter.full_name if inc.reporter else None
            } if inc.reporter else None
        })
    return result

# ================= HEATMAP ENDPOINT =================
@router.get("/incidents/heatmap")
async def get_heatmap_data(
    days: int = Query(30, ge=1, le=365),
    status: Optional[str] = Query("in-progress"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Admin or responder access required")
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    query = db.query(IncidentReport).filter(
        IncidentReport.created_at >= cutoff_date,
        IncidentReport.latitude.isnot(None),
        IncidentReport.longitude.isnot(None)
    )
    if status and status != "all":
        query = query.filter(IncidentReport.status == status)
    incidents = query.all()
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

# ================= HELPER FUNCTIONS =================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_closest_responder(incident_lat: float, incident_lon: float, db: Session) -> Optional[int]:
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(minutes=60)
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
    stale = db.query(ResponderLocation).join(
        User, ResponderLocation.responder_id == User.id
    ).filter(
        User.role == "responder",
        User.status == "active"
    ).all()
    if stale:
        closest = min(stale, key=lambda r: haversine(incident_lat, incident_lon, r.latitude, r.longitude))
        return closest.responder_id
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
    return assign_incident_to_responder(incident, responder_id, admin_user, db)

def assign_incident_to_responder(
    incident: IncidentReport,
    responder_id: Optional[int],
    admin_user: User,
    db: Session
) -> str:
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

# ================= DYNAMIC INCIDENT DETAILS =================
@router.get("/incidents/{incident_id}")
async def get_incident_details(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    if current_user.role != "admin" and incident.assigned_to != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    old_status = incident.status
    incident.status = status
    if status == "resolved" and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()
    db.add(incident)
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
    assign_incident_to_responder_helper(incident, responder_id, current_user, db)
    db.refresh(incident)
    return {"message": "Assignment updated", "incident_id": incident_id, "assigned_to": incident.assigned_to}

def assign_incident_to_responder_helper(
    incident: IncidentReport,
    responder_id: Optional[int],
    admin_user: User,
    db: Session
) -> str:
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

# ================= ANALYTICS ENDPOINT (OPTIMIZED + CACHED) =================
@router.get("/analytics")
async def get_analytics_data(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Enhanced analytics with date filtering, hourly, weekly, barangay,
    resolution time, vehicle types (limited to 500 records for speed),
    and barangay trends. Cached for 30 seconds.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Parse dates
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
    end = end + timedelta(days=1)

    # Cache key
    cache_key = f"analytics_{start.isoformat()}_{end.isoformat()}"
    if cache_key in analytics_cache:
        return analytics_cache[cache_key]

    from sqlalchemy import func, case, extract, and_, or_

    base_filter = and_(
        IncidentReport.created_at >= start,
        IncidentReport.created_at < end
    )

    # ---- All aggregations in SQL ----
    type_rows = db.query(
        IncidentReport.incident_type,
        func.count(IncidentReport.id).label('cnt')
    ).filter(base_filter, IncidentReport.incident_type.isnot(None))\
     .group_by(IncidentReport.incident_type).all()
    incidentsByType = [{"name": t or "Unknown", "count": cnt, "percentage": 0} for t, cnt in type_rows]

    sev_rows = db.query(
        IncidentReport.severity,
        func.count(IncidentReport.id).label('cnt')
    ).filter(base_filter, IncidentReport.severity.isnot(None))\
     .group_by(IncidentReport.severity).all()
    severityDistribution = [{"level": s or "Unknown", "count": cnt} for s, cnt in sev_rows]

    daily_rows = db.query(
        func.date(IncidentReport.created_at).label('date'),
        func.count(IncidentReport.id).label('cnt')
    ).filter(base_filter).group_by('date').order_by('date').all()
    daily = [{"date": d.strftime("%Y-%m-%d"), "activity": cnt} for d, cnt in daily_rows]

    week_start = start - timedelta(days=start.weekday())
    weeklyTrend = []
    for i in range(4):
        w_start = week_start + timedelta(weeks=i)
        w_end = w_start + timedelta(weeks=1)
        cnt = db.query(func.count(IncidentReport.id)).filter(
            IncidentReport.created_at >= w_start,
            IncidentReport.created_at < w_end
        ).scalar() or 0
        weeklyTrend.append({"week": w_start.strftime("%Y-%m-%d"), "count": cnt})

    barangay_rows = db.query(
        IncidentReport.barangay,
        func.count(IncidentReport.id).label('cnt')
    ).filter(base_filter, IncidentReport.barangay.isnot(None))\
     .group_by(IncidentReport.barangay)\
     .order_by(func.count(IncidentReport.id).desc()).limit(10).all()
    barangayDistribution = [{"barangay": b or "Unknown", "count": cnt} for b, cnt in barangay_rows]

    hourly_rows = db.query(
        extract('hour', IncidentReport.created_at).label('hour'),
        func.count(IncidentReport.id).label('cnt')
    ).filter(base_filter, IncidentReport.created_at.isnot(None))\
     .group_by('hour').order_by('hour').all()
    hourlyDistribution = [{"hour": int(h), "count": cnt} for h, cnt in hourly_rows]

    # Average resolution time (hours)
    avg_res = db.query(
        func.avg(
            func.timestampdiff('hour', IncidentReport.created_at, IncidentReport.resolved_at)
        )
    ).filter(
        base_filter,
        IncidentReport.status == "resolved",
        IncidentReport.resolved_at.isnot(None),
        IncidentReport.created_at.isnot(None)
    ).scalar()
    avg_resolution = round(avg_res or 0, 2)

    # ---- Barangay trends (today, week, month) in one query ----
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    trend_rows = db.query(
        IncidentReport.barangay,
        func.sum(case((IncidentReport.created_at >= today_start, 1), else_=0)).label('today'),
        func.sum(case((IncidentReport.created_at >= week_ago, 1), else_=0)).label('week'),
        func.sum(case((IncidentReport.created_at >= month_ago, 1), else_=0)).label('month')
    ).filter(
        IncidentReport.barangay.isnot(None),
        IncidentReport.created_at >= month_ago
    ).group_by(IncidentReport.barangay)\
     .order_by(func.sum(case((IncidentReport.created_at >= month_ago, 1), else_=0)).desc())\
     .limit(10).all()

    barangay_trends = [
        {
            "barangay": row.barangay,
            "today": row.today,
            "week": row.week,
            "month": row.month,
            "total": row.month
        }
        for row in trend_rows
    ]

    # ---- Vehicle types (fast: only fetch last 500 with analysis) ----
    from collections import defaultdict
    import json

    vehicle_rows = db.query(
        IncidentReport.image_analysis,
        IncidentReport.text_analysis
    ).filter(
        base_filter,
        or_(
            IncidentReport.image_analysis.isnot(None),
            IncidentReport.text_analysis.isnot(None)
        )
    ).limit(500).all()

    vehicle_counts = defaultdict(int)
    for row in vehicle_rows:
        if row.image_analysis:
            try:
                img = json.loads(row.image_analysis)
                vehicles = img.get('vehicles', {})
                if isinstance(vehicles, dict):
                    for v, count in vehicles.items():
                        vehicle_counts[v] += count
            except:
                pass
        if row.text_analysis:
            try:
                txt = json.loads(row.text_analysis)
                mentioned = txt.get('mentioned_vehicles', [])
                if isinstance(mentioned, list):
                    for v in mentioned:
                        vehicle_counts[v] += 1
            except:
                pass

    vehicle_types = [{"type": k, "count": v} for k, v in vehicle_counts.items()]
    vehicle_types.sort(key=lambda x: -x["count"])

    # Build final response
    result = {
        "incidentsByType": incidentsByType,
        "severityDistribution": severityDistribution,
        "activitySummary": {
            "daily": daily,
            "weekly": weeklyTrend,
            "monthly": []
        },
        "barangayDistribution": barangayDistribution,
        "hourlyDistribution": hourlyDistribution,
        "weeklyTrend": weeklyTrend,
        "avgResolutionHours": avg_resolution,
        "vehicleTypes": vehicle_types,
        "barangayTrends": barangay_trends,
    }

    # Store in cache
    analytics_cache[cache_key] = result
    return result

# ================= USER CREATION ENDPOINT =================
@router.post("/users", response_model=UserProfileOut)
async def create_user_admin(
    user_data: UserAdminCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    existing_user = crud_users.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_dict = user_data.dict(exclude={'send_welcome_email'})
    user = crud_users.create_user_admin(db, user_dict)
    return user

@router.get("/barangays")
async def get_barangays(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    from sqlalchemy import distinct
    barangays = db.query(distinct(User.barangay)).filter(
        User.barangay.isnot(None),
        User.barangay != ''
    ).order_by(User.barangay).all()
    barangay_list = [barangay[0] for barangay in barangays]
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    from models import ChatHistory
    from sqlalchemy import func, and_
    subq = db.query(
        ChatHistory.user_id,
        func.max(ChatHistory.created_at).label('last_activity')
    ).group_by(ChatHistory.user_id).subquery()
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
    result.sort(key=lambda x: x["last_activity"], reverse=True)
    return result

@router.get("/chats/{user_id}/messages")
async def get_user_chat_messages(
    user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
    message_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    emergencies = db.query(AnonymousEmergency)\
                    .order_by(AnonymousEmergency.timestamp.desc())\
                    .offset(skip).limit(limit)\
                    .all()
    result = []
    for e in emergencies:
        audio_url = e.audio_path
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
    print("  target_zone:", alert_data.target_zone)
    print("  target_roles:", alert_data.target_roles)

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
    geometry: Optional[str] = Form(None),
    expires_at: Optional[str] = Form(None),
    image: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    expiration = None
    if expires_at:
        try:
            expiration = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expires_at format")
    geom_dict = None
    if geometry:
        try:
            geom_dict = json.loads(geometry)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid geometry JSON")
    image_url = None
    if image:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        upload_dir = "uploads/alerts"
        os.makedirs(upload_dir, exist_ok=True)
        ext = os.path.splitext(image.filename)[1] or '.jpg'
        filename = f"alert_{uuid.uuid4()}{ext}"
        file_path = os.path.join(upload_dir, filename)
        contents = await image.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        image_url = f"/uploads/alerts/{filename}"
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
    image_url: Optional[str] = None

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
    if alert.image_url:
        file_path = alert.image_url.lstrip('/')
        if os.path.exists(file_path):
            os.remove(file_path)
    db.delete(alert)
    db.commit()
    return {"message": "Alert deleted"}

@router.get("/training-data-status")
async def admin_training_data_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
    incident = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
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
    incident.incident_type = corrected_type
    incident.severity = corrected_severity
    incident.status = "verified"
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

# ================= APPROVAL ENDPOINT (SETS verified_by) =================
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

    # Mark as approved
    incident.status = "in-progress"
    incident.verified_by = current_user.id   # ✅ set approver
    incident.updated_at = datetime.utcnow()
    db.commit()  # commit status change first

    # Auto-assign if not already assigned
    if incident.assigned_to is None:
        msg = assign_closest_responder(incident, current_user, db)
        # Optionally log the auto-assignment message
    return {"message": "Incident approved and marked in-progress"}

# ================= DELETE INCIDENT =================
@router.delete("/incidents/{incident_id}")
async def delete_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
        existing.updated_at = datetime.utcnow()
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
    current_user: User = Depends(get_current_user)
):
    incident = crud_incidents.get_incident_report(db, incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
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
    if len(coords) < 5:
        features = []
        for inc in incidents:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [inc.longitude, inc.latitude]},
                "properties": {"intensity": 0.8}
            })
        return {"type": "FeatureCollection", "features": features}
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(coords)
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

# ================= AUTO-ASSIGN (ONLY FOR APPROVED INCIDENTS) =================
@router.post("/incidents/auto-assign")
async def auto_assign_all_incidents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, "Admin access required")

    active_responders = db.query(User).filter(
        User.role == "responder",
        User.status == "active"
    ).all()
    if not active_responders:
        return {"assigned": 0, "total_unassigned": 0, "errors": ["No active responders available"]}

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

    # 🔐 Only unassigned incidents that are APPROVED (in-progress AND verified_by not None)
    unassigned = db.query(IncidentReport).filter(
        IncidentReport.assigned_to.is_(None),
        IncidentReport.status == "in-progress",
        IncidentReport.verified_by.isnot(None),   # Must be approved
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
            db.commit()
            db.refresh(inc)
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
    # 🔐 Must be approved
    if incident.verified_by is None or incident.status == "pending":
        raise HTTPException(400, "Incident must be approved before assignment")
    msg = assign_closest_responder(incident, current_user, db)
    db.commit()
    return {"message": msg, "assigned_to": incident.assigned_to}