import os, shutil
import traceback
import json
import google.generativeai as genai
import requests
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import asyncio
from admin import router as admin_router
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import asyncio
from database import SessionLocal  # adjust import if needed
from models import Alert
from schemas import AlertResponse
import os
import uuid
from fastapi import File, UploadFile, Form, Request

from services.predictor import predict_text, analyze_image, analyze_video
from schemas import LegalComplianceResponse
# Import your modules
from models import ResponderLocation
import models
import logging
import crud_users
from routing_service import RoutingService
from database import engine
from deps import get_db, get_current_user
from schemas import RegisterIn, LoginIn, TokenOut, UserOut, UserProfileOut, UserProfileUpdate
from security import create_access_token, decode_token
from models import User
from chat import router as chat_router
from gemini_map_service import map_service  # NEW: Import Gemini map service
from realtime_routing_service import RealTimeRoutingService
from websocket_manager import connection_manager
# Add these imports with your other imports

from schemas import (
    IncidentReportCreate, 
    IncidentReportResponse,
    MLTextAnalysisResponse,
    MLReportAnalysisResponse
)
from typing import List

# Add these imports at the top
import crud_incidents
from schemas import (
    IncidentReportCreate, IncidentReportResponse, 
    MLTextAnalysisResponse, MLReportAnalysisResponse,
    IncidentUpdateCreate, IncidentUpdateResponse,
    StatisticsResponse
)
from typing import List
# Add these to your existing imports
from fastapi import Form, BackgroundTasks


from fastapi import Depends, HTTPException, status
from deps import get_current_user  # or wherever your get_current_user is
from models import User
from models import IncidentReport, ResponderResolvedIncident

# ✨ NEW: Import the updated ml_analytics (you rewrote this)
import ml_analytics

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="RESQAPP API")

# ========== CORS MIDDLEWARE ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.middleware("http")
async def update_last_active_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Skip if not an authenticated route (you can adjust the condition)
    if request.url.path.startswith("/uploads") or request.url.path == "/health":
        return response
    
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = decode_token(token)
            user_id = int(payload.get("sub"))
            
            # Update last_active in a separate session to not interfere with request cycle
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    user.last_active = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            finally:
                db.close()
        except Exception:
            pass
    
    return response


UPLOAD_DIR = "uploads/avatars"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ========== PYDANTIC MODELS ==========
class RouteRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    profile: str = "driving"

class AIRouteRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    optimize_for: str = "fastest"
    user_context: Optional[Dict[str, Any]] = None

class RouteInsightsRequest(BaseModel):
    route_coords: List[List[float]]

class TrafficPredictionRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    hours_ahead: int = 2

class LiveTrafficRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float

class AlternativesRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    current_route_time: Optional[float] = None

class ETARequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    original_duration: float
    traffic_level: Optional[str] = None

class AlertsRequest(BaseModel):
    start_lat: float
    start_lng: float
    dest_lat: float
    dest_lng: float
    radius_km: Optional[float] = 5

class UserPositionRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    accuracy: Optional[float] = None


async def process_incident_background(incident_id: str, ml_result: dict):
    """Background processing for new incidents (e.g., notifications)."""
    print(f"📊 Processing report {incident_id} in background...")
    # Add any async tasks you need (email, SMS, etc.)
    # For now, just a placeholder.
    await asyncio.sleep(1)
    print(f"✅ Report {incident_id} processed successfully")

# Initialize services
realtime_service = RealTimeRoutingService()

# ========== BASIC ROUTES ==========
@app.post("/users/me/avatar")
def upload_avatar(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    payload = decode_token(credentials.credentials)
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    ext = file.filename.split(".")[-1]
    filename = f"user_{user.id}.{ext}"
    file_path = f"{UPLOAD_DIR}/{filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    user.profile_photo = file_path
    db.commit()
    
    return {"profile_photo": file_path}

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("🔥 UNHANDLED ERROR:")
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error_type": exc.__class__.__name__,
            "detail": str(exc),
        },
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/auth/register", response_model=UserOut)
def register(data: RegisterIn, db: Session = Depends(get_db)):
    if crud_users.get_user_by_email(db, data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = crud_users.create_user(db, data)
    return user

@app.post("/auth/login", response_model=TokenOut)
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = crud_users.authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

        # ✅ Set user online
    user.is_online = True
    db.commit()
    db.refresh(user)

    token = create_access_token(sub=str(user.id), role=user.role)
    return {"access_token": token, "token_type": "bearer"}

security = HTTPBearer()

@app.post("/auth/logout")
def logout(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    current_user.is_online = False
    db.commit()
    return {"message": "Logged out successfully"}

@app.get("/me")
def me(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    payload = decode_token(creds.credentials)
    user = db.query(User).filter(User.id == int(payload["sub"])).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role
    }

@app.get("/users/me", response_model=UserProfileOut)
def read_my_profile(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """Get full profile of the currently authenticated user."""
    payload = decode_token(credentials.credentials)
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/me", response_model=UserProfileOut)
def update_my_profile(
    payload: UserProfileUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """Update profile of the currently authenticated user."""
    token_data = decode_token(credentials.credentials)
    user_id = int(token_data.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only fields that are provided
    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user

# Include chat router
app.include_router(chat_router)

@app.get("/debug/cors")
async def debug_cors(request: Request):
    """Debug CORS headers"""
    headers = dict(request.headers)
    return {
        "headers": headers,
        "origin": request.headers.get("origin"),
        "cors_working": True
    }

@app.post("/api/route/calculate")
async def calculate_route(route_request: RouteRequest):
    # TEMPORARY: Override profile for testing
    # You can send a custom header like X-Profile: foot
    # Or just force it for now
    # profile = route_request.profile
    profile = "foot"   # Force walking to test
    
    routing_service = RoutingService()
    result = routing_service.calculate_route(
        start_lat=route_request.start_lat,
        start_lng=route_request.start_lng,
        dest_lat=route_request.dest_lat,
        dest_lng=route_request.dest_lng,
        profile=profile
    )
    return result

# ========== GEMINI AI MAP ENDPOINTS ==========
@app.post("/api/map/ai-route-analysis")
async def ai_route_analysis(request: AIRouteRequest):
    """Get AI-powered route analysis with Gemini"""
    try:
        result = await map_service.get_ai_route_analysis(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            dest_lat=request.dest_lat,
            dest_lng=request.dest_lng,
            optimize_for=request.optimize_for,
            user_context=request.user_context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/map/traffic-prediction")
async def traffic_prediction(request: TrafficPredictionRequest):
    """Get AI traffic prediction"""
    try:
        # Get base route
        route = await map_service.get_osrm_route(
            request.start_lat, request.start_lng,
            request.dest_lat, request.dest_lng
        )
        
        # Get real-time data
        realtime_data = await map_service.get_realtime_data(
            request.start_lat, request.start_lng,
            request.dest_lat, request.dest_lng
        )
        
        # Get prediction
        prediction = await map_service.predict_traffic_with_ai(route, realtime_data)
        
        return {
            "success": True,
            "prediction": prediction,
            "hours_ahead": request.hours_ahead,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/map/incident-analysis")
async def incident_analysis(start_lat: float, start_lng: float, radius_km: float = 5):
    """Analyze incidents in an area with AI"""
    try:
        # Get simulated incidents
        incidents = await map_service.get_simulated_incidents(
            start_lat - 0.01, start_lng - 0.01,
            start_lat + 0.01, start_lng + 0.01
        )
        
        # Add AI analysis if available
        analysis = {}
        if map_service.client:
            prompt = f"""
            Analyze these incidents in Calapan City:
            Location: {start_lat}, {start_lng}
            Radius: {radius_km} km
            
            Provide:
            1. Pattern analysis
            2. Risk assessment
            3. Recommendations for responders
            
            Format as JSON with: pattern_analysis, risk_score (1-10), recommendations
            """
            
            try:
                response = map_service.client.models.generate_content(
                    model=map_service.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=500,
                        response_mime_type="application/json"
                    )
                )
                analysis = json.loads(response.text)
            except:
                analysis = {"error": "AI analysis failed"}
        
        return {
            "success": True,
            "incidents": incidents,
            "ai_analysis": analysis,
            "total_incidents": len(incidents),
            "radius_km": radius_km,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map/weather-insights")
async def weather_insights(lat: float, lng: float):
    """Get AI-powered weather insights"""
    try:
        weather = await map_service.get_simulated_weather(lat, lng)
        
        insights = {}
        if map_service.client:
            prompt = f"""
            Weather in Calapan City at {lat}, {lng}:
            Condition: {weather.get('condition')}
            Temperature: {weather.get('temperature')}°C
            Humidity: {weather.get('humidity')}%
            
            Impact on:
            1. Road safety
            2. Emergency response
            3. Traffic flow
            4. Recommended precautions
            
            Format as JSON with: road_safety_impact, emergency_response_impact, 
            traffic_impact, precautions, overall_risk (1-10)
            """
            
            try:
                response = map_service.client.models.generate_content(
                    model=map_service.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=600,
                        response_mime_type="application/json"
                    )
                )
                insights = json.loads(response.text)
            except:
                insights = {"error": "AI insights failed"}
        
        return {
            "success": True,
            "current_weather": weather,
            "ai_insights": insights,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/map/emergency-facilities")
async def emergency_facilities(lat: float, lng: float, radius_km: float = 3):
    """Find nearby emergency facilities with AI analysis"""
    try:
        # Simulated facilities (replace with actual data)
        facilities = [
            {
                "type": "hospital",
                "name": "Oriental Mindoro Provincial Hospital",
                "distance_km": random.uniform(0.5, 3),
                "estimated_time_min": random.randint(5, 15),
                "contact": "(043) 288-2000"
            },
            {
                "type": "fire_station",
                "name": "Calapan City Fire Station",
                "distance_km": random.uniform(1, 4),
                "estimated_time_min": random.randint(8, 20),
                "contact": "288-3333"
            },
            {
                "type": "police_station",
                "name": "Calapan City Police Station",
                "distance_km": random.uniform(0.5, 2.5),
                "estimated_time_min": random.randint(5, 12),
                "contact": "288-4444"
            }
        ]
        
        # Sort by distance
        facilities.sort(key=lambda x: x["distance_km"])
        
        # Add AI recommendations if available
        recommendations = []
        if map_service.client:
            prompt = f"""
            For location {lat}, {lng} in Calapan City with {radius_km}km radius:
            Nearby emergency facilities: {json.dumps(facilities, indent=2)}
            
            Provide recommendations:
            1. Which facility to use for different emergencies
            2. Best routes to each
            3. Contact procedures
            4. Estimated response times
            
            Format as JSON with: facility_recommendations, emergency_types, 
            contact_procedures, estimated_response_times
            """
            
            try:
                response = map_service.client.models.generate_content(
                    model=map_service.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800,
                        response_mime_type="application/json"
                    )
                )
                recommendations = json.loads(response.text)
            except:
                recommendations = {"error": "AI recommendations failed"}
        
        return {
            "success": True,
            "facilities": facilities,
            "ai_recommendations": recommendations,
            "your_location": {"lat": lat, "lng": lng},
            "radius_km": radius_km,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== REAL-TIME ROUTING ENDPOINTS ==========
@app.post("/api/route/live-traffic")
async def get_live_traffic(request: LiveTrafficRequest):
    """Get live traffic data for a route"""
    try:
        result = realtime_service.get_live_traffic(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            dest_lat=request.dest_lat,
            dest_lng=request.dest_lng
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/route/alternatives")
async def get_alternative_routes(request: AlternativesRequest):
    """Find better alternative routes"""
    try:
        result = realtime_service.get_alternative_routes(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            dest_lat=request.dest_lat,
            dest_lng=request.dest_lng,
            current_route_time=request.current_route_time
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/route/eta")
async def calculate_current_eta(request: ETARequest):
    """Calculate current ETA with live traffic"""
    try:
        result = realtime_service.calculate_current_eta(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            dest_lat=request.dest_lat,
            dest_lng=request.dest_lng,
            original_duration=request.original_duration,
            traffic_level=request.traffic_level
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/route/alerts")
async def get_traffic_alerts(request: AlertsRequest):
    """Get traffic alerts for a route"""
    try:
        result = realtime_service.get_traffic_alerts(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            dest_lat=request.dest_lat,
            dest_lng=request.dest_lng,
            radius_km=request.radius_km
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/route/update-position")
async def update_user_position(request: UserPositionRequest):
    """Update user position for real-time tracking"""
    try:
        result = realtime_service.update_user_position(
            user_id=request.user_id,
            lat=request.lat,
            lng=request.lng,
            accuracy=request.accuracy
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/route/nearby-users")
async def get_nearby_users(lat: float, lng: float, radius_km: float = 5):
    """Get nearby users for social features"""
    try:
        nearby_users = realtime_service.get_nearby_users(lat, lng, radius_km)
        
        return {
            'success': True,
            'nearby_users': nearby_users,
            'total_nearby': len(nearby_users),
            'your_position': {'lat': lat, 'lng': lng}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/route/health")
async def route_service_health():
    """Check routing service health"""
    try:
        # Test OSRM connection
        test_url = "http://router.project-osrm.org/route/v1/driving/121.181,13.411;121.201,13.415"
        response = requests.get(test_url, timeout=5)
        
        return {
            'status': 'healthy',
            'osrm_connected': response.status_code == 200,
            'realtime_service': 'running',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'degraded',
            'osrm_connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# ========== WEBSOCKET ENDPOINTS ==========
@app.websocket("/ws/route/{user_id}")
async def websocket_route_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time route updates"""
    token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=1008, reason="Token required")
        return
    
    try:
        payload = decode_token(token)
        token_user_id = payload.get("sub")
        
        if token_user_id != user_id:
            await websocket.close(code=1008, reason="Invalid token for user")
            return
        
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await connection_manager.connect(websocket, user_id)
    
    try:
        await connection_manager.send_personal_message({
            "type": "connected",
            "message": "Connected to real-time route updates",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }, user_id)
        
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
                
                message_type = data.get("type")
                
                if message_type == "position_update":
                    lat = data.get("lat")
                    lng = data.get("lng")
                    accuracy = data.get("accuracy")
                    
                    if lat and lng:
                        connection_manager.update_user_position(user_id, lat, lng, accuracy)
                        
                        await connection_manager.send_personal_message({
                            "type": "position_updated",
                            "lat": lat,
                            "lng": lng,
                            "timestamp": datetime.now().isoformat()
                        }, user_id)
                
                elif message_type == "route_update":
                    route_id = data.get("route_id")
                    route_data = data.get("data")
                    
                    connection_manager.route_updates[route_id] = {
                        **route_data,
                        "user_id": user_id,
                        "updated_at": datetime.now().isoformat()
                    }
                
                elif message_type == "ping":
                    await connection_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, user_id)
                
                elif message_type == "subscribe_traffic":
                    route_id = data.get("route_id")
                    
                    asyncio.create_task(
                        send_traffic_updates(user_id, route_id)
                    )
                    
                    await connection_manager.send_personal_message({
                        "type": "subscribed",
                        "route_id": route_id,
                        "message": f"Subscribed to traffic updates for route {route_id}"
                    }, user_id)
                
                elif message_type == "get_nearby_users":
                    lat = data.get("lat")
                    lng = data.get("lng")
                    radius_km = data.get("radius_km", 5)
                    
                    if lat and lng:
                        nearby_users = connection_manager.get_nearby_users(lat, lng, radius_km)
                        await connection_manager.send_personal_message({
                            "type": "nearby_users",
                            "nearby": nearby_users,
                            "your_position": {"lat": lat, "lng": lng}
                        }, user_id)
                
                elif message_type == "get_ai_insights":
                    # New: Get AI insights via WebSocket
                    lat = data.get("lat")
                    lng = data.get("lng")
                    route_id = data.get("route_id")
                    
                    if lat and lng and map_service.client:
                        insights = await map_service.generate_ai_insights(
                            {"distance": 5000, "duration": 600},
                            {"traffic_level": "medium", "time_of_day": datetime.now().hour},
                            "fastest"
                        )
                        
                        await connection_manager.send_personal_message({
                            "type": "ai_insights",
                            "route_id": route_id,
                            "insights": insights,
                            "timestamp": datetime.now().isoformat()
                        }, user_id)
                
            except asyncio.TimeoutError:
                await connection_manager.send_personal_message({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }, user_id)
                
            except Exception as e:
                print(f"WebSocket error for user {user_id}: {e}")
                break
    
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket, user_id)

async def send_traffic_updates(user_id: str, route_id: str):
    """Send periodic traffic updates to subscribed users"""
    update_count = 0
    
    while update_count < 100:
        try:
            if user_id not in connection_manager.active_connections:
                break
            
            # Use Gemini for traffic predictions if available
            traffic_prediction = {}
            if map_service.client:
                try:
                    prediction = await map_service.predict_traffic_with_ai(
                        {"distance": 5000, "duration": 600},
                        {"traffic_level": random.choice(["low", "medium", "high"])}
                    )
                    traffic_prediction = prediction
                except:
                    pass
            
            traffic_data = {
                "route_id": route_id,
                "traffic_level": random.choice(["low", "medium", "high"]),
                "average_speed": random.randint(20, 80),
                "estimated_delay": random.randint(0, 300),
                "timestamp": datetime.now().isoformat(),
                "update_number": update_count + 1,
                "ai_prediction": traffic_prediction if traffic_prediction else None
            }
            
            await connection_manager.send_personal_message({
                "type": "traffic_update",
                "data": traffic_data
            }, user_id)
            
            update_count += 1
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Traffic update error for user {user_id}: {e}")
            break

@app.get("/api/websocket/status")
async def get_websocket_status():
    """Get WebSocket connection status"""
    return {
        "total_connections": len(connection_manager.active_connections),
        "connected_users": list(connection_manager.active_connections.keys()),
        "user_positions": connection_manager.user_positions,
        "timestamp": datetime.now().isoformat()
    }

# ========== MAP AI STATUS ==========
@app.get("/api/map/ai-status")
async def get_ai_status():
    """Check AI map service status"""
    return {
        "gemini_available": map_service.client is not None,
        "model": map_service.model if map_service.client else "none",
        "features_available": [
            "ai_route_analysis",
            "traffic_prediction",
            "incident_analysis",
            "weather_insights",
            "emergency_facilities"
        ],
        "timestamp": datetime.now().isoformat()
    }

# ========== TEST ENDPOINTS ==========
@app.get("/api/test/gemini")
async def test_gemini():
    """Test Gemini AI connection"""
    try:
        if not map_service.client:
            return {
                "success": False,
                "message": "Gemini not configured. Check GEMINI_API_KEY in .env",
                "timestamp": datetime.now().isoformat()
            }
        
        # Simple test
        test_response = map_service.client.models.generate_content(
            model=map_service.model,
            contents="Say 'Gemini is working with RESQAPP'"
        )
        
        return {
            "success": True,
            "message": "Gemini AI is connected and working",
            "test_response": test_response.text,
            "model": map_service.model,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Gemini test failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# ========== INCIDENT ML SERVICE (REMOVED OLD ENHANCED ML) ==========
# The old ml_service = EnhancedIncidentMLService(...) line has been removed.
# We now use services.predictor exclusively.

# ========== INCIDENT REPORTING ENDPOINTS ==========
from datetime import datetime

@app.post("/api/reports/submit", response_model=MLReportAnalysisResponse)
async def submit_incident_report(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        form = await request.form()
        
        description = form.get("description")
        latitude = float(form.get("latitude"))
        longitude = float(form.get("longitude"))
        contact_number = form.get("contact_number")
        barangay = form.get("barangay")
        address = form.get("address")
        emergency_contact = form.get("emergency_contact")
        
        if not description or not latitude or not longitude or not barangay:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        IMAGE_DIR = "uploads/images"
        VIDEO_DIR = "uploads/videos"
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        
        # Store both relative (for frontend) and absolute (for ML) paths
        image_paths = []       # relative URLs for frontend
        video_paths = []       # relative URLs for frontend
        image_abs_paths = []   # absolute paths for analysis
        video_abs_paths = []   # absolute paths for analysis
        
        # Find all file indices
        file_indices = set()
        for key in form.keys():
            if key.startswith("file_"):
                try:
                    idx = int(key.split("_")[1])
                    file_indices.add(idx)
                except:
                    continue
        
        for idx in sorted(file_indices):
            file_key = f"file_{idx}"
            file_obj = form.get(file_key)
            if not file_obj:
                continue
            
            file_type_key = f"file_type_{idx}"
            file_type = form.get(file_type_key, "image")
            if file_type not in ["image", "video"]:
                file_type = "image"
            
            ext = os.path.splitext(file_obj.filename)[1]
            if not ext:
                ext = ".jpg" if file_type == "image" else ".mp4"
            # 👇 ADD THIS BLOCK
            if ext == ".jfif":
                ext = ".jpg"   # Rename to .jpg since it's the same format
            filename = f"{uuid.uuid4()}{ext}"
            
            if file_type == "image":
                rel_path = f"/{IMAGE_DIR}/{filename}"
                abs_path = os.path.abspath(rel_path.lstrip('/'))  # convert to absolute
                image_paths.append(rel_path)
                image_abs_paths.append(abs_path)
                with open(abs_path, "wb") as f:
                    shutil.copyfileobj(file_obj.file, f)
            else:
                rel_path = f"/{VIDEO_DIR}/{filename}"
                abs_path = os.path.abspath(rel_path.lstrip('/'))
                video_paths.append(rel_path)
                video_abs_paths.append(abs_path)
                with open(abs_path, "wb") as f:
                    shutil.copyfileobj(file_obj.file, f)
        
        # ========== ML PREDICTIONS ==========
        text_pred = None
        image_pred = None
        video_pred = None
        
        try:
            # 1. Text analysis (always works)
            text_pred = predict_text(description)
            print(f"✅ Text prediction: {text_pred}")
            
            # 2. Image analysis – use absolute path
            if image_abs_paths:
                first_abs = image_abs_paths[0]
                print(f"📁 Analysing image: {first_abs} (exists: {os.path.exists(first_abs)})")
                if os.path.exists(first_abs):
                    image_pred = analyze_image(first_abs)
                    print(f"✅ Image analysis result: {image_pred}")
                else:
                    print(f"❌ Image file not found at {first_abs}")
            
            # 3. Video analysis – use absolute path
            if video_abs_paths:
                first_vid_abs = video_abs_paths[0]
                print(f"📁 Analysing video: {first_vid_abs} (exists: {os.path.exists(first_vid_abs)})")
                if os.path.exists(first_vid_abs):
                    video_pred = analyze_video(first_vid_abs)
                    print(f"✅ Video analysis: {video_pred}")
                else:
                    print(f"❌ Video file not found at {first_vid_abs}")
        
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback text (ensures at least text analysis exists)
            text_pred = {
                "incident_type": "Other",
                "severity": "medium",
                "type_confidence": 0.5,
                "severity_confidence": 0.5,
                "all_type_scores": {},
                "all_severity_scores": {}
            }
            image_pred = None
            video_pred = None
        
        # Build incident data
        report_data = {
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "contact_number": contact_number,
            "barangay": barangay,
            "address": address,
            "emergency_contact": emergency_contact,
            "user_id": str(current_user.id)
        }
        
        # Create incident with ML results
        incident = crud_incidents.create_incident_report(
            db=db,
            user_id=current_user.id,
            incident_data=report_data,
            ml_analysis={
                "type": text_pred["incident_type"],
                "severity": text_pred["severity"],
                "confidence": text_pred["type_confidence"],
                "analysis": text_pred,
                "recommendations": []
            },
            image_paths=image_paths,
            video_paths=video_paths,
            image_analysis=image_pred,
            video_analysis=video_pred,
            text_analysis=text_pred
        )
        
        # Background task (optional)
        background_tasks.add_task(
            process_incident_background,
            incident.id,
            text_pred,
        )
        
        # Build response
        response = MLReportAnalysisResponse(
            success=True,
            report_id=incident.id,
            incident_type=text_pred["incident_type"],
            severity=text_pred["severity"],
            priority=3,
            confidence=text_pred["type_confidence"],
            analysis={
                "text_analysis": text_pred,
                "image_analysis": image_pred,
                "video_analysis": video_pred
            },
            location={
                "lat": latitude,
                "lng": longitude,
                "barangay": barangay,
                "address": address
            },
            recommendations=[],
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print("🔥 Unhandled exception in /api/reports/submit:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(e)
            }
        )

# ─── UPDATED: Text-only ML analysis using predictor ───
@app.post("/api/ml/analyze-text", response_model=MLTextAnalysisResponse)
async def analyze_text(text: str = Form(...)):
    try:
        # Use the predictor directly
        result = predict_text(text)
        # result has keys: incident_type, severity, type_confidence, severity_confidence,
        #                   all_type_scores (dict), all_severity_scores (dict)
        
        # Convert all_type_scores dict to a list in a fixed order for the response
        type_order = ["Accident", "Fire", "Medical", "Crime", "Natural Disaster", "Infrastructure", "Other"]
        probs_list = [result["all_type_scores"].get(t, 0.0) for t in type_order]
        
        # Build response matching MLTextAnalysisResponse schema
        return {
            "type": result["incident_type"],
            "confidence": result["type_confidence"],
            "severity": result["severity"],
            "severity_confidence": result["severity_confidence"],
            "all_predictions": probs_list,
            "keywords": [] 
        }
    except Exception as e:
        print(f"🔥 ML analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/incident-types")
async def get_incident_types():
    """
    Get list of incident types that the ML model can detect
    """
    return {
        "incident_types": [
            "Accident",
            "Fire", 
            "Medical",
            "Crime",
            "Natural Disaster",
            "Infrastructure",
            "Other"
        ],
        "description": "AI-powered incident classification types"
    }

# ─── UPDATED: ML status endpoint (no more enhanced ML) ───
@app.get("/api/ml/status")
async def get_ml_service_status():
    """
    Check ML service status
    """
    return {
        "status": "operational",
        "text_classifier": "available (predictor)",
        "image_analyzer": "available (predictor)", 
        "video_analyzer": "available (predictor)",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# ─── REMOVED: /api/ml/train endpoint ───
# Training is not done on Render; you can re-implement with lazy imports if needed.

# ========== OTHER ENDPOINTS ==========
# (All the remaining endpoints are unchanged – they use crud_incidents, ml_analytics, etc.)

@app.get("/api/reports", response_model=List[IncidentReportResponse])
async def get_incidents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    type: Optional[str] = None,
    barangay: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get incidents with optional filters
    """
    incidents = crud_incidents.get_all_incidents(
        db, skip=skip, limit=limit, 
        status=status, incident_type=type, barangay=barangay
    )
    return incidents

@app.get("/api/reports/my", response_model=List[IncidentReportResponse])
async def get_my_incidents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    incidents = crud_incidents.get_user_incidents(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    # Build response manually to include assigned_to
    result = []
    for inc in incidents:
        result.append({
            "id": inc.id,
            "description": inc.description,
            "incident_type": inc.incident_type,
            "severity": inc.severity,
            "priority": inc.priority,
            "ml_confidence": inc.ml_confidence,
            "latitude": inc.latitude,
            "longitude": inc.longitude,
            "barangay": inc.barangay,
            "address": inc.address,
            "contact_number": inc.contact_number,
            "emergency_contact": inc.emergency_contact,
            "image_paths": inc.image_paths,
            "video_paths": inc.video_paths,
            "text_analysis": inc.text_analysis,
            "keywords": inc.keywords,
            "status": inc.status,
            "verified_by": inc.verified_by,
            "assigned_to": inc.assigned_to,   # <-- ADD THIS LINE
            "created_at": inc.created_at,
            "updated_at": inc.updated_at,
            "resolved_at": inc.resolved_at,
            "user_id": inc.user_id,
            "city": inc.city,
            "province": inc.province,
        })
    return result

@app.get("/api/reports/{report_id}", response_model=IncidentReportResponse)
async def get_incident(
    report_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    incident = crud_incidents.get_incident_report(db, report_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    # Convert to dict and ensure assigned_to is included
    return {
        "id": incident.id,
        "description": incident.description,
        "incident_type": incident.incident_type,
        "severity": incident.severity,
        "priority": incident.priority,
        "ml_confidence": incident.ml_confidence,
        "latitude": incident.latitude,
        "longitude": incident.longitude,
        "barangay": incident.barangay,
        "address": incident.address,
        "contact_number": incident.contact_number,
        "emergency_contact": incident.emergency_contact,
        "image_paths": incident.image_paths,
        "video_paths": incident.video_paths,
        "text_analysis": incident.text_analysis,
        "keywords": incident.keywords,
        "status": incident.status,
        "verified_by": incident.verified_by,
        "assigned_to": incident.assigned_to,   # <-- ADD THIS LINE
        "created_at": incident.created_at,
        "updated_at": incident.updated_at,
        "resolved_at": incident.resolved_at,
        "user_id": incident.user_id,
        "city": incident.city,
        "province": incident.province,
    }

@app.put("/api/reports/{report_id}/status")
async def update_status(
    report_id: str,
    status: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update incident status (admin/responder only)
    """
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    incident = crud_incidents.get_incident_report(db, report_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Permission: responder can only update incidents assigned to them
    if current_user.role == "responder" and incident.assigned_to != current_user.id:
        raise HTTPException(status_code=403, detail="Not assigned to you")

    old_status = incident.status
    incident.status = status
    if status == "resolved" and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()
    db.add(incident)

    # If a responder is marking it as resolved, log it
    if status == "resolved" and current_user.role == "responder" and old_status != "resolved":
        resolved_log = models.ResponderResolvedIncident(
            incident_id=report_id,
            responder_id=current_user.id,
            resolved_at=datetime.utcnow(),
            notes=f"Resolved by {current_user.full_name}"
        )
        db.add(resolved_log)

    db.commit()
    return {"success": True, "message": f"Status updated to {status}"}

@app.put("/api/reports/{report_id}/assign")
async def assign_incident(
    report_id: str,
    assignee_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Assign incident to responder (admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    incident = crud_incidents.assign_incident(
        db, report_id, assignee_id, current_user.id
    )
    if not incident:
        raise HTTPException(status_code=404, detail="Incident or assignee not found")
    
    return {"success": True, "message": "Incident assigned successfully"}

@app.get("/api/reports/{report_id}/updates", response_model=List[IncidentUpdateResponse])
async def get_incident_updates(
    report_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get updates for an incident
    """
    incident = crud_incidents.get_incident_report(db, report_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    updates = crud_incidents.get_incident_updates(db, report_id)
    
    # Format response with user names
    response = []
    for update in updates:
        user = db.query(User).filter(User.id == update.user_id).first()
        response.append({
            "id": update.id,
            "update_type": update.update_type,
            "content": update.content,
            "user_name": user.full_name if user else "Unknown",
            "created_at": update.created_at
        })
    
    return response

@app.post("/api/reports/{report_id}/updates")
async def create_incident_update(
    report_id: str,
    update_data: IncidentUpdateCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Add an update to an incident
    """
    incident = crud_incidents.get_incident_report(db, report_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    update = crud_incidents.create_incident_update(
        db, report_id, current_user.id,
        update_data.update_type, update_data.content, update_data.metadata
    )
    
    return {"success": True, "update_id": update.id}

@app.get("/api/emergency/facilities")
async def get_emergency_facilities(
    barangay: Optional[str] = None,
    type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get emergency facilities with filters
    """
    facilities = crud_incidents.get_emergency_facilities(db, barangay, type)
    return {"facilities": facilities}

@app.get("/api/emergency/nearby")
async def get_nearby_facilities(
    lat: float,
    lng: float,
    radius_km: float = 5,
    db: Session = Depends(get_db)
):
    """
    Get emergency facilities near a location
    """
    facilities = crud_incidents.get_nearby_facilities(db, lat, lng, radius_km)
    return {"facilities": facilities}

@app.get("/api/barangays")
async def get_barangays(db: Session = Depends(get_db)):
    """
    Get all barangays
    """
    barangays = crud_incidents.get_barangays(db)
    return {"barangays": barangays}

@app.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get incident statistics
    """
    if current_user.role not in ["admin", "responder"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    stats = crud_incidents.get_statistics(db, days)
    return stats

@app.get("/api/ml/incident-types")
async def get_incident_types(db: Session = Depends(get_db)):
    """
    Get list of incident types from database
    """
    categories = crud_incidents.get_incident_categories(db)
    if categories:
        return {
            "categories": [
                {
                    "name": cat.name,
                    "description": cat.description,
                    "icon": cat.icon,
                    "color": cat.color,
                    "default_priority": cat.default_priority,
                    "severity_weight": cat.severity_weight
                }
                for cat in categories
            ]
        }
    else:
        # Fallback if no categories in database
        return {
            "categories": [
                {
                    "name": "Accident",
                    "description": "Road accidents, vehicle collisions",
                    "icon": "🚗",
                    "color": "#f59e0b",
                    "default_priority": 3,
                    "severity_weight": 1.2
                },
                {
                    "name": "Fire",
                    "description": "Fire incidents, structural fires",
                    "icon": "🔥",
                    "color": "#dc2626",
                    "default_priority": 4,
                    "severity_weight": 1.5
                },
                {
                    "name": "Medical",
                    "description": "Medical emergencies, health crises",
                    "icon": "🏥",
                    "color": "#ef4444",
                    "default_priority": 4,
                    "severity_weight": 1.4
                },
                {
                    "name": "Crime",
                    "description": "Criminal activities, theft, assault",
                    "icon": "🚔",
                    "color": "#374151",
                    "default_priority": 3,
                    "severity_weight": 1.3
                },
                {
                    "name": "Natural Disaster",
                    "description": "Floods, earthquakes, typhoons",
                    "icon": "🌊",
                    "color": "#1d4ed8",
                    "default_priority": 5,
                    "severity_weight": 1.6
                },
                {
                    "name": "Infrastructure",
                    "description": "Road damage, power outages",
                    "icon": "🏗️",
                    "color": "#6b7280",
                    "default_priority": 2,
                    "severity_weight": 1.0
                },
                {
                    "name": "Other",
                    "description": "Other types of incidents",
                    "icon": "📝",
                    "color": "#9ca3af",
                    "default_priority": 2,
                    "severity_weight": 1.0
                }
            ]
        }

# ─── REMOVED: /api/ml/train endpoint ───
# Training is not done on Render; you can re-implement with lazy imports if needed.
# The endpoint has been removed to prevent accidental training on the server.

# ML Analytics Endpoints (using your updated ml_analytics)
@app.get("/api/reports/my-ml-stats")
async def get_my_ml_stats(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get user's ML prediction statistics"""
    stats = ml_analytics.get_user_ml_stats(current_user.id, days, db)
    stats["user_id"] = current_user.id
    stats["timestamp"] = datetime.now().isoformat()
    return stats

@app.get("/api/ml/performance")
async def get_model_performance(days: int = 30):
    """Get model performance metrics"""
    performance = ml_analytics.get_model_performance(days)
    performance["timestamp"] = datetime.now().isoformat()
    return performance

@app.get("/api/datasets/status")
async def get_dataset_status():
    """Get dataset status and storage info"""
    status = ml_analytics.get_dataset_status()
    status["timestamp"] = datetime.now().isoformat()
    return status

@app.post("/api/datasets/generate")
async def generate_synthetic_data(count: int = 100):
    """Generate small synthetic dataset (storage safe)"""
    result = ml_analytics.generate_synthetic_sample(count)
    
    # Save to training_data (lightweight JSON)
    import json
    filename = f"backend/training_data/synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(result["samples"], f, indent=2)
    
    result["saved_to"] = filename
    result["timestamp"] = datetime.now().isoformat()
    return result

# ─── REMOVED: /api/ml/models/status (old enhanced ML status) ───
# Replaced by /api/ml/status above.

app.include_router(admin_router)


@app.post("/api/emergency/anonymous")
async def create_anonymous_emergency(
    request: Request,
    lat: float = Form(...),
    lng: float = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)          # from deps.py – no auth required
):
    # Basic coordinate validation
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    # Ensure the uploaded file is an audio file
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Generate a unique filename
    ext = os.path.splitext(audio.filename)[1] or '.webm'
    filename = f"{uuid.uuid4()}{ext}"
    upload_dir = "uploads/emergencies"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    # Save the audio file
    try:
        contents = await audio.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save audio: {str(e)}")

    # Get client IP (optional)
    client_ip = request.client.host if request.client else None

    # Save metadata to database
    emergency = models.AnonymousEmergency(
        latitude=lat,
        longitude=lng,
        audio_path=file_path,
        ip_address=client_ip
    )
    db.add(emergency)
    db.commit()
    db.refresh(emergency)

    # Generate public URL for the audio file
    audio_url = f"/uploads/emergencies/{filename}"

    return {
        "success": True,
        "id": emergency.id,
        "audio_url": audio_url,
        "timestamp": emergency.timestamp.isoformat()
    }
    
@app.get("/api/alerts/active", response_model=List[AlertResponse])
async def get_active_alerts(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=24)
    alerts = db.query(Alert).filter(Alert.created_at >= cutoff).order_by(Alert.created_at.desc()).all()
    return alerts
    
 
from fastapi import Depends, HTTPException, status
from deps import get_current_user  # adjust import based on your actual structure

async def get_current_admin(current_user: User = Depends(get_current_user)):
    """Ensure the current user has admin privileges."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Admin ML analytics endpoints (add after other admin routes)
@app.get("/admin/ml-analytics/{user_id}")
async def get_ml_analytics(
    user_id: int,
    days: int = 30,
    current_user: User = Depends(get_current_admin)
):
    """
    Get ML analytics for a specific user (admin only).
    """
    db = SessionLocal()
    try:
        stats = ml_analytics.get_user_ml_stats(user_id, days, db)
        return stats
    finally:
        db.close()

@app.get("/admin/ml-performance")
async def get_ml_performance(
    current_user: User = Depends(get_current_admin)
):
    """
    Get global ML model performance metrics.
    """
    performance = ml_analytics.get_model_performance()
    return performance

@app.get("/admin/dataset-status")
async def get_dataset_status(
    current_user: User = Depends(get_current_admin)
):
    """
    Get dataset storage status.
    """
    status = ml_analytics.get_dataset_status()
    return status

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Public endpoint for users/responders (read-only, active only)
@app.get("/api/legal-compliances", response_model=List[LegalComplianceResponse])
async def get_public_legal_compliances(
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from models import LegalCompliance
    query = db.query(LegalCompliance).filter(LegalCompliance.is_active == True)
    if category:
        query = query.filter(LegalCompliance.category == category)
    return query.order_by(LegalCompliance.created_at.desc()).all()

@app.get("/api/responder/history")
async def get_responder_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "responder":
        raise HTTPException(status_code=403, detail="Only responders can access their history")

    incidents = db.query(IncidentReport).join(
        ResponderResolvedIncident,
        ResponderResolvedIncident.incident_id == IncidentReport.id
    ).filter(
        ResponderResolvedIncident.responder_id == current_user.id
    ).order_by(
        ResponderResolvedIncident.resolved_at.desc()
    ).all()

    # Manual conversion to dict with all fields
    result = []
    for inc in incidents:
        result.append({
            "id": inc.id,
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
            "user": {
                "full_name": inc.reporter.full_name if inc.reporter else None
            } if inc.reporter else None
        })
    return result

# ========== NEW PROFILE ROUTES (Reliable) ==========
@app.get("/api/profile", response_model=UserProfileOut)
def get_my_profile(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """Get full profile of the currently authenticated user."""
    payload = decode_token(credentials.credentials)
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/api/profile", response_model=UserProfileOut)
def update_my_profile(
    payload: UserProfileUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """Update profile of the currently authenticated user."""
    token_data = decode_token(credentials.credentials)
    user_id = int(token_data.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user

@app.post("/api/profile/avatar")
def upload_profile_avatar(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """Upload avatar image for the authenticated user."""
    payload = decode_token(credentials.credentials)
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    ext = file.filename.split(".")[-1]
    filename = f"user_{user.id}.{ext}"
    file_path = f"{UPLOAD_DIR}/{filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    user.profile_photo = file_path
    db.commit()
    
    return {"profile_photo": file_path}


logging.basicConfig(level=logging.INFO)

@app.get("/api/responder/location/{responder_id}")
async def get_responder_location(
    responder_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Verify the user has an incident assigned to this responder
        incident = db.query(IncidentReport).filter(
            IncidentReport.user_id == current_user.id,
            IncidentReport.assigned_to == responder_id
        ).first()
        
        if not incident:
            raise HTTPException(
                status_code=403,
                detail=f"No incident found for user {current_user.id} assigned to responder {responder_id}"
            )
        
        # Fetch responder's last known location
        loc = db.query(ResponderLocation).filter(
            ResponderLocation.responder_id == responder_id
        ).first()
        
        if not loc:
            return {
                "exists": False,
                "message": "Responder has not shared location yet"
            }
        
        # Build response safely
        response = {
            "exists": True,
            "responder_id": responder_id,
            "lat": loc.latitude,
            "lng": loc.longitude,
            "accuracy": loc.accuracy,
        }
        # Safely handle updated_at (if it's None, set to None)
        if loc.updated_at:
            response["updated_at"] = loc.updated_at.isoformat()
        else:
            response["updated_at"] = None
        
        return response
        
    except Exception as e:
        logging.error(f"Error in get_responder_location: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))