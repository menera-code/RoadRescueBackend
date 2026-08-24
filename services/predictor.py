import os
import cv2
import tempfile
import json
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

# ==========================================
# GEMINI TEXT CLASSIFIER (LIGHTWEIGHT)
# ==========================================
_yolo_model = None

def predict_text_with_gemini(text: str) -> Dict[str, Any]:
    """
    Use Gemini AI to classify text – no local model.
    """
    try:
        import google.generativeai as genai
        from gemini_map_service import map_service
        
        if map_service.client is None:
            raise Exception("Gemini client not initialized")
        
        # Use the same model as your map service (from env)
        model = genai.GenerativeModel(map_service.model)
        
        prompt = f"""
        You are an emergency incident classifier. Analyze the following report and return ONLY valid JSON.

        Report: "{text}"

        Determine:
        1. incident_type: choose from ["Accident", "Fire", "Medical", "Crime", "Natural Disaster", "Infrastructure", "Other"]
        2. severity: choose from ["low", "medium", "high", "critical"]
        3. confidence: a number between 0 and 1
        4. keywords: a list of important words (max 5)

        Return JSON exactly like:
        {{
            "incident_type": "Fire",
            "severity": "high",
            "confidence": 0.92,
            "keywords": ["flames", "building", "evacuate"]
        }}
        """
        
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Remove markdown code fences if present
        if raw.startswith('```json'):
            raw = raw[7:]
        if raw.endswith('```'):
            raw = raw[:-3]
        result = json.loads(raw)
        
        return {
            "incident_type": result.get("incident_type", "Other"),
            "severity": result.get("severity", "medium"),
            "type_confidence": result.get("confidence", 0.5),
            "severity_confidence": result.get("confidence", 0.5),
            "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
            "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS},
            "keywords": result.get("keywords", [])
        }
    except Exception as e:
        print(f"⚠️ Gemini classification failed: {e}, using fallback")
        return simple_keyword_classifier(text)

def simple_keyword_classifier(text: str) -> Dict[str, Any]:
    """Fallback using keyword matching – zero memory footprint."""
    text_lower = text.lower()
    incident_type = "Other"
    severity = "medium"
    confidence = 0.6
    
    if any(k in text_lower for k in ["fire", "flame", "smoke", "burn"]):
        incident_type = "Fire"
        severity = "high"
        confidence = 0.7
    elif any(k in text_lower for k in ["car", "crash", "accident", "collision", "hit", "ram"]):
        incident_type = "Accident"
        severity = "high" if "injury" in text_lower else "medium"
        confidence = 0.6
    elif any(k in text_lower for k in ["medical", "heart", "unconscious", "bleed", "ambulance"]):
        incident_type = "Medical"
        severity = "critical" if "unconscious" in text_lower else "high"
        confidence = 0.65
    elif any(k in text_lower for k in ["crime", "theft", "robbery", "assault", "shoot"]):
        incident_type = "Crime"
        severity = "critical" if "shoot" in text_lower else "high"
        confidence = 0.7
    elif any(k in text_lower for k in ["flood", "earthquake", "typhoon", "storm"]):
        incident_type = "Natural Disaster"
        severity = "critical"
        confidence = 0.75
    elif any(k in text_lower for k in ["road damage", "pothole", "broken pipe", "power outage"]):
        incident_type = "Infrastructure"
        severity = "medium"
        confidence = 0.6
    
    return {
        "incident_type": incident_type,
        "severity": severity,
        "type_confidence": confidence,
        "severity_confidence": confidence,
        "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
        "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS}
    }

# ==========================================
# MAIN TEXT PREDICT FUNCTION
# ==========================================
def predict_text(text: str) -> Dict[str, Any]:
    if not text or len(text.strip()) < 3:
        return {
            "incident_type": "Other",
            "severity": "medium",
            "type_confidence": 0.5,
            "severity_confidence": 0.5,
            "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
            "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS}
        }
    return predict_text_with_gemini(text)

# ==========================================
# YOLO LAZY LOADER (unchanged)
# ==========================================
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print("🔄 Loading YOLOv8n model (lazy load triggered)...")
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
        _yolo_model.to('cpu')
        print("✅ YOLO model loaded successfully.")
    return _yolo_model

# ==========================================
# INCIDENT TYPES & SEVERITIES
# ==========================================
INCIDENT_TYPES = [
    "Accident", "Fire", "Medical", "Crime", 
    "Natural Disaster", "Infrastructure", "Other"
]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# ==========================================
# IMAGE & VIDEO ANALYSIS (unchanged)
# ==========================================
def analyze_image(image_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ OpenCV failed, trying PIL fallback...")
            try:
                img = np.array(Image.open(image_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ PIL fallback also failed: {e}")
                return None
        
        model = get_yolo_model()
        results = model(img)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    detections.append({
                        "object": name,
                        "confidence": conf,
                        "class_id": cls
                    })
        
        detection_names = [d["object"].lower() for d in detections]
        incident_type = "Other"
        if any(o in detection_names for o in ["fire", "smoke", "flame"]):
            incident_type = "Fire"
        elif any(o in detection_names for o in ["car", "truck", "bus", "motorcycle", "bicycle", "person"]):
            incident_type = "Accident"
        elif "person" in detection_names:
            incident_type = "Medical"
        
        return {
            "incident_type": incident_type,
            "detections": detections,
            "confidence": max([d["confidence"] for d in detections]) if detections else 0.5,
            "total_objects": len(detections)
        }
    except Exception as e:
        print(f"❌ Image analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_video(video_path: str, sample_frames: int = 5) -> Optional[Dict[str, Any]]:
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        model = get_yolo_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        frame_indices = [int(i * total_frames / (sample_frames + 1)) for i in range(1, sample_frames + 1)]
        all_detections = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            result = analyze_image(temp_path)
            if result and result.get("detections"):
                all_detections.extend(result["detections"])
            
            try:
                os.unlink(temp_path)
            except:
                pass
        
        cap.release()
        
        if not all_detections:
            return None
        
        object_counts = {}
        for d in all_detections:
            name = d["object"]
            object_counts[name] = object_counts.get(name, 0) + 1
        
        incident_type = "Other"
        if "fire" in object_counts or "smoke" in object_counts:
            incident_type = "Fire"
        elif len(object_counts) >= 2:
            incident_type = "Accident"
        elif "person" in object_counts:
            incident_type = "Medical"
        
        return {
            "incident_type": incident_type,
            "object_counts": object_counts,
            "total_detections": len(all_detections),
            "frames_analyzed": len(frame_indices)
        }
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        return None