import os
import cv2
import tempfile
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from PIL import Image

# We will NOT load these at import time
# Instead, we use global variables set to None
_yolo_model = None
_text_classifier = None

# ==========================================
# LAZY LOADERS (Models load only when called)
# ==========================================

def get_yolo_model():
    """Lazy load YOLO model (only when first image/video is analyzed)."""
    global _yolo_model
    if _yolo_model is None:
        print("🔄 Loading YOLOv8n model (lazy load triggered)...")
        from ultralytics import YOLO
        
        # Force CPU usage to save memory on Render
        device = 'cpu'
        _yolo_model = YOLO("yolov8n.pt")
        _yolo_model.to(device)  # Ensure it's on CPU
        print("✅ YOLO model loaded successfully.")
    return _yolo_model

def get_text_classifier():
    """Lazy load BART zero-shot classifier (only when text is analyzed)."""
    global _text_classifier
    if _text_classifier is None:
        print("🔄 Loading BART zero-shot classifier (lazy load triggered)...")
        from transformers import pipeline
        _text_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # Force CPU (-1 means CPU)
            framework='pt'
        )
        print("✅ BART classifier loaded successfully.")
    return _text_classifier

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

# Incident types (consistent with your database)
INCIDENT_TYPES = [
    "Accident", "Fire", "Medical", "Crime", 
    "Natural Disaster", "Infrastructure", "Other"
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

def predict_text(text: str) -> Dict[str, Any]:
    """
    Analyze text description using lazy-loaded BART.
    Returns incident type and severity.
    """
    if not text or len(text.strip()) < 3:
        return {
            "incident_type": "Other",
            "severity": "medium",
            "type_confidence": 0.5,
            "severity_confidence": 0.5,
            "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
            "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS}
        }
    
    classifier = get_text_classifier()  # Loads BART here if not loaded
    
    # 1. Classify incident type
    type_result = classifier(text, candidate_labels=INCIDENT_TYPES)
    predicted_type = type_result['labels'][0]
    type_confidence = type_result['scores'][0]
    all_type_scores = dict(zip(type_result['labels'], type_result['scores']))
    
    # 2. Classify severity
    severity_result = classifier(text, candidate_labels=SEVERITY_LEVELS)
    predicted_severity = severity_result['labels'][0]
    severity_confidence = severity_result['scores'][0]
    all_severity_scores = dict(zip(severity_result['labels'], severity_result['scores']))
    
    return {
        "incident_type": predicted_type,
        "severity": predicted_severity,
        "type_confidence": type_confidence,
        "severity_confidence": severity_confidence,
        "all_type_scores": all_type_scores,
        "all_severity_scores": all_severity_scores
    }

def analyze_image(image_path: str) -> Optional[Dict[str, Any]]:
    """Analyze image using lazy-loaded YOLO – reads image with OpenCV to bypass extension issues."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Try OpenCV first
        img = cv2.imread(image_path)
        
        # If OpenCV fails, try PIL as fallback
        if img is None:
            print(f"⚠️ OpenCV failed, trying PIL fallback...")
            from PIL import Image
            import numpy as np
            try:
                img = np.array(Image.open(image_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # YOLO expects BGR
            except Exception as e:
                print(f"❌ PIL fallback also failed: {e}")
                return None
        
        # Load YOLO lazily
        model = get_yolo_model()
        
        # Run inference on the image array (NOT the file path)
        results = model(img)
        
        # Extract detections
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
        
        # Determine incident type (same logic)
        detection_names = [d["object"].lower() for d in detections]
        incident_type = "Other"
        
        fire_objects = ["fire", "smoke", "flame"]
        accident_objects = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
        
        if any(o in detection_names for o in fire_objects):
            incident_type = "Fire"
        elif any(o in detection_names for o in accident_objects):
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
        return Nones

def analyze_video(video_path: str, sample_frames: int = 5) -> Optional[Dict[str, Any]]:
    """
    Analyze video by sampling frames using lazy-loaded YOLO.
    """
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        model = get_yolo_model()  # Loads YOLO here if not loaded
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames evenly
        frame_indices = [int(i * total_frames / (sample_frames + 1)) for i in range(1, sample_frames + 1)]
        all_detections = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Save frame to temp file for YOLO
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            # Analyze the frame
            result = analyze_image(temp_path)
            if result and result.get("detections"):
                all_detections.extend(result["detections"])
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        cap.release()
        
        # Aggregate detections
        if not all_detections:
            return None
        
        # Count occurrences
        object_counts = {}
        for d in all_detections:
            name = d["object"]
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Determine incident type
        incident_type = "Other"
        if "fire" in object_counts or "smoke" in object_counts:
            incident_type = "Fire"
        elif len(object_counts) >= 2:  # Multiple vehicle types = likely accident
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