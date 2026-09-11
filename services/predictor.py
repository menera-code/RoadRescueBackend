import os
import cv2
import tempfile
import json
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

import onnxruntime as ort


# ==========================================
# GEMINI TEXT CLASSIFIER (with vehicle mentions)
# ==========================================
def predict_text_with_gemini(text: str) -> Dict[str, Any]:
    try:
        from gemini_map_service import map_service
        import google.genai as genai

        if map_service.client is None:
            raise Exception("Gemini client not initialized")

        prompt = f"""
        You are an emergency incident classifier. Analyze the following report and return ONLY valid JSON.

        Report: "{text}"

        Determine:
        1. incident_type: choose from ["Accident", "Fire", "Medical", "Crime", "Natural Disaster", "Infrastructure", "Other"]
        2. severity: choose from ["low", "medium", "high", "critical"]
        3. confidence: a number between 0 and 1
        4. keywords: a list of important words (max 5)
        5. mentioned_vehicles: a list of vehicle types mentioned (e.g., ["car", "truck", "motorcycle"])

        Return JSON exactly like:
        {{
            "incident_type": "Accident",
            "severity": "medium",
            "confidence": 0.95,
            "keywords": ["car", "motorcycle", "collision"],
            "mentioned_vehicles": ["car", "motorcycle"]
        }}
        """

        response = map_service.client.models.generate_content(
            model=map_service.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500,
                response_mime_type="application/json"
            )
        )

        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]

        result = json.loads(raw)

        return {
            "incident_type": result.get("incident_type", "Other"),
            "severity": result.get("severity", "medium"),
            "type_confidence": result.get("confidence", 0.5),
            "severity_confidence": result.get("confidence", 0.5),
            "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
            "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS},
            "keywords": result.get("keywords", []),
            "mentioned_vehicles": result.get("mentioned_vehicles", [])
        }
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}, using fallback")
        return simple_keyword_classifier(text)


def simple_keyword_classifier(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    incident_type = "Other"
    severity = "medium"
    confidence = 0.6
    mentioned_vehicles = []

    vehicle_keywords = {
        "car": ["car", "sedan", "suv", "van"],
        "truck": ["truck", "lorry", "dump truck"],
        "motorcycle": ["motorcycle", "motorbike", "bike", "scooter"],
        "bus": ["bus", "minibus"],
        "bicycle": ["bicycle", "bike"],
        "jeepney": ["jeepney", "jeep"],
    }
    for vehicle_type, keywords in vehicle_keywords.items():
        if any(k in text_lower for k in keywords):
            mentioned_vehicles.append(vehicle_type)

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
        "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS},
        "keywords": [],
        "mentioned_vehicles": mentioned_vehicles
    }


def predict_text(text: str) -> Dict[str, Any]:
    if not text or len(text.strip()) < 3:
        return {
            "incident_type": "Other",
            "severity": "medium",
            "type_confidence": 0.5,
            "severity_confidence": 0.5,
            "all_type_scores": {t: 0.0 for t in INCIDENT_TYPES},
            "all_severity_scores": {s: 0.0 for s in SEVERITY_LEVELS},
            "mentioned_vehicles": []
        }
    return predict_text_with_gemini(text)


# ==========================================
# YOLOv8 ONNX SESSION (lazy load)
# ==========================================
_yolo_session = None
_yolo_input_name = None

# YOLOv8 COCO class names (80 classes, index order matches the model)
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def get_yolo_session():
    """Lazy-load the ONNX session on first use (~50 MB instead of ~1 GB with torch)."""
    global _yolo_session, _yolo_input_name
    if _yolo_session is None:
        print("🔄 Loading YOLOv8n ONNX model...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        _yolo_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        _yolo_input_name = _yolo_session.get_inputs()[0].name
        print(f"✅ YOLO ONNX session ready. Input name: {_yolo_input_name}")
    return _yolo_session


# ==========================================
# Preprocessing
# ==========================================
def _preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize for YOLOv8 ONNX input. Returns (1, 3, 640, 640) float32."""
    img = cv2.resize(img_bgr, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
    img = np.expand_dims(img, axis=0)        # add batch dim
    return np.ascontiguousarray(img)


# ==========================================
# NMS (class-aware)
# ==========================================
def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45):
    if len(boxes_xyxy) == 0:
        return []
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ==========================================
# Postprocessing
# ==========================================
def _postprocess(outputs, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """
    YOLOv8 ONNX output shape: (1, 84, 8400)
      rows 0-3: cx, cy, w, h (in 640x640 pixel space)
      rows 4+:  80 class scores (already sigmoid-activated)
    """
    preds = outputs[0][0]          # (84, 8400)
    preds = preds.T                # (8400, 84)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert center-xywh -> corner-xyxy
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    unique_classes = np.unique(class_ids)
    for c in unique_classes:
        cls_mask = class_ids == c
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = _nms(cls_boxes, cls_scores, iou_threshold)
        for k in keep:
            cls_name = YOLO_CLASSES[int(c)] if int(c) < len(YOLO_CLASSES) else "unknown"
            detections.append({
                "object": cls_name,
                "confidence": float(cls_scores[k]),
                "class_id": int(c),
                "bbox": [float(v) for v in cls_boxes[k]]
            })
    return detections


# ==========================================
# HELPER – extract vehicles from detections
# ==========================================
VEHICLE_CLASSES = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'suv', 'van', 'pickup', 'jeep', 'lorry'
}


def extract_vehicles(detections: List[Dict]) -> Dict[str, int]:
    """Count vehicle detections by class name."""
    vehicles = {}
    for d in detections:
        obj = d.get('object', '').lower()
        if obj in VEHICLE_CLASSES:
            vehicles[obj] = vehicles.get(obj, 0) + 1
    return vehicles


# ==========================================
# IMAGE ANALYSIS (ONNX version)
# ==========================================
def analyze_image(image_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            print("⚠️ OpenCV failed, trying PIL fallback...")
            try:
                img = np.array(Image.open(image_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ PIL fallback also failed: {e}")
                return None

        session = get_yolo_session()
        input_tensor = _preprocess_image(img)
        outputs = session.run(None, {_yolo_input_name: input_tensor})
        detections = _postprocess(outputs)

        detection_names = [d["object"].lower() for d in detections]
        incident_type = "Other"
        if any(o in detection_names for o in ["fire", "smoke", "flame"]):
            incident_type = "Fire"
        elif any(o in detection_names for o in ["car", "truck", "bus", "motorcycle", "bicycle", "person"]):
            incident_type = "Accident"
        elif "person" in detection_names:
            incident_type = "Medical"

        vehicles = extract_vehicles(detections)

        return {
            "incident_type": incident_type,
            "detections": detections,
            "vehicles": vehicles,
            "confidence": max([d["confidence"] for d in detections]) if detections else 0.5,
            "total_objects": len(detections)
        }
    except Exception as e:
        print(f"❌ Image analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# VIDEO ANALYSIS (calls analyze_image per frame)
# ==========================================
def analyze_video(video_path: str, sample_frames: int = 5) -> Optional[Dict[str, Any]]:
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        frame_indices = [
            int(i * total_frames / (sample_frames + 1))
            for i in range(1, sample_frames + 1)
        ]
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
            except Exception:
                pass

        cap.release()

        if not all_detections:
            return None

        vehicle_counts = {}
        for d in all_detections:
            obj = d.get('object', '').lower()
            if obj in VEHICLE_CLASSES:
                vehicle_counts[obj] = vehicle_counts.get(obj, 0) + 1

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
            "vehicles": vehicle_counts,
            "total_detections": len(all_detections),
            "frames_analyzed": len(frame_indices)
        }
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        return None


# ==========================================
# CONSTANTS
# ==========================================
INCIDENT_TYPES = [
    "Accident", "Fire", "Medical", "Crime",
    "Natural Disaster", "Infrastructure", "Other"
]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]