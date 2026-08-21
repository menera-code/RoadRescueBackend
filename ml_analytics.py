import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from database import SessionLocal
from models import IncidentReport, TrainingDataset, MLModelVersion

# Production ML inference comes from the single predictor service.
# This replaces the old EnhancedIncidentMLService / EnhancedIncidentClassifier.
from services.predictor import predict_text


class MLAnalyticsService:
    """
    Database-backed analytics for RoadRescue's production ML predictor.

    ML inference is handled by services.predictor:
        - predict_text()
        - analyze_image()
        - analyze_video()

    This module only:
        1. Reads prediction history from the database.
        2. Reports model/training metadata stored in the database.
        3. Reports dataset/storage status.
        4. Generates small synthetic prediction samples for testing.
    """

    def get_user_ml_stats(
        self,
        user_id: int,
        days: int = 30,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get ML prediction statistics for a specific user."""
        close_db = False

        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            incidents = db.query(
                IncidentReport.incident_type,
                IncidentReport.severity,
                IncidentReport.ml_confidence,
                IncidentReport.created_at
            ).filter(
                IncidentReport.user_id == user_id,
                IncidentReport.created_at >= cutoff,
                IncidentReport.ml_confidence.isnot(None)
            ).all()

            stats = {
                "total_predictions": len(incidents),
                "by_type": {},
                "by_severity": {},
                "avg_confidence": 0.0,
                "trend_data": [],
                "days_analyzed": days,
                "top_predictions": {}
            }

            confidences = []
            weekly_stats = {}

            for inc in incidents:
                inc_type = inc.incident_type or "Unknown"
                stats["by_type"][inc_type] = (
                    stats["by_type"].get(inc_type, 0) + 1
                )

                severity = inc.severity or "Unknown"
                stats["by_severity"][severity] = (
                    stats["by_severity"].get(severity, 0) + 1
                )

                if inc.ml_confidence is not None:
                    confidences.append(inc.ml_confidence)

                if inc.created_at:
                    week = inc.created_at.strftime("%Y-W%U")
                    weekly_stats[week] = weekly_stats.get(week, 0) + 1

            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)

            stats["trend_data"] = sorted(
                weekly_stats.items(),
                key=lambda x: x[0]
            )

            stats["top_predictions"] = dict(
                sorted(
                    stats["by_type"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            )

            return stats

        finally:
            if close_db:
                db.close()

    def get_model_performance(self, db: Session = None) -> Dict[str, Any]:
        """
        Return model metadata stored in the database.

        The previous implementation exposed the EnhancedIncidentMLService's
        in-memory training buffer. That no longer exists because production
        inference is handled by services.predictor.
        """
        close_db = False

        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            subquery = db.query(
                MLModelVersion.model_type,
                func.max(MLModelVersion.created_at).label("latest_created")
            ).filter(
                MLModelVersion.is_production == True
            ).group_by(
                MLModelVersion.model_type
            ).subquery()

            latest_models = db.query(MLModelVersion).join(
                subquery,
                and_(
                    MLModelVersion.model_type == subquery.c.model_type,
                    MLModelVersion.created_at == subquery.c.latest_created
                )
            ).all()

            result = {}

            for model in latest_models:
                result[model.model_type] = {
                    "version": model.version,
                    "accuracy": model.accuracy or 0.0,
                    "precision": model.precision or 0.0,
                    "recall": model.recall or 0.0,
                    "f1_score": model.f1_score or 0.0,
                    "training_samples": model.training_samples or 0,
                    "validation_samples": model.validation_samples or 0,
                    "trained_at": (
                        model.created_at.isoformat()
                        if model.created_at
                        else None
                    )
                }

            result["_meta"] = {
                "total_training_runs": db.query(MLModelVersion).count(),
                "inference_service": "services.predictor",
                "predictor_models": {
                    "text": "facebook/bart-large-mnli",
                    "image_video": "yolov8n.pt"
                }
            }

            return result

        finally:
            if close_db:
                db.close()

    def get_dataset_status(self, db: Session = None) -> Dict[str, Any]:
        """Return storage information and training dataset counts."""
        close_db = False

        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))

            incident_images = os.path.join(
                base_dir, "datasets/incident_images"
            )
            training_data_dir = os.path.join(
                base_dir, "training_data"
            )
            uploads_dir = os.path.join(
                base_dir, "uploads"
            )

            status = {
                "incident_images": self._dir_info(incident_images),
                "training_data": self._dir_info(training_data_dir),
                "uploads": self._dir_info(uploads_dir),
                "total_size_mb": 0,
                "storage_warning": False
            }

            verified = db.query(TrainingDataset).filter(
                TrainingDataset.is_verified == True
            ).count()

            used = db.query(TrainingDataset).filter(
                TrainingDataset.used_in_training == True
            ).count()

            status["verified_samples"] = verified
            status["used_in_training"] = used

            total_mb = (
                status["incident_images"]["size_mb"]
                + status["training_data"]["size_mb"]
                + status["uploads"]["size_mb"]
            )

            status["total_size_mb"] = total_mb
            status["storage_warning"] = total_mb > 8000

            return status

        finally:
            if close_db:
                db.close()

    def get_training_data_status(self, db: Session = None) -> Dict[str, Any]:
        """Return counts of verified and used training samples."""
        close_db = False

        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            verified = db.query(TrainingDataset).filter(
                TrainingDataset.is_verified == True
            ).count()

            used = db.query(TrainingDataset).filter(
                TrainingDataset.used_in_training == True
            ).count()

            return {
                "verified_samples": verified,
                "used_in_training": used
            }

        finally:
            if close_db:
                db.close()

    def get_training_stats(
        self,
        days: int = 30,
        db: Session = None
    ) -> Dict[str, Any]:
        """Return summary statistics for recent ML predictions."""
        close_db = False

        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            incidents = db.query(
                IncidentReport.incident_type,
                IncidentReport.severity,
                IncidentReport.ml_confidence
            ).filter(
                IncidentReport.created_at >= cutoff,
                IncidentReport.ml_confidence.isnot(None)
            ).all()

            by_type = {}
            by_severity = {}
            confidences = []

            for inc in incidents:
                incident_type = inc.incident_type or "Unknown"
                by_type[incident_type] = (
                    by_type.get(incident_type, 0) + 1
                )

                severity = inc.severity or "Unknown"
                by_severity[severity] = (
                    by_severity.get(severity, 0) + 1
                )

                if inc.ml_confidence is not None:
                    confidences.append(inc.ml_confidence)

            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences else 0.0
            )

            return {
                "total_predictions": len(incidents),
                "avg_confidence": avg_confidence,
                "by_type": by_type,
                "by_severity": by_severity,
                "days_analyzed": days
            }

        finally:
            if close_db:
                db.close()

    def _dir_info(self, path: str) -> Dict[str, Any]:
        """Return existence, file count, and size in MB for a directory."""
        info = {
            "exists": os.path.exists(path),
            "file_count": 0,
            "size_mb": 0.0
        }

        if os.path.exists(path) and os.path.isdir(path):
            try:
                files = [
                    f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                ]

                info["file_count"] = len(files)

                total = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in files
                )

                info["size_mb"] = round(
                    total / (1024 * 1024),
                    1
                )

            except OSError:
                pass

        return info

    def generate_synthetic_sample(
        self,
        count: int = 100
    ) -> Dict[str, Any]:
        """
        Generate synthetic prediction results using the production predictor.

        Note:
        predictor.predict_text() intentionally returns incident_type='Accident',
        so the original synthetic true labels are retained only as test labels.
        """
        synthetic = []

        templates = {
            "Accident": [
                "Car crash on highway",
                "Motorcycle accident",
                "Pedestrian collision"
            ],
            "Fire": [
                "House fire",
                "Vehicle fire",
                "Building blaze"
            ],
            "Medical": [
                "Heart attack",
                "Unconscious person",
                "Injury from fall"
            ]
        }

        for _ in range(count):
            true_label = random.choice(list(templates.keys()))
            text = random.choice(templates[true_label])

            prediction = predict_text(text)

            synthetic.append({
                "text": text,
                "predicted_type": prediction.get(
                    "incident_type",
                    "Unknown"
                ),
                "confidence": prediction.get(
                    "severity_confidence",
                    0.0
                ),
                "severity": prediction.get(
                    "severity",
                    "Unknown"
                ),
                "mentioned_vehicles": prediction.get(
                    "mentioned_vehicles",
                    []
                ),
                "true_label": true_label
            })

        serialized = json.dumps(synthetic)
        size_bytes = len(serialized.encode("utf-8"))

        return {
            "success": True,
            "samples": synthetic,
            "size_bytes": size_bytes,
            "storage_mb": round(
                size_bytes / (1024 * 1024),
                2
            )
        }


# Global instance
ml_analytics = MLAnalyticsService()