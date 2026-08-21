from fastapi import Query
from datetime import datetime, timedelta
import numpy as np
from sklearn.neighbors import KernelDensity

@app.get("/admin/incidents/predict-heatmap")
async def predict_heatmap(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    future_hours: int = Query(6, ge=1, le=72),
    current_admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Returns a GeoJSON feature collection of predicted incident hotspots
    for the given future time window (future_hours from now).
    Uses historical incidents in [start_date, end_date] as training data.
    """
    # 1. Load historical incidents in the period
    incidents = db.query(IncidentReport).filter(
        IncidentReport.created_at >= start_date,
        IncidentReport.created_at <= end_date,
        IncidentReport.status == 'resolved'   # only reliable data
    ).all()
    
    if len(incidents) < 10:
        raise HTTPException(400, "Not enough historical data for prediction")
    
    # 2. Extract coordinates and weights (e.g., severity weight)
    coords = []
    weights = []
    severity_weight = {'low':1, 'medium':2, 'high':4, 'critical':8}
    for inc in incidents:
        coords.append([inc.latitude, inc.longitude])
        weights.append(severity_weight.get(inc.severity, 1))
    
    # 3. Kernel density estimation (weighted)
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    # Convert to radians for haversine
    coords_rad = np.radians(coords)
    kde.fit(coords_rad, sample_weight=weights)
    
    # 4. Generate prediction grid over Calapan area
    lat_grid = np.linspace(13.35, 13.45, 50)
    lng_grid = np.linspace(121.13, 121.23, 50)
    grid_points = np.array([[lat, lng] for lat in lat_grid for lng in lng_grid])
    grid_rad = np.radians(grid_points)
    densities = np.exp(kde.score_samples(grid_rad))
    
    # 5. Normalise and build GeoJSON
    densities = densities / densities.max()
    features = []
    for i, (lat, lng) in enumerate(grid_points):
        if densities[i] > 0.1:   # only show significant hotspots
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {"intensity": float(densities[i])}
            })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "prediction_window_hours": future_hours,
        "based_on_period": f"{start_date} to {end_date}"
    }

