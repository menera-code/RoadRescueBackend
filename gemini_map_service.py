# gemini_map_service.py

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
import google.genai as genai
import requests
import random


load_dotenv()


class GeminiMapService:
    """
    AI-powered map service with Gemini integration.

    Gemini is intentionally limited to ONE request per complete
    AI route analysis.

    Local processing is used for:
    - Traffic prediction
    - Alternative route analysis
    - Route scoring
    - Recommendations
    - Traffic pattern fallback
    """

    def __init__(self):

        # ---------------------------------------------------------
        # Initialize Gemini
        # ---------------------------------------------------------
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            self.client = genai.Client(api_key=api_key)

            self.model = os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash"
            )

            print(
                f"✅ Gemini Map Service initialized "
                f"with model: {self.model}"
            )

        else:
            self.client = None
            self.model = "simulated"

            print(
                "⚠️ GEMINI_API_KEY not found, "
                "using simulated/local AI features"
            )

        # ---------------------------------------------------------
        # External APIs
        # ---------------------------------------------------------
        self.weather_api_key = os.getenv(
            "WEATHER_API_KEY"
        )

        self.traffic_api_key = os.getenv(
            "TRAFFIC_API_KEY",
            "demo"
        )

        # ---------------------------------------------------------
        # Cache
        # ---------------------------------------------------------
        self.route_cache = {}
        self.insights_cache = {}

    # ============================================================
    # MAIN AI ROUTE ANALYSIS
    # ============================================================

    async def get_ai_route_analysis(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        optimize_for: str = "fastest",
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered route analysis.

        Gemini usage:
            MAXIMUM 1 request

        Local processing:
            - realtime data
            - traffic prediction
            - alternative routes
            - route scoring
            - recommendations
        """

        try:

            # -----------------------------------------------------
            # 1. Get primary route from OSRM
            # -----------------------------------------------------
            base_route = await self.get_osrm_route(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng
            )

            # -----------------------------------------------------
            # 2. Get realtime data
            #
            # IMPORTANT:
            # get_realtime_data() no longer calls Gemini.
            # -----------------------------------------------------
            realtime_data = await self.get_realtime_data(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng
            )

            # -----------------------------------------------------
            # 3. ONE Gemini request
            #
            # This is the only Gemini request in the complete
            # route-analysis process.
            # -----------------------------------------------------
            ai_insights = await self.generate_ai_insights(
                base_route,
                realtime_data,
                optimize_for,
                user_context
            )

            # -----------------------------------------------------
            # 4. Alternative routes
            #
            # No Gemini calls.
            # -----------------------------------------------------
            alternatives = await self.get_ai_alternatives(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng,
                base_route,
                realtime_data,
                optimize_for
            )

            # -----------------------------------------------------
            # 5. Local traffic prediction
            #
            # No Gemini calls.
            # -----------------------------------------------------
            traffic_prediction = await self.predict_traffic_with_ai(
                base_route,
                realtime_data
            )

            # -----------------------------------------------------
            # 6. Local recommendations
            # -----------------------------------------------------
            recommendations = await self.generate_recommendations(
                base_route,
                realtime_data,
                ai_insights,
                user_context
            )

            # -----------------------------------------------------
            # 7. Calculate primary route score
            # -----------------------------------------------------
            primary_score = self.calculate_ai_score(
                base_route,
                optimize_for
            )

            # -----------------------------------------------------
            # 8. Calculate confidence
            # -----------------------------------------------------
            confidence = self.calculate_route_confidence(
                base_route,
                realtime_data
            )

            # -----------------------------------------------------
            # 9. Return complete response
            # -----------------------------------------------------
            return {
                "success": True,

                "primary_route": {
                    **base_route,
                    "ai_score": primary_score,
                    "confidence": confidence
                },

                "traffic_prediction": traffic_prediction,

                "ai_insights": ai_insights,

                "alternative_routes": alternatives,

                "recommendations": recommendations,

                "realtime_data": realtime_data,

                "optimization_type": optimize_for,

                "estimated_time_savings": (
                    self.calculate_time_savings(
                        base_route,
                        alternatives
                    )
                ),

                "timestamp": datetime.now().isoformat(),

                "ai_model_used": (
                    self.model
                    if self.client
                    else "local-simulation"
                )
            }

        except Exception as e:

            print(
                f"❌ AI route analysis error: {e}"
            )

            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ============================================================
    # REALTIME DATA
    # ============================================================

    async def get_realtime_data(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> Dict[str, Any]:
        """
        Get realtime traffic, weather and incident data.

        IMPORTANT:
        This function DOES NOT call Gemini.
        """

        try:

            current_hour = datetime.now().hour
            current_day = datetime.now().strftime(
                "%A"
            )

            # -----------------------------------------------------
            # Simulated realtime data
            # -----------------------------------------------------
            realtime_data = {

                "traffic_level": random.choice(
                    [
                        "low",
                        "medium",
                        "high"
                    ]
                ),

                "average_speed": random.randint(
                    20,
                    80
                ),

                "incidents": await self.get_simulated_incidents(
                    start_lat,
                    start_lng,
                    dest_lat,
                    dest_lng
                ),

                "weather": await self.get_simulated_weather(
                    start_lat,
                    start_lng
                ),

                "road_conditions": (
                    self.get_simulated_road_conditions()
                ),

                "time_of_day": current_hour,

                "day_of_week": current_day,

                "last_updated": (
                    datetime.now().isoformat()
                )
            }

            # -----------------------------------------------------
            # NO GEMINI CALL HERE
            #
            # Previously this function called:
            #
            # analyze_traffic_patterns()
            #
            # That has been removed from the normal route flow
            # to reduce Gemini quota usage.
            # -----------------------------------------------------

            return realtime_data

        except Exception as e:

            print(
                f"Realtime data error: {e}"
            )

            return self.get_fallback_realtime_data()

    # ============================================================
    # GEMINI MAIN INSIGHTS
    # ============================================================

    async def generate_ai_insights(
        self,
        route_data: Dict,
        realtime_data: Dict,
        optimize_for: str,
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered insights about the route.

        This is the ONLY normal Gemini request made during
        complete AI route analysis.
        """

        # ---------------------------------------------------------
        # No Gemini configured
        # ---------------------------------------------------------
        if not self.client:

            return self.get_simulated_insights(
                route_data,
                realtime_data,
                optimize_for
            )

        try:

            # -----------------------------------------------------
            # Prepare prompt
            # -----------------------------------------------------
            prompt = f"""
Analyze this route for Calapan City emergency response.

ROUTE INFORMATION:
- Distance: {route_data.get('distance', 0) / 1000:.1f} km
- Estimated Time: {route_data.get('duration', 0) / 60:.0f} minutes
- Optimization Goal: {optimize_for}

REAL-TIME CONDITIONS:
- Traffic Level: {realtime_data.get('traffic_level', 'unknown')}
- Average Speed: {realtime_data.get('average_speed', 0)} km/h
- Time of Day: {realtime_data.get('time_of_day', 12)}:00
- Day of Week: {realtime_data.get('day_of_week', 'Monday')}
- Weather: {realtime_data.get('weather', {}).get('condition', 'Clear')}
- Incidents: {len(realtime_data.get('incidents', []))} reported
- Road Conditions: {json.dumps(realtime_data.get('road_conditions', {}))}

USER CONTEXT:
{json.dumps(user_context or {}, indent=2)}

Analyze the route specifically for emergency response.

Please provide:

1. Safety assessment from 1-10
2. Emergency vehicle accessibility
3. Potential delays and reasons
4. Alternative route suggestions
5. Weather impact
6. Time-sensitive recommendations
7. Confidence score

Return ONLY valid JSON.

Use exactly these keys:

{{
    "safety_score": 1-10,
    "emergency_access": "Good",
    "potential_delays": [],
    "alternatives": [],
    "weather_impact": "",
    "recommendations": [],
    "confidence": 0.0
}}
"""

            # -----------------------------------------------------
            # ONE Gemini request
            # -----------------------------------------------------
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )

            # -----------------------------------------------------
            # Parse JSON
            # -----------------------------------------------------
            if not response or not response.text:

                raise ValueError(
                    "Gemini returned an empty response"
                )

            insights = json.loads(
                response.text
            )

            # -----------------------------------------------------
            # Ensure expected fields exist
            # -----------------------------------------------------
            insights.setdefault(
                "safety_score",
                7
            )

            insights.setdefault(
                "emergency_access",
                "Good"
            )

            insights.setdefault(
                "potential_delays",
                []
            )

            insights.setdefault(
                "alternatives",
                []
            )

            insights.setdefault(
                "weather_impact",
                "Minimal"
            )

            insights.setdefault(
                "recommendations",
                []
            )

            insights.setdefault(
                "confidence",
                0.75
            )

            return insights

        except Exception as e:

            error_message = str(e)

            # -----------------------------------------------------
            # Gemini quota exhausted
            # -----------------------------------------------------
            if (
                "429" in error_message
                or
                "RESOURCE_EXHAUSTED" in error_message
                or
                "quota" in error_message.lower()
            ):

                print(
                    "⚠️ Gemini quota exhausted. "
                    "Using local AI insight fallback."
                )

            else:

                print(
                    f"AI insights error: {e}"
                )

            # -----------------------------------------------------
            # Local fallback
            # -----------------------------------------------------
            return self.get_simulated_insights(
                route_data,
                realtime_data,
                optimize_for
            )

    # ============================================================
    # LOCAL TRAFFIC PREDICTION
    # ============================================================

    async def predict_traffic_with_ai(
        self,
        route_data: Dict,
        realtime_data: Dict
    ) -> Dict[str, Any]:
        """
        Predict traffic locally.

        Despite the old function name, this function no longer
        calls Gemini.

        This prevents traffic prediction from consuming Gemini
        quota.
        """

        try:

            # -----------------------------------------------------
            # Current traffic
            # -----------------------------------------------------
            traffic_level = realtime_data.get(
                "traffic_level",
                "medium"
            )

            # -----------------------------------------------------
            # Current time
            # -----------------------------------------------------
            hour = datetime.now().hour

            # -----------------------------------------------------
            # Default values
            # -----------------------------------------------------
            confidence = 0.70
            estimated_delay = 0

            predicted_level = traffic_level

            # -----------------------------------------------------
            # Base traffic delay
            # -----------------------------------------------------
            if traffic_level == "low":

                estimated_delay = random.randint(
                    0,
                    5
                )

                confidence = 0.80

                predicted_level = "low"

            elif traffic_level == "medium":

                estimated_delay = random.randint(
                    5,
                    15
                )

                confidence = 0.75

                predicted_level = "medium"

            elif traffic_level == "high":

                estimated_delay = random.randint(
                    15,
                    30
                )

                confidence = 0.85

                predicted_level = "high"

            # -----------------------------------------------------
            # Peak hour detection
            # -----------------------------------------------------
            morning_peak = (
                7 <= hour <= 9
            )

            evening_peak = (
                17 <= hour <= 19
            )

            is_peak = (
                morning_peak
                or
                evening_peak
            )

            if is_peak:

                if traffic_level == "low":

                    predicted_level = "medium"

                elif traffic_level == "medium":

                    predicted_level = "high"

                else:

                    predicted_level = "high"

                estimated_delay += 5

            # -----------------------------------------------------
            # Incident adjustment
            # -----------------------------------------------------
            incidents = realtime_data.get(
                "incidents",
                []
            )

            if incidents:

                estimated_delay += min(
                    len(incidents) * 3,
                    15
                )

                severe_incident = any(
                    isinstance(incident, dict)
                    and incident.get("severity") == "high"
                    for incident in incidents
                )

                if severe_incident:

                    predicted_level = "high"

                    estimated_delay += 10

                    confidence = min(
                        confidence + 0.05,
                        0.95
                    )

            # -----------------------------------------------------
            # Weather adjustment
            # -----------------------------------------------------
            weather = realtime_data.get(
                "weather",
                {}
            )

            weather_condition = weather.get(
                "condition",
                "Clear"
            )

            if weather_condition == "Rain":

                estimated_delay += 5

                if predicted_level == "low":
                    predicted_level = "medium"

            elif weather_condition == "Storm":

                estimated_delay += 10

                predicted_level = "high"

                confidence = min(
                    confidence + 0.05,
                    0.95
                )

            # -----------------------------------------------------
            # Generate recommendations
            # -----------------------------------------------------
            recommendations = []

            if predicted_level == "high":

                recommendations.append(
                    "Heavy traffic expected. "
                    "Allow additional travel time."
                )

            elif predicted_level == "medium":

                recommendations.append(
                    "Moderate traffic expected. "
                    "Drive carefully."
                )

            else:

                recommendations.append(
                    "Traffic conditions are currently favorable."
                )

            if is_peak:

                recommendations.append(
                    "Current time is within a typical "
                    "peak traffic period."
                )

            if weather_condition in [
                "Rain",
                "Storm"
            ]:

                recommendations.append(
                    "Weather conditions may increase "
                    "travel time."
                )

            if incidents:

                recommendations.append(
                    f"{len(incidents)} incident(s) "
                    "reported in the route area."
                )

            # -----------------------------------------------------
            # Return local prediction
            # -----------------------------------------------------
            return {

                "predicted_level": predicted_level,

                "confidence": round(
                    confidence,
                    2
                ),

                "estimated_delay_minutes": (
                    estimated_delay
                ),

                "peak_hours": [
                    "7-9 AM",
                    "5-7 PM"
                ],

                "recommendations": recommendations,

                "predicted_at": (
                    datetime.now().isoformat()
                ),

                "model": "local-traffic-logic"
            }

        except Exception as e:

            print(
                f"Local traffic prediction error: {e}"
            )

            return self.get_simulated_traffic_prediction(
                route_data
            )

    # ============================================================
    # ALTERNATIVE ROUTES
    # ============================================================

    async def get_ai_alternatives(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        primary_route: Dict,
        realtime_data: Dict,
        optimize_for: str
    ) -> List[Dict]:
        """
        Get and locally analyze alternative routes.

        IMPORTANT:
        No Gemini calls are made here.
        """

        # ---------------------------------------------------------
        # Get routes from OSRM
        # ---------------------------------------------------------
        alternatives = await self.get_osrm_alternatives(
            start_lat,
            start_lng,
            dest_lat,
            dest_lng
        )

        ai_alternatives = []

        # ---------------------------------------------------------
        # Analyze maximum 3 routes
        # ---------------------------------------------------------
        for i, alt in enumerate(
            alternatives[:3]
        ):

            try:

                # -------------------------------------------------
                # Local route analysis
                # -------------------------------------------------
                analysis = self.simulate_route_analysis(
                    alt,
                    optimize_for
                )

                # -------------------------------------------------
                # Local score
                # -------------------------------------------------
                score = self.calculate_route_score(
                    alt,
                    optimize_for
                )

                # -------------------------------------------------
                # Compare against primary route
                # -------------------------------------------------
                recommendation = (
                    self.get_recommendation_level(
                        alt,
                        primary_route
                    )
                )

                # -------------------------------------------------
                # Calculate time saving
                # -------------------------------------------------
                primary_duration = primary_route.get(
                    "duration",
                    0
                )

                alternative_duration = alt.get(
                    "duration",
                    0
                )

                time_difference = (
                    primary_duration
                    -
                    alternative_duration
                )

                time_saving_minutes = max(
                    0,
                    time_difference / 60
                )

                # -------------------------------------------------
                # Add route
                # -------------------------------------------------
                ai_alternatives.append({

                    **alt,

                    "ai_analysis": analysis,

                    "ai_score": score,

                    "recommendation": recommendation,

                    "time_saving_minutes": round(
                        time_saving_minutes,
                        2
                    )
                })

            except Exception as e:

                print(
                    f"Alternative analysis error "
                    f"for route {i}: {e}"
                )

                # -------------------------------------------------
                # Keep route even if analysis fails
                # -------------------------------------------------
                ai_alternatives.append({

                    **alt,

                    "ai_analysis": (
                        self.simulate_route_analysis(
                            alt,
                            optimize_for
                        )
                    ),

                    "ai_score": 0.70,

                    "recommendation": "good",

                    "time_saving_minutes": 0
                })

        return ai_alternatives

    # ============================================================
    # TRAFFIC PATTERN ANALYSIS
    # ============================================================

    async def analyze_traffic_patterns(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        realtime_data: Dict
    ) -> Dict:
        """
        Optional Gemini traffic pattern analysis.

        IMPORTANT:
        This function is retained for compatibility, but it is
        NOT automatically called by get_realtime_data().

        This prevents unnecessary Gemini requests.
        """

        # ---------------------------------------------------------
        # If Gemini unavailable
        # ---------------------------------------------------------
        if not self.client:

            return {
                "status": "simulated",
                "typical_traffic": (
                    realtime_data.get(
                        "traffic_level",
                        "medium"
                    )
                ),
                "peak_hours": [
                    "7-9 AM",
                    "5-7 PM"
                ],
                "best_times": [
                    "10 AM-4 PM",
                    "8 PM-6 AM"
                ]
            }

        try:

            prompt = f"""
Analyze traffic patterns between these coordinates
in Calapan City:

Start:
{start_lat}, {start_lng}

Destination:
{dest_lat}, {dest_lng}

Current Data:
{json.dumps(realtime_data)}

Current Time:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Provide analysis of:

1. Typical traffic patterns
2. Peak hours to avoid
3. Best times to travel
4. Historical incident hotspots

Return valid JSON.
"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                    response_mime_type="application/json"
                )
            )

            if not response or not response.text:

                raise ValueError(
                    "Empty Gemini response"
                )

            return json.loads(
                response.text
            )

        except Exception as e:

            error_message = str(e)

            if (
                "429" in error_message
                or
                "RESOURCE_EXHAUSTED" in error_message
            ):

                print(
                    "⚠️ Gemini traffic-pattern "
                    "quota exhausted."
                )

            else:

                print(
                    f"Traffic pattern analysis error: {e}"
                )

            return {
                "status": "fallback",
                "typical_traffic": (
                    realtime_data.get(
                        "traffic_level",
                        "medium"
                    )
                ),
                "peak_hours": [
                    "7-9 AM",
                    "5-7 PM"
                ],
                "best_times": [
                    "10 AM-4 PM",
                    "8 PM-6 AM"
                ],
                "error": str(e)
            }

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    async def generate_recommendations(
        self,
        route_data: Dict,
        realtime_data: Dict,
        ai_insights: Dict,
        user_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate recommendations locally.

        No Gemini calls.
        """

        recommendations = []

        # ---------------------------------------------------------
        # Safety
        # ---------------------------------------------------------
        safety_score = ai_insights.get(
            "safety_score",
            7
        )

        try:
            safety_score = float(
                safety_score
            )
        except Exception:
            safety_score = 7

        if safety_score < 7:

            recommendations.append({
                "type": "safety",
                "priority": "high",
                "message": (
                    "Consider a safer alternative route."
                ),
                "icon": "🛡️"
            })

        # ---------------------------------------------------------
        # Traffic
        # ---------------------------------------------------------
        traffic_level = realtime_data.get(
            "traffic_level",
            "medium"
        )

        if traffic_level == "high":

            recommendations.append({
                "type": "traffic",
                "priority": "high",
                "message": (
                    "Heavy traffic expected. "
                    "Allow additional travel time."
                ),
                "icon": "🚦"
            })

        elif traffic_level == "medium":

            recommendations.append({
                "type": "traffic",
                "priority": "medium",
                "message": (
                    "Moderate traffic expected."
                ),
                "icon": "🚦"
            })

        # ---------------------------------------------------------
        # Weather
        # ---------------------------------------------------------
        weather = realtime_data.get(
            "weather",
            {}
        )

        weather_condition = weather.get(
            "condition",
            "Clear"
        )

        if weather_condition in [
            "Rain",
            "Storm"
        ]:

            recommendations.append({
                "type": "weather",
                "priority": "high",
                "message": (
                    "Poor weather conditions. "
                    "Drive carefully and allow extra time."
                ),
                "icon": "🌧️"
            })

        # ---------------------------------------------------------
        # Incidents
        # ---------------------------------------------------------
        incidents = realtime_data.get(
            "incidents",
            []
        )

        if incidents:

            recommendations.append({
                "type": "incident",
                "priority": "high",
                "message": (
                    f"{len(incidents)} incident(s) "
                    "reported near the route."
                ),
                "icon": "⚠️"
            })

        # ---------------------------------------------------------
        # Peak hour
        # ---------------------------------------------------------
        hour = datetime.now().hour

        if (
            7 <= hour <= 9
            or
            17 <= hour <= 19
        ):

            recommendations.append({
                "type": "timing",
                "priority": "medium",
                "message": (
                    "Current time is within a typical "
                    "peak traffic period."
                ),
                "icon": "⏰"
            })

        # ---------------------------------------------------------
        # Gemini recommendations
        #
        # These are already contained inside the ONE Gemini
        # response generated by generate_ai_insights().
        # We do NOT make another request here.
        # ---------------------------------------------------------
        ai_recs = ai_insights.get(
            "recommendations",
            []
        )

        if isinstance(
            ai_recs,
            list
        ):

            for rec in ai_recs[:3]:

                if isinstance(
                    rec,
                    str
                ):

                    recommendations.append({
                        "type": "ai",
                        "priority": "low",
                        "message": rec,
                        "icon": "🤖"
                    })

        return recommendations

    # ============================================================
    # OSRM PRIMARY ROUTE
    # ============================================================

    async def get_osrm_route(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng
    ):
        """
        Get primary route from OSRM.
        """

        try:

            osrm_url = os.getenv(
                "OSRM_URL",
                "http://router.project-osrm.org"
            )

            url = (
                f"{osrm_url}"
                f"/route/v1/driving/"
                f"{start_lng},{start_lat};"
                f"{dest_lng},{dest_lat}"
                f"?overview=full"
                f"&geometries=geojson"
                f"&alternatives=3"
            )

            response = requests.get(
                url,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            if data.get("code") == "Ok":

                routes = data.get(
                    "routes",
                    []
                )

                if routes:

                    route = routes[0]

                    return {

                        "distance": route.get(
                            "distance",
                            0
                        ),

                        "duration": route.get(
                            "duration",
                            0
                        ),

                        "geometry": route.get(
                            "geometry",
                            {
                                "type": "LineString",
                                "coordinates": []
                            }
                        ),

                        "legs": route.get(
                            "legs",
                            []
                        ),

                        "confidence": 0.90
                    }

        except Exception as e:

            print(
                f"OSRM error: {e}"
            )

        # ---------------------------------------------------------
        # Fallback route
        # ---------------------------------------------------------
        return {

            "distance": 5000,

            "duration": 600,

            "geometry": {
                "type": "LineString",
                "coordinates": []
            },

            "legs": [],

            "confidence": 0.50
        }

    # ============================================================
    # OSRM ALTERNATIVES
    # ============================================================

    async def get_osrm_alternatives(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng
    ):
        """
        Get alternative routes from OSRM.
        """

        try:

            osrm_url = os.getenv(
                "OSRM_URL",
                "http://router.project-osrm.org"
            )

            url = (
                f"{osrm_url}"
                f"/route/v1/driving/"
                f"{start_lng},{start_lat};"
                f"{dest_lng},{dest_lat}"
                f"?overview=full"
                f"&geometries=geojson"
                f"&alternatives=true"
            )

            response = requests.get(
                url,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            alternatives = []

            if data.get("code") == "Ok":

                routes = data.get(
                    "routes",
                    []
                )

                for i, route in enumerate(
                    routes[:3]
                ):

                    alternatives.append({

                        "id": f"alt_{i + 1}",

                        "distance": route.get(
                            "distance",
                            0
                        ),

                        "duration": route.get(
                            "duration",
                            0
                        ),

                        "geometry": route.get(
                            "geometry",
                            {
                                "type": "LineString",
                                "coordinates": []
                            }
                        ),

                        "legs": route.get(
                            "legs",
                            []
                        ),

                        "summary": (
                            f"Alternative {i + 1}"
                        ),

                        "time_saving": 0
                    })

            return alternatives

        except Exception as e:

            print(
                f"OSRM alternatives error: {e}"
            )

            return []

    # ============================================================
    # ROUTE SCORE
    # ============================================================

    def calculate_ai_score(
        self,
        route,
        optimize_for
    ):
        """
        Calculate local AI-style route score.
        """

        base_score = 0.70

        duration = route.get(
            "duration",
            0
        )

        distance = route.get(
            "distance",
            0
        )

        # ---------------------------------------------------------
        # Fastest
        # ---------------------------------------------------------
        if optimize_for == "fastest":

            if duration < 1800:
                base_score += 0.10
            else:
                base_score -= 0.10

        # ---------------------------------------------------------
        # Safest
        # ---------------------------------------------------------
        elif optimize_for == "safest":

            base_score += 0.15

        # ---------------------------------------------------------
        # Scenic
        # ---------------------------------------------------------
        elif optimize_for == "scenic":

            base_score += 0.05

        # ---------------------------------------------------------
        # Short routes
        # ---------------------------------------------------------
        if distance > 0 and distance < 5000:

            base_score += 0.03

        # ---------------------------------------------------------
        # Small random variation
        # ---------------------------------------------------------
        base_score += random.uniform(
            -0.05,
            0.05
        )

        return round(
            min(
                max(
                    base_score,
                    0.50
                ),
                0.95
            ),
            2
        )

    # ============================================================
    # ALTERNATIVE ROUTE SCORE
    # ============================================================

    def calculate_route_score(
        self,
        route,
        optimize_for
    ):
        """
        Calculate score for an alternative route.
        """

        duration = route.get(
            "duration",
            0
        )

        distance = route.get(
            "distance",
            0
        )

        score = 0.70

        # ---------------------------------------------------------
        # Fastest
        # ---------------------------------------------------------
        if optimize_for == "fastest":

            if duration < 1200:

                score += 0.15

            elif duration < 1800:

                score += 0.10

            elif duration > 3600:

                score -= 0.10

        # ---------------------------------------------------------
        # Safest
        # ---------------------------------------------------------
        elif optimize_for == "safest":

            score += 0.10

            if distance < 10000:

                score += 0.05

        # ---------------------------------------------------------
        # Scenic
        # ---------------------------------------------------------
        elif optimize_for == "scenic":

            score += 0.10

        # ---------------------------------------------------------
        # Shorter route bonus
        # ---------------------------------------------------------
        if distance > 0:

            if distance < 5000:

                score += 0.05

            elif distance > 20000:

                score -= 0.05

        return round(
            min(
                max(
                    score,
                    0.50
                ),
                0.95
            ),
            2
        )

    # ============================================================
    # RECOMMENDATION LEVEL
    # ============================================================

    def get_recommendation_level(
        self,
        alternative,
        primary_route
    ):
        """
        Compare an alternative route with the primary route.
        """

        alternative_duration = alternative.get(
            "duration",
            0
        )

        primary_duration = primary_route.get(
            "duration",
            0
        )

        alternative_distance = alternative.get(
            "distance",
            0
        )

        primary_distance = primary_route.get(
            "distance",
            0
        )

        # ---------------------------------------------------------
        # Invalid data
        # ---------------------------------------------------------
        if primary_duration <= 0:

            return "good"

        # ---------------------------------------------------------
        # Calculate differences
        # ---------------------------------------------------------
        time_difference = (
            primary_duration
            -
            alternative_duration
        )

        distance_difference = (
            primary_distance
            -
            alternative_distance
        )

        # ---------------------------------------------------------
        # Much faster
        # ---------------------------------------------------------
        if time_difference >= 300:

            return "best"

        # ---------------------------------------------------------
        # Moderately faster
        # ---------------------------------------------------------
        if time_difference >= 120:

            return "better"

        # ---------------------------------------------------------
        # Shorter route
        # ---------------------------------------------------------
        if distance_difference >= 2000:

            return "better"

        # ---------------------------------------------------------
        # Slightly faster
        # ---------------------------------------------------------
        if time_difference > 0:

            return "good"

        # ---------------------------------------------------------
        # Slightly slower but close
        # ---------------------------------------------------------
        if (
            alternative_duration
            <=
            primary_duration * 1.10
        ):

            return "good"

        # ---------------------------------------------------------
        # Much slower
        # ---------------------------------------------------------
        return "avoid"

    # ============================================================
    # LOCAL ALTERNATIVE ANALYSIS
    # ============================================================

    def simulate_route_analysis(
        self,
        route,
        optimize_for
    ):
        """
        Analyze an alternative route locally.

        No Gemini request.
        """

        distance_km = (
            route.get(
                "distance",
                0
            )
            /
            1000
        )

        duration_minutes = (
            route.get(
                "duration",
                0
            )
            /
            60
        )

        # ---------------------------------------------------------
        # Traffic assessment
        # ---------------------------------------------------------
        if duration_minutes <= 15:

            traffic_assessment = (
                "Route has a relatively short "
                "estimated travel time."
            )

        elif duration_minutes <= 30:

            traffic_assessment = (
                "Route has a moderate estimated "
                "travel time."
            )

        else:

            traffic_assessment = (
                "Route has a longer estimated "
                "travel time."
            )

        # ---------------------------------------------------------
        # Emergency access
        # ---------------------------------------------------------
        if distance_km <= 10:

            emergency_access = (
                "Good for emergency response."
            )

        elif distance_km <= 20:

            emergency_access = (
                "Generally suitable for emergency response."
            )

        else:

            emergency_access = (
                "Long route. Consider a faster alternative "
                "when available."
            )

        # ---------------------------------------------------------
        # Recommendation
        # ---------------------------------------------------------
        if optimize_for == "fastest":

            recommendation = (
                "Prioritize this route if its estimated "
                "travel time is lower."
            )

        elif optimize_for == "safest":

            recommendation = (
                "Consider this route when safety and "
                "road accessibility are priorities."
            )

        elif optimize_for == "scenic":

            recommendation = (
                "This route may be considered when "
                "route experience is a priority."
            )

        else:

            recommendation = (
                "Compare this route with the primary "
                "route before selecting it."
            )

        return {

            "distance_km": round(
                distance_km,
                2
            ),

            "estimated_minutes": round(
                duration_minutes
            ),

            "optimization": optimize_for,

            "traffic_assessment": (
                traffic_assessment
            ),

            "emergency_access": (
                emergency_access
            ),

            "recommendation": (
                recommendation
            ),

            "confidence": 0.70
        }

    # ============================================================
    # TIME SAVINGS
    # ============================================================

    def calculate_time_savings(
        self,
        primary,
        alternatives
    ):
        """
        Calculate time savings between primary route
        and alternative routes.
        """

        savings = []

        primary_time = primary.get(
            "duration",
            0
        )

        if primary_time <= 0:

            return savings

        for alt in alternatives:

            time_diff = (
                primary_time
                -
                alt.get(
                    "duration",
                    0
                )
            )

            if time_diff > 0:

                savings.append({

                    "alternative_id": (
                        alt.get("id")
                    ),

                    "minutes_saved": round(
                        time_diff / 60,
                        2
                    ),

                    "percentage": round(
                        (
                            time_diff
                            /
                            primary_time
                        )
                        *
                        100,
                        2
                    )
                })

        return savings

    # ============================================================
    # ROUTE CONFIDENCE
    # ============================================================

    def calculate_route_confidence(
        self,
        route,
        realtime_data
    ):
        """
        Calculate local confidence score.
        """

        confidence = 0.70

        # ---------------------------------------------------------
        # OSRM confidence
        # ---------------------------------------------------------
        route_confidence = route.get(
            "confidence",
            0.70
        )

        try:

            confidence = (
                confidence
                +
                float(route_confidence)
            ) / 2

        except Exception:

            confidence = 0.70

        # ---------------------------------------------------------
        # Realtime data
        # ---------------------------------------------------------
        if realtime_data:

            confidence += 0.05

        # ---------------------------------------------------------
        # Incidents make prediction less certain
        # ---------------------------------------------------------
        incidents = realtime_data.get(
            "incidents",
            []
        )

        if len(incidents) > 2:

            confidence -= 0.05

        return round(
            min(
                max(
                    confidence,
                    0.50
                ),
                0.95
            ),
            2
        )

    # ============================================================
    # SIMULATED AI INSIGHTS
    # ============================================================

    def get_simulated_insights(
        self,
        route_data,
        realtime_data,
        optimize_for
    ):
        """
        Local fallback when Gemini is unavailable.
        """

        traffic_level = realtime_data.get(
            "traffic_level",
            "medium"
        )

        weather = realtime_data.get(
            "weather",
            {}
        )

        incidents = realtime_data.get(
            "incidents",
            []
        )

        # ---------------------------------------------------------
        # Safety score
        # ---------------------------------------------------------
        safety_score = 8

        if traffic_level == "medium":

            safety_score = 7

        elif traffic_level == "high":

            safety_score = 6

        if incidents:

            safety_score -= min(
                len(incidents),
                2
            )

        if weather.get(
            "condition"
        ) in [
            "Rain",
            "Storm"
        ]:

            safety_score -= 1

        safety_score = max(
            1,
            min(
                safety_score,
                10
            )
        )

        # ---------------------------------------------------------
        # Emergency access
        # ---------------------------------------------------------
        if safety_score >= 8:

            emergency_access = "Excellent"

        elif safety_score >= 6:

            emergency_access = "Good"

        elif safety_score >= 4:

            emergency_access = "Moderate"

        else:

            emergency_access = "Poor"

        # ---------------------------------------------------------
        # Delays
        # ---------------------------------------------------------
        potential_delays = []

        if traffic_level == "high":

            potential_delays.append(
                "Heavy traffic may cause significant delays."
            )

        elif traffic_level == "medium":

            potential_delays.append(
                "Moderate traffic may cause some delays."
            )

        else:

            potential_delays.append(
                "Traffic is currently relatively light."
            )

        if incidents:

            potential_delays.append(
                f"{len(incidents)} incident(s) "
                "may affect travel time."
            )

        # ---------------------------------------------------------
        # Weather
        # ---------------------------------------------------------
        weather_condition = weather.get(
            "condition",
            "Clear"
        )

        if weather_condition == "Storm":

            weather_impact = (
                "High impact. Storm conditions may "
                "significantly affect travel."
            )

        elif weather_condition == "Rain":

            weather_impact = (
                "Moderate impact. Rain may increase "
                "travel time and reduce visibility."
            )

        else:

            weather_impact = (
                "Minimal weather impact expected."
            )

        # ---------------------------------------------------------
        # Recommendations
        # ---------------------------------------------------------
        recommendations = []

        if traffic_level == "high":

            recommendations.append(
                "Consider an alternative route "
                "if available."
            )

        if incidents:

            recommendations.append(
                "Check incident locations before departure."
            )

        if weather_condition in [
            "Rain",
            "Storm"
        ]:

            recommendations.append(
                "Reduce speed and maintain a safe "
                "following distance."
            )

        if not recommendations:

            recommendations.append(
                "Current conditions appear suitable "
                "for emergency travel."
            )

        return {

            "safety_score": safety_score,

            "emergency_access": (
                emergency_access
            ),

            "potential_delays": (
                potential_delays
            ),

            "alternatives": [],

            "weather_impact": (
                weather_impact
            ),

            "recommendations": (
                recommendations
            ),

            "confidence": 0.70,

            "mode": "local-fallback"
        }

    # ============================================================
    # SIMULATED TRAFFIC PREDICTION
    # ============================================================

    def get_simulated_traffic_prediction(
        self,
        route_data
    ):
        """
        Fallback traffic prediction.
        """

        traffic_level = random.choice(
            [
                "low",
                "medium",
                "high"
            ]
        )

        if traffic_level == "low":

            delay = random.randint(
                0,
                5
            )

        elif traffic_level == "medium":

            delay = random.randint(
                5,
                15
            )

        else:

            delay = random.randint(
                15,
                30
            )

        return {

            "predicted_level": (
                traffic_level
            ),

            "confidence": round(
                random.uniform(
                    0.60,
                    0.80
                ),
                2
            ),

            "estimated_delay_minutes": (
                delay
            ),

            "peak_hours": [
                "7-9 AM",
                "5-7 PM"
            ],

            "recommendations": [
                "Avoid peak hours if possible."
            ],

            "predicted_at": (
                datetime.now().isoformat()
            ),

            "model": "local-simulation"
        }

    # ============================================================
    # SIMULATED INCIDENTS
    # ============================================================

    async def get_simulated_incidents(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng
    ):
        """
        Simulate incidents.

        Replace this later with your actual incident
        database/API.
        """

        incidents = []

        # 30% chance of an incident
        if random.random() > 0.70:

            incidents.append({

                "type": random.choice([
                    "accident",
                    "construction",
                    "road_closed"
                ]),

                "severity": random.choice([
                    "low",
                    "medium",
                    "high"
                ]),

                "location": (
                    "Near city center"
                ),

                "description": (
                    "Minor incident reported"
                )
            })

        return incidents

    # ============================================================
    # SIMULATED WEATHER
    # ============================================================

    async def get_simulated_weather(
        self,
        lat,
        lng
    ):
        """
        Simulate weather.

        Replace this later with your actual weather API.
        """

        conditions = [
            "Clear",
            "Partly Cloudy",
            "Cloudy",
            "Rain",
            "Storm"
        ]

        return {

            "condition": random.choice(
                conditions
            ),

            "temperature": random.randint(
                25,
                35
            ),

            "humidity": random.randint(
                60,
                90
            ),

            "visibility": random.choice([
                "Good",
                "Moderate",
                "Poor"
            ])
        }

    # ============================================================
    # SIMULATED ROAD CONDITIONS
    # ============================================================

    def get_simulated_road_conditions(
        self
    ):
        """
        Simulate road conditions.
        """

        return {

            "surface": random.choice([
                "Good",
                "Fair",
                "Poor"
            ]),

            "maintenance": random.choice([
                "Well-maintained",
                "Needs repair"
            ]),

            "lighting": random.choice([
                "Good",
                "Adequate",
                "Poor"
            ])
        }

    # ============================================================
    # FALLBACK REALTIME DATA
    # ============================================================

    def get_fallback_realtime_data(
        self
    ):
        """
        Fallback realtime data.
        """

        return {

            "traffic_level": "medium",

            "average_speed": 40,

            "incidents": [],

            "weather": {
                "condition": "Clear",
                "temperature": 30,
                "humidity": 70,
                "visibility": "Good"
            },

            "road_conditions": {
                "surface": "Good",
                "maintenance": "Well-maintained",
                "lighting": "Good"
            },

            "time_of_day": (
                datetime.now().hour
            ),

            "day_of_week": (
                datetime.now().strftime(
                    "%A"
                )
            ),

            "last_updated": (
                datetime.now().isoformat()
            )
        }


# ================================================================
# SINGLETON INSTANCE
# ================================================================

map_service = GeminiMapService()