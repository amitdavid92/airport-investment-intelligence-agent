"""Route-level analysis: the long-haul share of departures.

This is the one question in the brief that the key-free sources cannot answer.
BTS publishes T-100 only in an airport-aggregated form on its public
FeatureServer (no destination, no distance, no seats), the FAA layers are
infrastructure-only, and OpenSky's historical flight endpoints stopped serving
anonymous callers in March 2026:

    GET /api/flights/departure?airport=KSFO -> 403 "You cannot access
    historical flights"

Rather than approximate a percentage and present it as fact, the default
provider reports what the published data genuinely supports - long-haul
*capability*, from runway length - and says plainly that the actual route mix
is unavailable. Supplying OpenSky credentials switches on the real
computation with no other change.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..scoring.kpis import LONG_HAUL_RUNWAY_FT

log = logging.getLogger(__name__)

OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
OPENSKY_DEPARTURE_URL = "https://opensky-network.org/api/flights/departure"

# Statute miles. The common industry line between medium- and long-haul; it is
# a convention, not a legal definition, so it is stated wherever it is used.
LONG_HAUL_THRESHOLD_MI = 2_500

EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


@dataclass
class LongHaulResult:
    """Answer plus its own provenance.

    `available` is what stops the agent reporting a proxy as if it were a
    measured percentage.
    """

    iata: str
    available: bool
    source: str
    method: str
    long_haul_pct: float | None = None
    flights_sampled: int | None = None
    window_days: int | None = None
    capability: dict[str, Any] | None = None
    caveat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "iata": self.iata,
            "long_haul_share_available": self.available,
            "source": self.source,
            "method": self.method,
            "long_haul_threshold_miles": LONG_HAUL_THRESHOLD_MI,
        }
        if self.available:
            payload["long_haul_pct_of_departures"] = self.long_haul_pct
            payload["flights_sampled"] = self.flights_sampled
            payload["window_days"] = self.window_days
        if self.capability:
            payload["runway_capability"] = self.capability
        if self.caveat:
            payload["caveat"] = self.caveat
        return payload


class LongHaulProvider(Protocol):
    def long_haul_share(self, iata: str) -> LongHaulResult: ...


class RunwayCapabilityProvider:
    """Default provider. Reports capability, never a fabricated percentage."""

    source = "faa_runway_capability"

    def __init__(self, airports) -> None:
        self._airports = airports

    def long_haul_share(self, iata: str) -> LongHaulResult:
        rows = self._airports[self._airports["iata"].str.upper() == iata.upper()]
        capability: dict[str, Any] | None = None

        if not rows.empty:
            row = rows.iloc[0]
            longest = row.get("runway_len_max")
            longest = None if longest is None or longest != longest else float(longest)
            capability = {
                "longest_runway_ft": longest,
                # Derived here rather than read from the row: this provider is
                # handed the raw ingested table, which has no computed columns.
                "widebody_long_haul_capable": bool(
                    longest is not None and longest >= LONG_HAUL_RUNWAY_FT
                ),
                "runway_count": None
                if row.get("runway_count") is None
                else float(row["runway_count"]),
                "interpretation": (
                    "A runway of 10,000 ft or more can support widebody long-haul "
                    "departures. This says what the airport is physically able to "
                    "do - not how many long-haul flights it actually operates."
                ),
            }

        return LongHaulResult(
            iata=iata.upper(),
            available=False,
            source=self.source,
            method="runway length as a capability proxy",
            capability=capability,
            caveat=(
                "The actual share of long-haul departures cannot be computed from "
                "the public data sources in use. BTS publishes T-100 on its open "
                "FeatureServer only in airport-aggregated form, with no destination "
                "or distance, and OpenSky's historical flight endpoints now require "
                "credentials. Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET to "
                "compute the real figure from live departure records."
            ),
        )


class OpenSkyRouteProvider:
    """Real long-haul share, computed from OpenSky departure records.

    Enabled when OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET are present. Each
    departure carries an estimated arrival ICAO; joining that to OurAirports
    coordinates gives a great-circle distance per flight, and the long-haul
    share is the fraction of those above the threshold.
    """

    source = "opensky"

    def __init__(self, airports, *, window_days: int = 2) -> None:
        self._airports = airports
        self._window_days = window_days
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._coords = self._build_coord_index(airports)

    @staticmethod
    def _build_coord_index(airports) -> dict[str, tuple[float, float]]:
        index: dict[str, tuple[float, float]] = {}
        for _, row in airports.iterrows():
            lat, lon = row.get("lat"), row.get("lon")
            if lat is None or lon is None:
                continue
            try:
                point = (float(lat), float(lon))
            except (TypeError, ValueError):
                continue
            if isinstance(row.get("icao"), str) and row["icao"]:
                index[row["icao"].upper()] = point
        return index

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("OPENSKY_CLIENT_ID") and os.getenv("OPENSKY_CLIENT_SECRET"))

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                OPENSKY_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": os.environ["OPENSKY_CLIENT_ID"],
                    "client_secret": os.environ["OPENSKY_CLIENT_SECRET"],
                },
            )
            response.raise_for_status()
            payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + float(payload.get("expires_in", 1800))
        return self._token

    def long_haul_share(self, iata: str) -> LongHaulResult:
        rows = self._airports[self._airports["iata"].str.upper() == iata.upper()]
        if rows.empty or not rows.iloc[0].get("icao"):
            return RunwayCapabilityProvider(self._airports).long_haul_share(iata)

        row = rows.iloc[0]
        origin_icao = str(row["icao"]).upper()
        end = int(time.time())
        begin = end - self._window_days * 86_400

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.get(
                    OPENSKY_DEPARTURE_URL,
                    params={"airport": origin_icao, "begin": begin, "end": end},
                    headers={"Authorization": f"Bearer {self._access_token()}"},
                )
                response.raise_for_status()
                flights = response.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("OpenSky lookup failed for %s (%s); falling back", iata, exc)
            fallback = RunwayCapabilityProvider(self._airports).long_haul_share(iata)
            fallback.caveat = f"OpenSky request failed ({exc}). {fallback.caveat}"
            return fallback

        origin = self._coords.get(origin_icao)
        if origin is None or not flights:
            return RunwayCapabilityProvider(self._airports).long_haul_share(iata)

        distances = []
        for flight in flights:
            destination = flight.get("estArrivalAirport")
            if not destination:
                continue
            point = self._coords.get(str(destination).upper())
            if point is None:
                continue
            distances.append(haversine_miles(origin[0], origin[1], point[0], point[1]))

        if not distances:
            fallback = RunwayCapabilityProvider(self._airports).long_haul_share(iata)
            fallback.caveat = (
                "OpenSky returned departures but none had a resolvable destination "
                f"airport. {fallback.caveat}"
            )
            return fallback

        long_haul = sum(1 for d in distances if d >= LONG_HAUL_THRESHOLD_MI)
        return LongHaulResult(
            iata=iata.upper(),
            available=True,
            source=self.source,
            method=(
                "great-circle distance from OpenSky departure records, joined to "
                "OurAirports coordinates"
            ),
            long_haul_pct=round(100 * long_haul / len(distances), 2),
            flights_sampled=len(distances),
            window_days=self._window_days,
            caveat=(
                f"Based on {len(distances)} departures with a resolvable destination "
                f"over the last {self._window_days} days. This is a short live "
                "sample, not an annual figure, and OpenSky's ADS-B coverage is "
                "uneven, so treat it as indicative."
            ),
        )


def get_provider(airports) -> LongHaulProvider:
    """Pick the strongest provider the current environment supports."""
    if OpenSkyRouteProvider.available():
        log.info("OpenSky credentials found - using live route data")
        return OpenSkyRouteProvider(airports)
    log.info("no OpenSky credentials - long-haul share will report as unavailable")
    return RunwayCapabilityProvider(airports)
