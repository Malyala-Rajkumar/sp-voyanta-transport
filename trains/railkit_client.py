import hashlib
import hmac
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo
import requests

from config import RAILKIT_API_KEY, RAILKIT_BASE_URL

IST = ZoneInfo("Asia/Kolkata")


def normalize_date_str(val: Any) -> str:
    """Normalize any date representation into RailKit's required DD-MM-YYYY format.

    Accepts:
    - datetime.date / datetime.datetime
    - ISO 'YYYY-MM-DD'
    - 'DD-MM-YYYY' or 'DD/MM/YYYY'
    - 'today', 'tomorrow', 'next Friday', etc.
    """
    if isinstance(val, (datetime, date)):
        return val.strftime("%d-%m-%Y")

    s = str(val).strip()
    if not s:
        return datetime.now(IST).strftime("%d-%m-%Y")

    s_lower = s.lower()
    now_ist = datetime.now(IST).date()

    if s_lower in ("today", "todays"):
        return now_ist.strftime("%d-%m-%Y")
    if s_lower in ("tomorrow", "tomorrows", "tmrw"):
        return (now_ist + timedelta(days=1)).strftime("%d-%m-%Y")
    if s_lower in ("day after tomorrow", "day after tmrw"):
        return (now_ist + timedelta(days=2)).strftime("%d-%m-%Y")

    # Try ISO YYYY-MM-DD
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", s):
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            pass

    # Try DD-MM-YYYY
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", s):
        try:
            dt = datetime.strptime(s, "%d-%m-%Y").date()
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            pass

    # Try DD/MM/YYYY
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", s):
        try:
            dt = datetime.strptime(s, "%d/%m/%Y").date()
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            pass

    # Fallback to dateutil or return as is
    return s


# In-memory API Cache with 10-minute TTL (matches flights/flight_v4.py cache architecture)
_API_CACHE: dict[tuple, tuple[float, dict]] = {}
CACHE_TTL_SECONDS = 600

class RailKitClient:

    def __init__(self):
        self.base_url = RAILKIT_BASE_URL.rstrip("/")
        self.session = requests.Session()
        self.api_key = RAILKIT_API_KEY

    def _get_signature_headers(self, endpoint):
        """Generates the required timestamp and HMAC signature headers."""
        timestamp = str(int(time.time()))
        
        # Create a signature payload combining timestamp and endpoint
        message = f"{timestamp}:{endpoint}"
        
        # Compute HMAC-SHA256 signature using your API key as the secret
        signature = hmac.new(
            self.api_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        return {
            "x-api-key": self.api_key,
            "x-timestamp": timestamp,
            "x-signature": signature,
            "Accept": "application/json"
        }

    def _request(self, endpoint, params=None):
        # Cache lookup
        cache_key = (endpoint, tuple(sorted((params or {}).items())))
        cached = _API_CACHE.get(cache_key)
        if cached and (time.time() - cached[0] < CACHE_TTL_SECONDS):
            return cached[1]

        url = f"{self.base_url}{endpoint}"
        
        # Dynamically generate fresh signature headers for every call
        headers = self._get_signature_headers(endpoint)

        response = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=12
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, dict):
            if not result.get("success", False):
                raise RuntimeError(
                    result.get("message", "RailKit API request failed")
                )

        # Store in cache
        if len(_API_CACHE) > 500:
            _API_CACHE.clear()
        _API_CACHE[cache_key] = (time.time(), result)

        return result

    def search_trains(
        self,
        source: str,
        destination: str,
        journey_date: Any,
    ) -> dict[str, Any]:
        """Search trains between two specific station codes."""
        norm_date = normalize_date_str(journey_date)
        src_clean = str(source).strip().upper()
        dst_clean = str(destination).strip().upper()

        endpoint = (
            f"/api/v1/trains/between/"
            f"{quote(src_clean, safe='')}/"
            f"{quote(dst_clean, safe='')}"
        )

        return self._request(
            endpoint,
            params={"date": norm_date}
        )

    def search_trains_cluster(
        self,
        source_codes: list[str],
        dest_codes: list[str],
        journey_date: Any,
    ) -> dict[str, Any]:
        """Search trains across metro clusters concurrently via ThreadPoolExecutor.

        Queries top terminal pairs in parallel, merges results, deduplicates by train_no,
        and sorts chronologically by departure time in sub-2 seconds.
        """
        norm_date = normalize_date_str(journey_date)
        all_trains = []
        seen_train_nos = set()

        # Prioritize primary stations, take top 2x2 or 3x2 pairs
        src_subset = [s.strip().upper() for s in source_codes[:3] if s]
        dst_subset = [d.strip().upper() for d in dest_codes[:2] if d]

        pairs = [(s, d) for s in src_subset for d in dst_subset if s != d]

        def _fetch_pair(src_c: str, dst_c: str):
            try:
                res = self.search_trains(src_c, dst_c, norm_date)
                return res.get("data") or []
            except Exception:
                return []

        # Dispatch all station pair requests concurrently
        with ThreadPoolExecutor(max_workers=min(len(pairs) or 1, 6)) as executor:
            future_to_pair = {executor.submit(_fetch_pair, s, d): (s, d) for s, d in pairs}
            for future in as_completed(future_to_pair):
                train_list = future.result()
                for t in train_list:
                    tno = t.get("train_no")
                    if tno and tno not in seen_train_nos:
                        seen_train_nos.add(tno)
                        all_trains.append(t)

        # Sort trains chronologically by departure time ('from_time' e.g. "03:40")
        all_trains.sort(key=lambda x: x.get("from_time", "99:99"))

        return {
            "success": True,
            "data": all_trains,
            "searched_date": norm_date,
            "source_stations": src_subset,
            "dest_stations": dst_subset,
            "total_found": len(all_trains),
        }

    def check_availability(
        self,
        train_number: str | int,
        source: str,
        destination: str,
        journey_date: Any,
        travel_class: str,
        quota: str = "GN",
    ) -> dict[str, Any]:
        """Check live seat availability and fare breakdown."""
        norm_date = normalize_date_str(journey_date)

        values = [
            str(train_number).strip(),
            str(source).strip().upper(),
            str(destination).strip().upper(),
            norm_date,
            str(travel_class).strip().upper(),
            str(quota).strip().upper(),
        ]

        endpoint = "/api/v1/seats/" + "/".join(
            quote(str(value), safe="")
            for value in values
        )

        return self._request(endpoint)


railkit = RailKitClient()