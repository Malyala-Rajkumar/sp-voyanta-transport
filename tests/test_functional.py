"""Functional tests for SP Voyanta end-to-end transport orchestration.

Runs fast deterministic functional tests with mocked external APIs by default.
Set RUN_LIVE_API_TESTS=1 to also run live API smoke tests against RailKit/SerpAPI/Google Maps.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "flights", ROOT_DIR / "trains"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import orchestrator_tools


class TestEndToEndFunctionalMocked(unittest.TestCase):
    @patch("orchestrator_tools.find_off_grid_destination")
    @patch("orchestrator_tools.execute_train_search")
    @patch("orchestrator_tools.get_live_intercity_distance")
    def test_short_corridor_disables_flights(self, mock_dist, mock_train, mock_off_grid):
        mock_off_grid.return_value = None
        mock_dist.return_value = (144.0, "2h 55m", True)
        mock_train.invoke.return_value = json.dumps({
            "status": "success",
            "top_trains": [
                {
                    "train_number": "20707",
                    "train_name": "VSKP VANDEBHARAT",
                    "departure_time": "05:00",
                    "arrival_time": "06:39",
                    "duration": "1h 39m",
                    "price": 525,
                    "seat_status": "AVL 171",
                },
                {
                    "train_number": "11019",
                    "train_name": "KONARK EXPRESS",
                    "departure_time": "02:40",
                    "arrival_time": "05:58",
                    "duration": "3h 18m",
                    "price": 520,
                    "seat_status": "WL 5",
                },
            ],
            "recommendations": {
                "best_overall": {
                    "train_number": "20707",
                    "train_name": "VSKP VANDEBHARAT",
                    "departure_time": "05:00",
                    "arrival_time": "06:39",
                    "duration": "1h 39m",
                    "price": 525,
                    "seat_status": "AVL 171",
                },
                "cheapest": {
                    "train_number": "11019",
                    "train_name": "KONARK EXPRESS",
                    "departure_time": "02:40",
                    "arrival_time": "05:58",
                    "duration": "3h 18m",
                    "price": 520,
                    "seat_status": "WL 5",
                },
            },
        })

        res = json.loads(orchestrator_tools.plan_end_to_end_journey.invoke({
            "source": "Secunderabad",
            "destination": "Warangal",
            "travel_date": "2026-11-27",
            "passengers": 1,
            "travel_class": "CC",
        }))

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["journey_type"], "INTERCITY_SHORT_CORRIDOR")
        self.assertFalse(res["flight_available"])
        self.assertEqual(res["packages"]["option_a"]["total_price_inr"], 525)
        self.assertEqual(res["packages"]["option_b"]["total_price_inr"], 520)

    @patch("orchestrator_tools.calculate_generic_road_transit")
    @patch("orchestrator_tools.execute_flight_search")
    @patch("orchestrator_tools.execute_train_search")
    @patch("orchestrator_tools.get_live_intercity_distance")
    @patch("orchestrator_tools.find_off_grid_destination")
    def test_intercity_3leg_door_to_door_and_fastest_pick(
        self, mock_off_grid, mock_dist, mock_train, mock_flight, mock_road
    ):
        mock_off_grid.return_value = None
        mock_dist.return_value = (1148.0, "19h 0m", True)

        # First call (Alappuzha) returns no trains; second call (Cochin) returns live train
        mock_train.invoke.side_effect = [
            json.dumps({"status": "no_results", "trains": []}),
            json.dumps({
                "status": "success",
                "top_trains": [{
                    "train_number": "20629",
                    "train_name": "SABARI SF EXP",
                    "departure_time": "14:25",
                    "arrival_time": "12:50",
                    "duration": "22h 25m",
                    "price": 1650,
                    "seat_status": "Available",
                }],
                "recommendations": {},
            }),
        ]

        mock_flight.invoke.return_value = json.dumps({
            "status": "success",
            "booking_url": "https://www.google.com/travel/flights",
            "top_flights": [
                {
                    "airline": "Air India Express",
                    "flight_number": "IX 1246",
                    "departure_time": "6:15 AM",
                    "arrival_time": "9:45 AM",
                    "duration_formatted": "3h 30m",
                    "stops": 1,
                    "price": 34500,
                },
                {
                    "airline": "IndiGo",
                    "flight_number": "6E 695",
                    "departure_time": "7:55 AM",
                    "arrival_time": "9:35 AM",
                    "duration_formatted": "1h 40m",
                    "stops": 0,
                    "price": 39384,
                },
            ],
        })

        mock_road.return_value = {
            "distance_km": 50.0,
            "estimated_duration_formatted": "1h 30m",
            "estimated_duration_minutes": 90,
            "options": {
                "private_cab": {"total_fare_inr": 1050},
                "auto": {"total_fare_inr": 750},
            },
        }

        res = json.loads(orchestrator_tools.plan_end_to_end_journey.invoke({
            "source": "Gachibowli, Hyderabad",
            "destination": "Alappuzha",
            "travel_date": "2026-12-04",
            "passengers": 2,
            "travel_class": "3A",
        }))

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["flight_available"])
        self.assertEqual(len(res["packages"]["option_a"]["legs"]), 3)
        self.assertEqual(len(res["packages"]["option_b"]["legs"]), 3)
        self.assertEqual(res["curated_recommendations"]["best_overall"]["title"], "Air India Express (IX 1246)")
        self.assertEqual(res["curated_recommendations"]["fastest"]["title"], "IndiGo (6E 695)")
        self.assertEqual(res["curated_recommendations"]["cheapest"]["title"], "SABARI SF EXP (20629)")
        self.assertEqual(res["curated_recommendations"]["other_available_flights"][0]["duration"], "1h 40m")


@unittest.skipUnless(os.getenv("RUN_LIVE_API_TESTS") == "1", "Set RUN_LIVE_API_TESTS=1 to run live API smoke test")
class TestLiveAPISmoke(unittest.TestCase):
    def test_live_short_corridor(self):
        res = json.loads(orchestrator_tools.plan_end_to_end_journey.invoke({
            "source": "Secunderabad",
            "destination": "Warangal",
            "travel_date": "2026-11-27",
            "passengers": 1,
            "travel_class": "CC",
        }))
        self.assertEqual(res["status"], "success")
        self.assertFalse(res["flight_available"])
        self.assertTrue(len(res["packages"]["option_a"]["legs"]) >= 1)


if __name__ == "__main__":
    unittest.main()
