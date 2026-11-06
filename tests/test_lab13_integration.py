import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api.model_loader import ModelLoader

ROOT = Path(__file__).resolve().parents[1]


class ProductionApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(ModelLoader()))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_reports_prepared_model(self):
        metadata = json.loads(
            (ROOT / "deployment" / "model_metadata.json").read_text(encoding="utf-8")
        )
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": metadata["model_version"]})

    def test_real_model_predicts_coffee_record(self):
        metadata = json.loads(
            (ROOT / "deployment" / "model_metadata.json").read_text(encoding="utf-8")
        )
        response = self.client.post(
            "/predict",
            json={
                "species": "Arabica",
                "country_of_origin": "Ethiopia",
                "variety": "Heirloom",
                "processing_method": "Washed / Wet",
                "color": "Green",
                "number_of_bags": 300,
                "bag_weight": "60 kg",
                "harvest_year": "2024",
                "moisture": 0.12,
                "category_one_defects": 0,
                "quakers": 0,
                "category_two_defects": 0,
                "altitude_mean_meters": 1800,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["target"], "Total.Cup.Points")
        self.assertEqual(result["model_version"], metadata["model_version"])
        self.assertTrue(0 <= result["prediction"] <= 100)


if __name__ == "__main__":
    unittest.main()
