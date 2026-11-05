import unittest

from fastapi.testclient import TestClient

from api.main import create_app


class StubModelLoader:
    metadata = {"model_version": "test", "target": "Total.Cup.Points"}

    def load(self):
        return None

    def predict_records(self, records):
        return [82.5] * len(records)


class CoffeeApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(StubModelLoader()))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": "test"})

    def test_single_prediction_accepts_coffee_features(self):
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
        self.assertEqual(response.json()["prediction"], 82.5)
        self.assertEqual(response.json()["target"], "Total.Cup.Points")

    def test_batch_size_is_bounded(self):
        response = self.client.post(
            "/predict/batch",
            json={"records": [{}] * 129},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
