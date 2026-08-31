import unittest

from evaluate import classify, contains, metrics
from lcr import WEIGHTS, crop_box, response, select, zscore
from selectground import prediction


class CoreTests(unittest.TestCase):
    def test_coordinates(self):
        self.assertEqual(prediction("[500,250]", (800, 400))["point"], [400, 100])
        self.assertIsNone(prediction("unknown", (800, 400))["point"])

    def test_geometry(self):
        case = {"target": [40, 40, 60, 60], "type": "xyxy"}
        self.assertEqual(classify([50, 50], case, (100, 100)), "correct")
        self.assertEqual(classify([65, 50], case, (100, 100)), "localization_miss")
        self.assertEqual(classify([90, 50], case, (100, 100)), "semantic_miss")
        self.assertEqual(classify([101, 50], case, (100, 100)), "others")
        self.assertEqual(classify(None, case, (100, 100)), "others")
        self.assertTrue(contains([50, 50], [40, 40, 20, 20], "bbox"))
        self.assertTrue(contains([2, 2], [0, 0, 10, 0, 10, 10, 0, 10], "polygon"))

    def test_crop_coordinates(self):
        self.assertEqual(crop_box((0, 0), (1000, 800), .4), (0, 0, 400, 320))
        self.assertEqual(crop_box((999, 799), (1000, 800), .6), (400, 320, 1000, 800))
        self.assertEqual(response([200, 100], (100, 0, 300, 200)), "[500,500]")
        self.assertIsNone(response([300, 100], (100, 0, 300, 200)))

    def test_cross_view_selection(self):
        candidates = [
            {"name": "p0", "source_view": "full", "point": [10, 10]},
            {"name": "q0", "source_view": "q0", "point": [90, 90]},
        ]
        evidence = {
            "full": {"p0": ("[100,100]", -1.), "q0": ("[900,900]", -3.)},
            "q0": {"p0": ("[100,100]", -2.), "q0": ("[900,900]", -1.)},
        }
        # Both receive only the other view's evidence; equal scores retain view order.
        best = select(candidates, evidence, {"p0": 0., "q0": 0.}, (100, 100), (0., 0., 0.))
        self.assertEqual(best["name"], "p0")
        self.assertEqual(zscore([1., 1.]), [0., 0.])
        self.assertEqual(WEIGHTS["mmbench_gui_l2"], (.475, 1.625, .3))

    def test_denominator(self):
        result = metrics([{"label": "correct"}, {"label": "others"}])
        self.assertEqual(result["accuracy"], 50.)


if __name__ == "__main__":
    unittest.main()
