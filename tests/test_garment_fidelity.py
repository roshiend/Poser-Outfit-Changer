import unittest

import numpy as np
from PIL import Image

from pipeline.garment_extract import (
    GarmentExtractionError,
    analyse_garment,
    extract_garment_from_person,
)
from pipeline.outfit_quality import assess_outfit_preservation, masked_rgb_mae


class GarmentRoutingTests(unittest.TestCase):
    def _parse(self):
        return np.zeros((100, 80), dtype=np.uint8)

    def test_auto_detects_upper_only(self):
        parse = self._parse()
        parse[20:55, 20:60] = 4
        info = analyse_garment(parse, "auto", out_size=(80, 100))
        self.assertEqual(info.resolved_type, "upper_body")
        self.assertGreater(info.confidence, 0.5)

    def test_auto_detects_lower_only(self):
        parse = self._parse()
        parse[45:95, 22:58] = 6
        info = analyse_garment(parse, "auto", out_size=(80, 100))
        self.assertEqual(info.resolved_type, "lower_body")

    def test_auto_detects_full_outfit_when_upper_and_lower_visible(self):
        parse = self._parse()
        parse[12:48, 18:62] = 4
        parse[48:95, 22:58] = 6
        info = analyse_garment(parse, "auto", out_size=(80, 100))
        self.assertEqual(info.resolved_type, "dresses")

    def test_auto_detects_parser_labelled_dress(self):
        parse = self._parse()
        parse[12:90, 18:62] = 7
        info = analyse_garment(parse, "auto", out_size=(80, 100))
        self.assertEqual(info.resolved_type, "dresses")

    def test_no_garment_never_falls_back_to_full_person(self):
        person = Image.new("RGB", (80, 100), (120, 80, 40))
        parse = self._parse()
        with self.assertRaises(GarmentExtractionError):
            extract_garment_from_person(person, parse, "auto", out_size=(80, 100))

    def test_extraction_keeps_white_background(self):
        person = Image.new("RGB", (80, 100), (10, 20, 200))
        parse = self._parse()
        parse[20:70, 20:60] = 4
        garment, info = extract_garment_from_person(
            person,
            parse,
            "auto",
            out_size=(80, 100),
            return_analysis=True,
        )
        arr = np.asarray(garment)
        self.assertEqual(info.resolved_type, "upper_body")
        self.assertTrue(np.all(arr[0, 0] == 255))
        self.assertLess(float(arr.mean()), 254.0)


class OutfitQualityTests(unittest.TestCase):
    def test_masked_mae_is_zero_for_same_image(self):
        image = Image.new("RGB", (20, 20), "white")
        mask = np.ones((20, 20), dtype=bool)
        self.assertEqual(masked_rgb_mae(image, image, mask), 0.0)

    def test_large_protected_and_background_change_warns(self):
        base = Image.new("RGB", (20, 20), "white")
        changed = Image.new("RGB", (20, 20), "black")
        parse = np.zeros((20, 20), dtype=np.uint8)
        parse[5:10, 5:10] = 11
        parse[2:5, 5:10] = 2
        diagnostics = assess_outfit_preservation(base, changed, parse)
        self.assertIsNotNone(diagnostics.face_hair_mae)
        self.assertIsNotNone(diagnostics.background_mae)
        self.assertGreaterEqual(len(diagnostics.warnings), 2)


if __name__ == "__main__":
    unittest.main()
