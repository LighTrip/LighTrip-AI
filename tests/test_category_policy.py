from __future__ import annotations

import unittest

from app.services.category_policy import category_fallback_reason, normalize_category


class CategoryPolicyTest(unittest.TestCase):
    def test_allows_known_operational_categories(self) -> None:
        for category in ("카페", "식당", "술집", "문화", "운동", "쇼핑", "공원", "기타"):
            self.assertIsNone(category_fallback_reason(category))

    def test_normalizes_outer_whitespace_and_quotes(self) -> None:
        self.assertEqual(normalize_category(' "카페" '), "카페")
        self.assertIsNone(category_fallback_reason(" 카페 "))

    def test_fallback_reasons_for_invalid_categories(self) -> None:
        self.assertEqual(category_fallback_reason(None), "missing_category")
        self.assertEqual(category_fallback_reason(""), "empty_category")
        self.assertEqual(category_fallback_reason("여행"), "category_outside_allowed_set")
        self.assertEqual(category_fallback_reason("카페, 식당"), "category_outside_allowed_set")


if __name__ == "__main__":
    unittest.main()
