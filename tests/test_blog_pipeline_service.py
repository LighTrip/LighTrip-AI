from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch


def configure_required_env() -> None:
    os.environ.setdefault("GEMMA_MODEL_PATH", "models/gemma-4-E2B-it-Q4_K_S.gguf")
    os.environ.setdefault("GEMMA_MMPROJ_PATH", "models/mmproj-F16.gguf")
    os.environ.setdefault("GEMMA_PROMPT_PATH", "configs/draft_prompt_boundary_v2.txt")
    os.environ.setdefault("GEMMA_N_CTX", "1024")
    os.environ.setdefault("GEMMA_MAX_TOKENS", "128")
    os.environ.setdefault("GEMMA_TEMPERATURE", "0.2")
    os.environ.setdefault("GEMMA_TOP_P", "0.9")
    os.environ.setdefault("GEMMA_TOP_K", "40")
    os.environ.setdefault("GEMMA_REPEAT_PENALTY", "1.1")
    os.environ.setdefault("GEMMA_STOP_TOKENS", "<end_of_turn>")
    os.environ.setdefault("GEMMA_N_GPU_LAYERS", "0")
    os.environ.setdefault("GEMMA_MAIN_GPU", "0")
    os.environ.setdefault("GEMMA_OFFLOAD_KQV", "false")
    os.environ.setdefault("GEMMA_MMPROJ_USE_GPU", "false")
    os.environ.setdefault(
        "CATEGORY_ARTIFACT_PATH",
        "experiments/category_classifier/artifacts/places365_2_manual_full_calibrated/calibrated_linear_svm_tfidf.joblib",
    )
    os.environ.setdefault("CATEGORY_UNKNOWN_LABEL", "기타")


class BlogPipelineServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_required_env()
        from app.services import blog_pipeline_service as service
        from app.services.category_service import CategoryPrediction
        from app.services.gemma_service import GemmaDirectResult, parse_direct_output

        cls.service = service
        cls.CategoryPrediction = CategoryPrediction
        cls.GemmaDirectResult = GemmaDirectResult
        cls.parse_direct_output = staticmethod(parse_direct_output)

    def test_direct_parser_preserves_empty_category_for_fallback(self) -> None:
        result = self.parse_direct_output(
            '{"draft":"테이블에 앉아 잠깐 쉬었다.\\n오늘 기록으로 남겨두고 싶다.","category":""}'
        )

        self.assertEqual(result.draft, "테이블에 앉아 잠깐 쉬었다.\n오늘 기록으로 남겨두고 싶다.")
        self.assertIsNone(result.category)
        self.assertEqual(result.raw_category, "")

    def test_direct_parser_rejects_category_outside_allowed_set(self) -> None:
        result = self.parse_direct_output(
            '{"draft":"새로운 곳을 천천히 걸었다.\\n오늘 기록으로 남겨두고 싶다.","category":"여행"}'
        )

        self.assertIsNone(result.category)
        self.assertEqual(result.raw_category, "여행")

    def test_uses_gemma_category_when_valid(self) -> None:
        direct_result = self.GemmaDirectResult(
            draft="커피 향이 좋아 잠깐 쉬어갔다.\n창가 자리가 유난히 편했다.",
            category="카페",
            raw_category="카페",
            raw_output='{"draft":"...","category":"카페"}',
            parse_status="json",
        )

        with patch.object(
            self.service,
            "generate_blog_draft_and_category_from_bytes",
            return_value=direct_result,
        ), patch.object(self.service, "classify_text") as classify_text:
            result = self.service.generate_draft_and_classify(
                llm=Mock(),
                image_bytes=b"image",
                filename="sample.jpg",
            )

        self.assertEqual(result.draft, direct_result.draft)
        self.assertEqual(result.category, "카페")
        self.assertEqual(result.category_source, "gemma_direct")
        classify_text.assert_not_called()

    def test_falls_back_to_svm_when_category_is_missing(self) -> None:
        direct_result = self.GemmaDirectResult(
            draft="넓은 잔디 옆 벤치에 앉아 잠시 쉬었다.\n햇살이 좋아 천천히 걷고 싶은 날이었다.",
            category=None,
            raw_category=None,
            raw_output='{"draft":"..."}',
            parse_status="json",
        )
        svm_prediction = self.CategoryPrediction(
            label="공원",
            raw_label="공원",
            confidence=0.83,
            score=0.83,
            scores={"공원": 0.83},
            model="calibrated_linear_svm",
        )

        with patch.object(
            self.service,
            "generate_blog_draft_and_category_from_bytes",
            return_value=direct_result,
        ), patch.object(self.service, "classify_text", return_value=svm_prediction) as classify_text:
            result = self.service.generate_draft_and_classify(
                llm=Mock(),
                image_bytes=b"image",
                filename="sample.jpg",
            )

        self.assertEqual(result.category, "공원")
        self.assertEqual(result.category_source, "svm_fallback")
        self.assertEqual(result.fallback_reason, "missing_category")
        classify_text.assert_called_once_with(
            text=direct_result.draft,
            unknown_threshold=None,
        )

    def test_falls_back_to_svm_when_category_is_outside_allowed_set(self) -> None:
        direct_result = self.GemmaDirectResult(
            draft="새로운 곳을 둘러보며 천천히 걸었다.\n오늘의 기록으로 남겨두고 싶은 순간이었다.",
            category=None,
            raw_category="여행",
            raw_output='{"draft":"...","category":"여행"}',
            parse_status="json",
        )
        svm_prediction = self.CategoryPrediction(
            label="기타",
            raw_label="공원",
            confidence=0.31,
            score=0.31,
            scores={"공원": 0.31},
            model="calibrated_linear_svm",
        )

        with patch.object(
            self.service,
            "generate_blog_draft_and_category_from_bytes",
            return_value=direct_result,
        ), patch.object(self.service, "classify_text", return_value=svm_prediction):
            result = self.service.generate_draft_and_classify(
                llm=Mock(),
                image_bytes=b"image",
                filename="sample.jpg",
                unknown_threshold=0.49,
            )

        self.assertEqual(result.category, "기타")
        self.assertEqual(result.category_source, "svm_fallback")
        self.assertEqual(result.fallback_reason, "category_outside_allowed_set")


if __name__ == "__main__":
    unittest.main()
