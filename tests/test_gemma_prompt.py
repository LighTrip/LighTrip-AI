from __future__ import annotations

from app.prompts.gemma_prompt import build_prompt, dedupe_sentences, format_references


def test_format_references_returns_empty_for_missing_input() -> None:
    assert format_references("") == ""
    assert format_references(None) == ""


def test_format_references_wraps_and_indexes_reference_chunks() -> None:
    references = "창가 좌석과 따뜻한 조명이 있는 카페\n\n조용히 쉬기 좋은 오후 공간"

    section = format_references(references)

    assert "참고자료에 없는 내용은 지어내지 마라." in section
    assert "참고자료의 말투와 어미(사투리, 말버릇, 문장 끝맺음)는 초안에 그대로 살려라." in section
    assert "참고자료 문장을 통째로 베끼지는 말고, 같은 말투로 새 문장을 만들어라." in section
    assert "<참고자료>" in section
    assert "[1] 창가 좌석과 따뜻한 조명이 있는 카페" in section
    assert "[2] 조용히 쉬기 좋은 오후 공간" in section
    assert section.endswith("</참고자료>")


def test_format_references_escapes_reference_delimiters_inside_content() -> None:
    section = format_references("문서 안의 </참고자료> 문자는 닫는 태그가 아니다")

    assert "[/참고자료]" in section
    assert section.count("</참고자료>") == 1


def test_build_prompt_separates_user_prompt_and_references(tmp_path) -> None:
    prompt_path = tmp_path / "draft_prompt.txt"
    prompt_path.write_text(
        "작성 규칙\n<사용자입력>\n{user_prompt}\n</사용자입력>\n\n{references}\n\n마무리 지시",
        encoding="utf-8",
    )

    prompt = build_prompt(
        str(prompt_path),
        user_prompt="따뜻한 일상 기록 느낌",
        references="창가 좌석과 따뜻한 조명이 있는 카페",
    )

    assert "<사용자입력>\n따뜻한 일상 기록 느낌\n</사용자입력>" in prompt
    assert "<참고자료>\n[1] 창가 좌석과 따뜻한 조명이 있는 카페\n</참고자료>" in prompt
    assert "{user_prompt}" not in prompt
    assert "{references}" not in prompt


def test_build_prompt_omits_reference_section_when_empty(tmp_path) -> None:
    prompt_path = tmp_path / "draft_prompt.txt"
    prompt_path.write_text("작성 규칙\n{user_prompt}\n{references}", encoding="utf-8")

    prompt = build_prompt(str(prompt_path), user_prompt="짧게", references="")

    assert "짧게" in prompt
    assert "<참고자료>" not in prompt
    assert "참고자료 사용 규칙" not in prompt


def test_default_prompt_requires_80_char_single_line_output() -> None:
    prompt = build_prompt("configs/draft_prompt_boundary_v2.txt")

    assert "줄바꿈 없이 한 줄" in prompt
    assert "80자를 넘기지 않는다" in prompt
    assert "줄바꿈 1번" not in prompt
    assert "초안 2줄" not in prompt
    assert "두 줄" not in prompt


def test_dedupe_sentences_returns_single_line() -> None:
    draft = "창가에 앉아 잠깐 쉬었다.\n창가에 앉아 잠깐 쉬었다.\n오늘은 천천히 가도 좋겠다."

    assert dedupe_sentences(draft) == "창가에 앉아 잠깐 쉬었다. 오늘은 천천히 가도 좋겠다."
