"""Tests for the OCR number repair and disclaimer stripping (no Tesseract needed)."""

from adalat_mt.data.pdf_recover import number_key, repair_numbers, strip_disclaimer


def test_number_key_ignores_ocr_confusions() -> None:
    assert number_key("1989") == number_key("989") == number_key("4989")


def test_repair_numbers_restores_dropped_ones() -> None:
    ocr = "वर्ष 985 में परीक्षा हुई और 24.04.989 को पत्र जारी किया, धारा 302 लागू।"
    layer = ["1985", "24.04.1989", "302"]
    fixed, changes = repair_numbers(ocr, layer)
    assert fixed == "वर्ष 1985 में परीक्षा हुई और 24.04.1989 को पत्र जारी किया, धारा 302 लागू।"
    assert changes == [("985", "1985"), ("24.04.989", "24.04.1989")]


def test_repair_numbers_keeps_unaligned_numbers() -> None:
    fixed, changes = repair_numbers("संख्या 55 और 77", ["55"])
    assert fixed == "संख्या 55 और 77"
    assert changes == []


def test_repair_numbers_maps_devanagari_digits() -> None:
    fixed, _ = repair_numbers("दिनांक २७.०५.२००३", ["27.05.2003"])
    assert fixed == "दिनांक 27.05.2003"


def test_strip_disclaimer_tolerates_garbled_heading() -> None:
    page = "2. अपील खारिज की जाती है।\nउद्घोष्णा\nक्षेत्रीय भाषा में अनुवादित निर्णय वादी के लिए है।"
    assert strip_disclaimer(page) == "2. अपील खारिज की जाती है।"


def test_repair_numbers_handles_dropped_digit_group() -> None:
    fixed, _ = repair_numbers("दिनांक 23..2009 के पत्र", ["23.11.2009"])
    assert fixed == "दिनांक 23.11.2009 के पत्र"
