"""Excel ingestion uses only synthetic workbooks, never private input data."""

from pathlib import Path
from unittest.mock import Mock
from zipfile import BadZipFile

import pytest
from openpyxl import Workbook, load_workbook

from sayari_poc.excel import read_all_sheets, read_sheet

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_list.xlsx"


def write_workbook(
    path: Path,
    rows: list[tuple[str | int | None, ...]],
    sheet: str = "portfolio",
) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.title = sheet
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()
    return path


def test_fixture_rows_accounted_for_and_diacritics_preserved() -> None:
    # Every fixture row is either read or reported, and names keep their native spelling.
    result = read_sheet(FIXTURE, "list_3")
    assert len(result.entities) == 3
    assert result.entities[0].name == "ZF Friedrichshafen"
    assert result.entities[0].address == "Löwentaler Straße 1"
    assert result.entities[0].country == "DEU"
    assert result.entities[1].name == "Peña Components"
    assert result.entities[2].name == "示例供应商"
    assert {entity.sheet for entity in result.entities} == {"list_3"}
    assert [entity.row_number for entity in result.entities] == [2, 3, 4]
    assert result.exceptions == [
        {"row_number": "5", "sheet": "list_3", "reason": "missing name", "kind": "dropped"},
    ]
    accounted = {entity.row_number for entity in result.entities} | {
        int(error["row_number"]) for error in result.exceptions
    }
    assert accounted == {2, 3, 4, 5}


def test_missing_header_identifies_required_field_without_sheet_value() -> None:
    # The missing-header error names the required field but repeats nothing from the sheet.
    with pytest.raises(ValueError, match="Selected sheet.*country"):
        read_sheet(FIXTURE, "bad_headers")


def test_workbook_bytes_and_modification_time_unchanged() -> None:
    # Reading the supplied workbook changes neither its bytes nor its modification time.
    before = FIXTURE.read_bytes()
    modified = FIXTURE.stat().st_mtime_ns
    read_sheet(FIXTURE, "list_3")
    with pytest.raises(ValueError):
        read_all_sheets(FIXTURE)
    assert FIXTURE.read_bytes() == before
    assert FIXTURE.stat().st_mtime_ns == modified


@pytest.mark.parametrize("country", ["DE", "deu", "Deu", "DEUU", "D3U", "DÉU", "ZZZ", 123])
def test_invalid_country_retains_entity_and_records_exception(
    country: str | int,
    tmp_path: Path,
) -> None:
    # A bad country code is recorded as a diagnostic, and the named supplier is kept.
    path = write_workbook(
        tmp_path / "invalid.xlsx",
        [
            ("name", "address", "country"),
            ("Supplier", "Address", country),
        ],
    )
    result = read_sheet(path, "portfolio")
    assert len(result.entities) == 1
    assert result.entities[0].country is None
    assert result.exceptions == [
        {
            "row_number": "2",
            "sheet": "portfolio",
            "reason": "invalid country: expected an ISO 3166-1 alpha-3 code",
            "kind": "annotated",
        }
    ]


def test_whitespace_collapsed_without_changing_case_or_diacritics(tmp_path: Path) -> None:
    # Collapsing whitespace keeps the original case and accented characters.
    path = write_workbook(
        tmp_path / "whitespace.xlsx",
        [
            ("country", " name ", "address", "notes"),
            (" DEU ", "  Müller\t &  Söhne\nGmbH  ", " Straße\n  10  ", "ignored"),
        ],
    )
    entity = read_sheet(path, "portfolio").entities[0]
    assert entity.name == "Müller & Söhne GmbH"
    assert entity.address == "Straße 10"
    assert entity.country == "DEU"


def test_missing_optional_values_are_none(tmp_path: Path) -> None:
    # Empty optional cells read as None and produce no diagnostics.
    path = write_workbook(
        tmp_path / "optional.xlsx",
        [
            ("name", "address", "country"),
            ("Supplier", "  ", None),
        ],
    )
    result = read_sheet(path, "portfolio")
    assert result.entities[0].address is None
    assert result.entities[0].country is None
    assert result.exceptions == []


def test_malformed_cells_do_not_drop_other_rows(tmp_path: Path) -> None:
    # A malformed cell affects only its own row; later named rows are still read.
    path = write_workbook(
        tmp_path / "malformed.xlsx",
        [
            ("name", "address", "country"),
            ("  ", "Address", "DEU"),
            (42, "Address", "DEU"),
            ("Supplier", 42, "DEU"),
            ("Next", "Address", "USA"),
        ],
    )
    result = read_sheet(path, "portfolio")
    assert [entity.name for entity in result.entities] == ["Supplier", "Next"]
    assert result.entities[0].address is None
    assert [(error["row_number"], error["reason"]) for error in result.exceptions] == [
        ("2", "missing name"),
        ("3", "invalid name: expected text"),
        ("4", "invalid address: expected text"),
    ]


def test_diagnostics_distinguish_dropped_rows_without_double_counting(tmp_path: Path) -> None:
    # Warnings on optional fields annotate a row but never count it as dropped.
    path = write_workbook(
        tmp_path / "accounting.xlsx",
        [
            ("name", "address", "country"),
            (None, "Address", "DEU"),
            ("Annotated supplier", 42, "Germany"),
            ("Valid supplier", "Address", "USA"),
        ],
    )
    result = read_sheet(path, "portfolio")
    assert len(result.entities) == 2
    assert [(issue["row_number"], issue["kind"]) for issue in result.exceptions] == [
        ("2", "dropped"),
        ("3", "annotated"),
        ("3", "annotated"),
    ]
    dropped = sum(issue["kind"] == "dropped" for issue in result.exceptions)
    assert len(result.entities) + dropped == 3


def test_blank_row_inside_sheet_is_accounted_for(tmp_path: Path) -> None:
    # A blank row inside the sheet gets a diagnostic, and later suppliers keep their row numbers.
    path = write_workbook(
        tmp_path / "blank.xlsx",
        [
            ("name", "address", "country"),
            (None, None, None),
            ("Supplier", "Address", "DEU"),
        ],
    )
    result = read_sheet(path, "portfolio")
    assert result.exceptions[0]["row_number"] == "2"
    assert result.exceptions[0]["reason"] == "missing name"
    assert result.entities[0].row_number == 3


def test_read_all_sheets_includes_all_portfolios_and_arbitrary_names(tmp_path: Path) -> None:
    # Every worksheet is read in workbook order, and any sheet name works as a portfolio name.
    path = tmp_path / "portfolios.xlsx"
    workbook = Workbook()
    for index, name in enumerate(("list_1", "list_2", "list_3", "new_client")):
        sheet = workbook.active if index == 0 else workbook.create_sheet()
        assert sheet is not None
        sheet.title = name
        sheet.append(("name", "address", "country"))
        sheet.append(("Synthetic " + name, "Address", "DEU"))
    workbook.save(path)
    workbook.close()
    results = read_all_sheets(path)
    assert list(results) == ["list_1", "list_2", "list_3", "new_client"]
    for name, result in results.items():
        assert result.entities[0].sheet == name
        assert result.entities[0].name == "Synthetic " + name


def test_read_all_sheets_rejects_invalid_sheet_instead_of_skipping() -> None:
    # A malformed worksheet raises an error instead of being skipped.
    with pytest.raises(ValueError, match="Selected sheet: missing headers"):
        read_all_sheets(FIXTURE)


def test_duplicate_required_header_rejected(tmp_path: Path) -> None:
    # A duplicated required header is rejected rather than quietly picking one of the columns.
    path = write_workbook(
        tmp_path / "duplicate.xlsx",
        [
            ("name", "address", "country", "name"),
        ],
    )
    with pytest.raises(ValueError, match="Selected sheet.*duplicate.*name"):
        read_sheet(path, "portfolio")


def test_empty_sheet_rejected_with_sheet_name(tmp_path: Path) -> None:
    # A worksheet with no header row fails input validation.
    path = write_workbook(tmp_path / "empty.xlsx", [])
    with pytest.raises(ValueError, match="Selected sheet.*missing.*name"):
        read_sheet(path, "portfolio")


def test_header_only_sheet_is_valid_empty_portfolio(tmp_path: Path) -> None:
    # A sheet with valid headers and no data rows is a valid, empty portfolio.
    path = write_workbook(tmp_path / "headers.xlsx", [("name", "address", "country")])
    result = read_sheet(path, "portfolio")
    assert result.entities == []
    assert result.exceptions == []


def test_missing_sheet_names_requested_sheet() -> None:
    # Asking for a worksheet that does not exist raises the expected input error.
    with pytest.raises(ValueError, match="Selected sheet not found"):
        read_sheet(FIXTURE, "absent")


@pytest.mark.parametrize("sheet", ["list_3", "bad_headers", "absent"])
def test_read_only_workbook_closed_on_success_and_failure(
    sheet: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The read-only workbook is closed on every exit path.
    workbook = load_workbook(FIXTURE, read_only=True, data_only=True)
    close = Mock(wraps=workbook.close)
    monkeypatch.setattr(workbook, "close", close)
    loader = Mock(return_value=workbook)
    monkeypatch.setattr("sayari_poc.excel.load_workbook", loader)
    if sheet == "list_3":
        read_sheet(FIXTURE, sheet)
    else:
        with pytest.raises(ValueError):
            read_sheet(FIXTURE, sheet)
    assert loader.call_args.kwargs["read_only"] is True
    close.assert_called_once_with()


def test_missing_file_raises(tmp_path: Path) -> None:
    # A missing workbook raises an error instead of producing empty evidence.
    with pytest.raises(FileNotFoundError):
        read_sheet(tmp_path / "missing.xlsx", "portfolio")


def test_corrupt_workbook_raises(tmp_path: Path) -> None:
    # A corrupt workbook raises an error instead of producing empty evidence.
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(b"not an Excel workbook")
    with pytest.raises(BadZipFile):
        read_all_sheets(path)
