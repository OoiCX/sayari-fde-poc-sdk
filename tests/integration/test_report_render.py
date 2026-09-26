"""Task 12 integration contracts: synthetic workbook, cached stages, no network."""

import html as html_module
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

import duckdb
import httpx
import pytest

from sayari_poc import analysis, pipeline, presentation, report
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.enrich import fetch_profiles
from sayari_poc.identity import psa_exposure
from sayari_poc.models import Findings
from sayari_poc.transport import OfflineCacheMiss
from tests.ontology_support import pipeline_ontology as pipeline_ontology
from tests.pipeline_support import _node, _prepare, _upstream

pytestmark = pytest.mark.usefixtures("forbid_external_stages", "pipeline_ontology")


def _html(findings: Findings, tmp_path: Path) -> str:
    path = tmp_path / "render.html"
    report.render_report(findings, path)
    return path.read_bytes().decode("utf-8")


def _shared_findings(html: str) -> str:
    return html.split('<div id="convergence-table"', 1)[1].split('<nav id="entity-pagination"', 1)[
        0
    ]


def test_every_upstream_result_reaches_warehouse(settings: Settings, tmp_path: Path) -> None:
    # Every retrieval outcome survives pipeline assembly, with one status per input row.
    configured, cache = _prepare(
        settings,
        tmp_path,
        {
            "list_3": [(f"Input {i}", f"s{i}") for i in range(1, 5)] + [("No match", None)],
        },
    )
    _upstream(cache, configured, "s1", {"only": _node("only", 1, [])})
    _upstream(cache, configured, "s2", {}, partial=True)
    cache.put(
        cache.key(
            httpx.Request(
                "GET",
                configured.sayari_api_base + "/v1/supply_chain/upstream/s4",
                params={
                    "max_depth": configured.max_upstream_depth,
                    "limit": configured.upstream_limit,
                },
            ),
        ),
        json.dumps(
            {
                "filters": {},
                "explored_count": 0,
                "data": {"paths": [], "entities": []},
                "partial_results": False,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    result = pipeline.run_pipeline(configured, offline=True, output_dir=tmp_path / "out")
    with duckdb.connect(str(configured.duckdb_path)) as con:
        statuses = con.execute(
            "SELECT coverage_status FROM suppliers ORDER BY row_number"
        ).fetchall()
        assert statuses == [("assessed",), ("partial",), ("no_data",), ("error",), (None,)]
        assert (
            analysis.coverage_summary(con)
            == result.coverage
            == {
                "list_3": {
                    "assessed": 1,
                    "partial": 1,
                    "no_data": 1,
                    "error": 1,
                    "not_attempted": 1,
                }
            }
        )
    assert [r["coverage_status"] for r in cast(list[dict[str, Any]], result.suppliers)] == [
        s[0] for s in statuses
    ]
    assert cast(list[dict[str, Any]], result.suppliers)[-1]["psa_status"] == "not_resolved"
    assert (
        cast(list[dict[str, Any]], result.suppliers)[-1]["psa_count"]
        is cast(list[dict[str, Any]], result.suppliers)[-1]["psa_risky"]
        is None
    )
    # Malformed SDK payloads fail closed as a ValidationError.
    assert any(
        e["stage"] == "upstream" and e["error_type"] == "ValidationError" for e in result.exceptions
    )
    html = _html(result, tmp_path)
    for label in presentation.COVERAGE_LABELS.values():
        assert label in html


def test_upstream_offline_miss_is_not_silently_successful(
    graph: tuple[Settings, ResponseCache],
    tmp_path: Path,
) -> None:
    # An upstream cache miss becomes explicit failure evidence.
    configured, cache = graph
    key = cache.key(
        httpx.Request(
            "GET",
            configured.sayari_api_base + "/v1/supply_chain/upstream/s1",
            params={
                "max_depth": configured.max_upstream_depth,
                "limit": configured.upstream_limit,
            },
        ),
    )
    (cache.cache_dir / f"{key}.json").unlink()
    with pytest.raises(OfflineCacheMiss):
        pipeline.run_pipeline(configured, offline=True, output_dir=tmp_path / "out")
    data = json.loads((tmp_path / "out/findings.json").read_text(encoding="utf-8"))
    assert any(
        e["stage"] == "upstream" and e["error_type"] == "OfflineCacheMiss"
        for e in data["exceptions"]
    )
    assert data["coverage"]["list_3"]["error"] == 2  # Both input rows retain the failed attempt.


def test_supplier_diagnostics_and_psa_attach_by_row(
    graph: tuple[Settings, ResponseCache],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rows that share one entity keep their own diagnostics and PSA availability.
    original = fetch_profiles

    def profiles(client: Any, rows: Any) -> Any:
        result = original(client, rows)
        duplicate = next(row for row in rows if row.input_name == "Duplicate input")
        duplicate.profile_error = "Synthetic row profile unavailable"
        duplicate.profile_error_type = "SyntheticProfileError"
        duplicate.candidate_errors = {1: "Malformed alternate candidate"}
        return result

    monkeypatch.setattr(pipeline, "fetch_profiles", profiles)
    result = pipeline.run_pipeline(
        graph[0], all_sheets=True, offline=True, output_dir=tmp_path / "out"
    )
    same_id = [
        r
        for r in cast(list[dict[str, Any]], result.suppliers)
        if r["portfolio"] == "list_3" and r["entity_id"] == "s1"
    ]
    assert len(same_id) == 2 and same_id[0]["row_number"] != same_id[1]["row_number"]
    assert same_id[0]["psa_status"] == "available"
    assert type(same_id[0]["psa_count"]) is int and type(same_id[0]["psa_risky"]) is bool
    assert same_id[1]["psa_status"] == "unavailable"
    assert same_id[1]["psa_count"] is same_id[1]["psa_risky"] is same_id[1]["profile"] is None
    assert same_id[1]["profile_status"] == "error"
    assert same_id[1]["profile_error_type"] == "SyntheticProfileError"
    assert same_id[1]["candidate_errors"] == {"1": "Malformed alternate candidate"}
    assert same_id[1]["input_address"] == "Synthetic address"
    assert same_id[0]["profile"]["degree"] == 8
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    html = _html(result, tmp_path)
    assert "Synthetic row profile unavailable" in html and "Malformed alternate candidate" in html


@pytest.mark.parametrize("drift", ["entity", "row"])
def test_psa_key_drift_fails_loudly(
    graph: tuple[Settings, ResponseCache],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    # A mismatched PSA row key stops publication instead of attaching evidence to the wrong row.
    original = psa_exposure

    def exposure(*args: Any) -> Any:
        frame = original(*args)
        frame.loc[0, "entity_id" if drift == "entity" else "row_number"] = (
            "wrong-identity" if drift == "entity" else 900
        )
        return frame

    monkeypatch.setattr(pipeline, "psa_exposure", exposure)
    with pytest.raises(ValueError, match="PSA"):
        pipeline.run_pipeline(graph[0], offline=True, output_dir=tmp_path / "out")
    assert not (tmp_path / "out/findings.json").exists()


def test_support_queries_scope_distinct_factors_funnel_and_order(
    findings: Findings,
    graph: tuple[Settings, ResponseCache],
) -> None:
    # The support queries agree on scoped memberships, risk factors and funnel counts.
    with duckdb.connect(str(graph[0].duckdb_path)) as con:
        evidence = analysis.shared_nodes(con, 2, "list_3").to_dict("records")
        assert [r["upstream_id"] for r in evidence] == ["n-top", "n-boundary"]
        assert [s["entity_id"] for s in evidence[0]["suppliers"]] == [f"s{i}" for i in range(1, 7)]
        assert evidence[0]["suppliers"][0]["input_names"] == ["Duplicate input", "Input 1"]
        assert evidence[0]["severe_factors"] == ["sanctioned_synthetic"]
        assert evidence[0]["other_factors"] == ["ilab_forced_labor", "mystery"]
        secondary = analysis.shared_nodes(con, 2, "another_portfolio").to_dict("records")
        assert [s["entity_id"] for s in secondary[0]["suppliers"]] == ["t1", "t2"]
        hubs = analysis.suppressed_hubs(con, 2, "list_3", 10).to_dict("records")
        assert [h["upstream_id"] for h in hubs] == ["h-severe", "h-broad"]
        assert hubs[1]["severe_factors"] == []  # Suppressed before severity filtering.
        assert all(h["country_count"] > 2 for h in hubs)
        assert analysis.suppressed_hubs(con, 2, "list_3", 1).to_dict("records") == hubs[:1]
        funnel = analysis.convergence_funnel(con, 2, "list_3")
        assert list(funnel.values()) == [6, 6, 5, 2, 3, 1, 2]
        assert funnel["after_severity_filter"] == len(analysis.shared_nodes(con, 2, "list_3"))
        assert funnel["shared_nodes"] == funnel["suppressed_hubs"] + funnel["after_hub_suppression"]
        assert all(type(value) is int for value in funnel.values())
        assert analysis.shared_nodes(con, 2, "missing").empty
        assert not any(analysis.convergence_funnel(con, 2, "missing").values())
        for portfolio in ["another_portfolio", "list_3"]:
            assert analysis.convergence_funnel(con, 2, portfolio)["after_severity_filter"] == len(
                analysis.shared_nodes(con, 2, portfolio)
            )
        assert evidence == analysis.shared_nodes(con, 2, "list_3").to_dict("records")
        assert hubs == analysis.suppressed_hubs(con, 2, "list_3", 10).to_dict("records")
        json.dumps([evidence, hubs, funnel], allow_nan=False)


@pytest.mark.parametrize("helper", ["convergence_funnel", "shared_nodes", "suppressed_hubs"])
def test_support_queries_reject_unclassified(
    findings: Findings,
    graph: tuple[Settings, ResponseCache],
    helper: str,
) -> None:
    # Support queries refuse evidence whose ontology classification is missing.
    with duckdb.connect(str(graph[0].duckdb_path)) as con:
        con.execute("UPDATE risk_factors SET is_severe=NULL")
        with pytest.raises(analysis.UnclassifiedRiskFactors):
            args = (con, 2, "list_3", 10) if helper == "suppressed_hubs" else (con, 2, "list_3")
            getattr(analysis, helper)(*args)


def test_unresolved_candidates_cannot_borrow_other_portfolio_evidence(
    findings: Findings,
    graph: tuple[Settings, ResponseCache],
) -> None:
    # Candidates that were not accepted cannot inherit another portfolio's convergence.
    with duckdb.connect(str(graph[0].duckdb_path)) as con:
        con.execute(
            "INSERT INTO suppliers (portfolio,row_number,entity_id,input_name,"
            "resolution_status) VALUES ('isolated',1,'s1','Candidate','weak'),"
            "('isolated',2,'s2','Candidate','error')"
        )
        assert not any(analysis.convergence_funnel(con, 2, "isolated").values())
        assert analysis.shared_nodes(con, 2, "isolated").empty
        assert analysis.suppressed_hubs(con, 2, "isolated", 10).empty


def test_full_findings_and_supplier_evidence(findings: Findings, tmp_path: Path) -> None:
    # Findings keep the approved evidence contract and none of the removed features.
    payload = findings.model_dump(mode="json")
    assert "restricted_party_hits" not in payload and "control_comparison" not in payload
    assert "watchlist_coverage" not in cast(dict[str, Any], findings.manifest)
    assert "control_portfolio" not in cast(dict[str, Any], findings.manifest)
    html = _html(findings, tmp_path)
    for phrase in [
        "Input 1",
        "not proof of direct supply",
        "a screening choice made here",
        "not a Sayari assessment",
        "logistics providers, trading intermediaries or large distributors",
        # The methodology section no longer restates these; the glossary carries the binding one, so
        # assert it still reaches the reader from there.
        "trade-record evidence, not proof of direct supply",
    ]:
        assert phrase in html
    for removed in ["Restricted-party result", "Control comparison", "designation context"]:
        assert removed not in html
    # Run-specific measurements were removed from this section: it states method and configuration,
    # not what this particular corpus happened to measure.
    text = _visible_text(_section(html, "methodology"))
    for measurement in [
        "of 721 published factors",
        "not the proportion of entities at risk",
        "entities respectively",
        "Distinct factors seen in this evidence",
        "These figures describe this evidence only",
    ]:
        assert measurement not in text
    assert re.search(r"\(about \d+%\)", text) is None
    assert re.search(r"keep [\d, and]+ entities\.", text) is None


def test_methodology_states_the_sayari_api_configuration(
    findings: Findings, tmp_path: Path
) -> None:
    """The section must say how Sayari was called, from the run's own settings.

    Asserted against the manifest rather than literals, so changing a bound updates
    this sentence and a stale one fails here instead of shipping.
    """
    text = _visible_text(_section(_html(findings, tmp_path), "methodology"))
    manifest = cast(dict[str, Any], findings.manifest)
    depth = manifest["max_upstream_depth"]
    assert f"The trade search runs to a depth of {depth}." in text
    # Sayari's path array begins at the T1-T2 edge, so the root is Tier 1 and depth N reaches Tier
    # N+1. A reader asking why paths stop at Tier 4 is answered here.
    assert f"this limit allows paths as far as Tier {depth + 1}" in text
    # The section states what this run did, not what the API would have permitted: no optional
    # parameter it declined to send, and no accepted range it sat inside.
    for permission in ["Sayari accepts", "optional", "are not sent", "default matching"]:
        assert permission not in text
    assert f"at most {manifest['upstream_limit']} supply chain paths" in text
    assert f"at most {manifest['path_max_per_node']} example paths" in text
    # Name the endpoints actually called, so the SDK path is auditable from the report.
    for endpoint in ["resolution", "entity_summary", "upstream_trade_traversal"]:
        assert endpoint in text
    # The provenance claim the section leads with: the data is Sayari's own.
    assert "comes from Sayari’s own data" in text
    # The fields actually sent to resolution.
    assert "matched on its name, address and country" in text


@pytest.mark.parametrize("partial", [False, True])
def test_search_coverage_reports_the_partial_flag_without_claiming_completeness(
    findings: Findings, tmp_path: Path, partial: bool
) -> None:
    # Coverage wording reports the partial flag Sayari returned without promising completeness.
    supplier = cast(dict[str, Any], findings.suppliers[0])
    supplier["upstream_partial_results"] = partial
    supplier["upstream_entity_count"] = 12
    entries = _supplier_entries(_html(findings, tmp_path))
    assert entries
    for entry in entries.values():
        coverage = _visible_text(
            entry.split("<h3>Retrieval coverage</h3>", 1)[1].split("</div>", 1)[0]
        )
        assert not re.search(r"\bcomplete(?:d|ly)?\b", coverage, re.I)
    expected = (
        "Sayari flagged this search as partial."
        if partial
        else "Sayari did not flag this search as partial."
    )
    assert expected in _visible_text(entries["supplier-1"])


@pytest.mark.parametrize("limit", [7, 23])
def test_path_limit_wording_uses_the_manifest_and_does_not_promise_a_finished_search(
    findings: Findings, tmp_path: Path, limit: int
) -> None:
    # The path limit shown comes from the manifest and never implies the search was exhausted.
    findings.manifest["upstream_limit"] = limit
    html = _html(findings, tmp_path)
    method = _visible_text(_section(html, "methodology"))
    coverage = _visible_text(_section(html, "coverage"))
    assert f"at most {limit} supply chain paths" in method
    assert "the search finished inside those limits" not in method
    assert "Sufficient upstream evidence" not in coverage
    assert "reached its own limits before it finished" not in coverage
    assert "or because Sayari flagged the upstream search as partial" in coverage
    for section in (method, coverage):
        assert "Entities along those paths can outnumber the path limit" in section
        assert "A search can stop at the path limit without being flagged as partial" in section


def test_delivered_methodology_makes_no_promise_the_path_cap_cannot_keep() -> None:
    """The section must not promise coverage the per-entity example cap forbids.

    The shipped sentence said every supplier with retrieved path evidence is
    represented. PATH_MAX_PER_NODE caps stored examples at six, and the committed
    evidence holds three supplier/entity pairs that have path evidence and no stored
    example, so the promise was false. The cap is now disclosed instead. Asserted
    against the delivered artifact, which is what a reader actually receives.
    """
    delivered = (Path(__file__).resolve().parents[2] / "docs/report.html").read_text(
        encoding="utf-8"
    )
    text = _visible_text(_section(delivered, "methodology"))
    assert "every supplier with retrieved path evidence is represented" not in text
    assert "some of them have no example in this report" in text


def test_report_datetime_renders_a_readable_utc_date() -> None:
    """The meta line must read as a date, and must not depend on the host's zone.

    The stored form is ISO-8601 with microseconds, which reads as machine output and
    invites a reader in another zone to conclude the date is wrong. Converting to
    local time would fix the appearance and break byte-identical replay, so the
    instant is canonicalised to UTC and labelled.
    """
    assert (
        report.report_datetime("2026-09-24T01:52:59.468774+00:00") == "24 September 2026, 01:52 UTC"
    )
    # A recorded offset is canonicalised, never rendered as somebody's local clock.
    assert report.report_datetime("2026-09-24T09:52:59+08:00") == "24 September 2026, 01:52 UTC"
    # An unreadable or absent value is shown as recorded rather than invented.
    assert report.report_datetime("not a date") == "not a date"
    assert report.report_datetime(None) == ""


def test_report_meta_shows_the_formatted_date_and_no_machine_timestamp(
    findings: Findings, tmp_path: Path
) -> None:
    # The report shows the stored time formatted as a date, with no raw microseconds.
    html = _html(findings, tmp_path)
    expected = report.report_datetime(findings.generated_at)
    assert f"Report content generated: {expected}" in html
    # No microsecond ISO timestamp survives anywhere a reader can see it.
    assert re.search(r"\.\d{6}\+", html) is None


def test_custom_portfolio_is_analysed(
    settings: Settings,
    tmp_path: Path,
) -> None:
    # A custom sheet can become the headline portfolio and keeps its evidence.
    configured, _ = _prepare(settings, tmp_path, {"custom_suppliers": [("Custom", "custom")]})
    result = pipeline.run_pipeline(
        configured, all_sheets=True, offline=True, output_dir=tmp_path / "out"
    )
    assert result.manifest["headline_portfolio"] == "custom_suppliers"
    assert len(result.suppliers) == 1
    assert result.suppliers[0]["resolution_status"] == "resolved"
    assert "Custom" in _html(result, tmp_path)


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.depths: list[int] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self.depths.append(len(self.stack))
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.stack:
            del self.stack[len(self.stack) - 1 - self.stack[::-1].index(tag) :]


def test_report_sections_ranked_table_and_self_containment(
    findings: Findings, tmp_path: Path
) -> None:
    # The report keeps its sections in order and its ranked rows, and loads no remote assets.
    html = _html(findings, tmp_path)
    ids = [
        "report-header",
        "executive-summary",
        "coverage",
        "convergence",
        "supplier-worklist",
        "key-finding",
        "suppliers",
        "flagged-entities",
        "methodology",
    ]
    assert [html.index(f'id="{key}"') for key in ids] == sorted(
        html.index(f'id="{key}"') for key in ids
    )
    for phrase in [
        "供应商",
        "节点",
        "Translated",
        "lower bounds",
        "original identifier for it instead of a name",
        "Add the entity to the supplier workbook",
        "No shared-entity finding in this report matched the current search and filters.",
        "Screening a new entity requires a new screening run.",
    ]:
        assert phrase.casefold() in html.casefold()
    table = _shared_findings(html)
    assert table.count("data-filter-row") == len(cast(list[dict[str, Any]], findings.shared_nodes))
    for node in cast(list[dict[str, Any]], findings.shared_nodes):
        # Fixture names contain ID-like text; the sentinel test checks omitted ID fields.
        assert node["label"] in table
    document = _Document()
    document.feed(html)
    for tag, attrs in document.tags:
        assert tag not in {"link", "iframe", "img", "object", "embed", "base"}
        assert not any(k.startswith("on") or k in {"src", "href", "xlink:href"} for k in attrs)
    assert "fetch(" not in html and "XMLHttpRequest" not in html and "url(" not in html
    assert len(re.findall(r"<script[ >]", html)) == 1
    assert "connect-src 'none'" in html
    assert str(tmp_path) not in html and "synthetic-secret" not in html
    assert "Traceback" not in html and "SELECT " not in html
    for module in (report, presentation):
        source = Path(cast(str, module.__file__)).read_text(encoding="utf-8")
        assert not re.search(
            r"(?:import|from) (?:duckdb|sayari_poc\."
            r"(?:analysis|identity|sayari|pipeline|findings))",
            source,
        )


def test_svg_bounded_supplier_subset_escaping_and_data_driven_selection(
    findings: Findings,
    tmp_path: Path,
) -> None:
    # The diagram takes a bounded set of suppliers from the evidence and escapes source text.
    top = next(
        n for n in cast(list[dict[str, Any]], findings.shared_nodes) if n["portfolio"] == "list_3"
    )
    unsafe = "测试 <script>alert(\"x\")</script> & 'quoted'"
    top["label"] = unsafe
    top["translated_label"] = 'Translation </text><image href="bad">'
    top["suppliers"][0]["label"] = unsafe
    html = _html(findings, tmp_path)
    svgs = re.findall(r"<svg\b.*?</svg>", html, flags=re.S)
    assert len(svgs) == 1
    first = ET.fromstring(svgs[0])
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert unsafe in "".join(cast(ET.Element, first.find("s:title", ns)).itertext())
    # Every connected supplier is drawn, because six is within the cap.
    assert "All 6 connected suppliers are shown." in svgs[0]
    drawn_titles = ["".join(e.itertext()) for e in first.findall(".//s:title", ns)]
    for supplier in top["suppliers"]:
        assert any(supplier["label"] + " — " in text for text in drawn_titles)
    assert "&lt;script&gt;" in svgs[0] and "&lt;image" in svgs[0]
    assert not re.search(r"<(script|image|style)\b", svgs[0])
    boxes = first.findall("s:rect[@class='supplier-box']", ns)
    target = cast(ET.Element, first.find("s:rect[@class='upstream-box']", ns))
    connectors = first.findall("s:path[@class='edge']", ns)
    assert len(boxes) == len(connectors) == 6
    label = cast(ET.Element, first.find("s:text[@class='edge-label']", ns))
    trunk_x = float(target.attrib["x"]) + float(target.attrib["width"]) / 2
    for box, connector in zip(boxes, connectors, strict=True):
        # Every continuous connector starts on its supplier and reaches the shared entity.
        coords = re.fullmatch(
            r"M ([\d.]+) ([\d.]+) V ([\d.]+) H ([\d.]+) V ([\d.]+)", connector.attrib["d"]
        )
        assert coords is not None
        start_x, start_y, bus_y, end_x, end_y = map(float, coords.groups())
        assert start_x == float(box.attrib["x"]) + float(box.attrib["width"]) / 2
        assert start_y == float(box.attrib["y"]) + float(box.attrib["height"])
        assert end_x == trunk_x
        assert end_y == float(target.attrib["y"])
        assert start_y < bus_y < float(label.attrib["y"]) < end_y
        assert float(label.attrib["x"]) > end_x
        # Wrapped rows only stay legible if no box sits across the corridor the connectors descend,
        # so the trunk never crosses a supplier on its way to the shared entity.
        assert not (
            float(box.attrib["x"]) < trunk_x < float(box.attrib["x"]) + float(box.attrib["width"])
        )
    # Suppliers occupy more than one row, and each row turns on its own bus line.
    assert len({float(box.attrib["y"]) for box in boxes}) == 2
    assert len({connector.attrib["d"].split(" V ")[1] for connector in connectors}) == 2
    assert len(first.findall("s:rect", ns)) == 7  # No opaque box covers the connectors.
    assert "upstream connection" in svgs[0]
    assert not re.search(r"supplies|sells to|direct supplier|\d+ hops?", svgs[0])
    top["label"] = "Changed synthetic top node"
    changed = _html(findings, tmp_path)
    assert "Changed synthetic top node" in re.findall(r"<svg\b.*?</svg>", changed, re.S)[0]
    assert top["label"] not in Path(report.__file__).with_name("templates").joinpath(
        "_exhibit.svg.j2"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("no_top", 0),
        ("missing_labels", 0),
        ("blank_labels", 0),
        ("missing_suppliers", 0),
        ("intact", 1),
    ],
)
def test_svg_fallback_preserves_complete_report(
    findings: Findings,
    tmp_path: Path,
    mode: str,
    expected: int,
) -> None:
    # Leaving out the diagram keeps every report section and the ranked evidence intact.
    if mode == "no_top":
        findings.shared_nodes = []
    if mode in {"missing_labels", "blank_labels"}:
        for node in cast(list[dict[str, Any]], findings.shared_nodes):
            node["label"] = None if mode == "missing_labels" else " \t"
    if mode == "missing_suppliers":
        next(
            n
            for n in cast(list[dict[str, Any]], findings.shared_nodes)
            if n["portfolio"] == "list_3"
        )["suppliers"] = []
    html = _html(findings, tmp_path)
    assert html.count("<svg ") == expected
    assert all(
        f'id="{key}"' in html for key in ["suppliers", "convergence", "coverage", "methodology"]
    )
    table = _shared_findings(html)
    assert table.count("data-filter-row") == len(cast(list[dict[str, Any]], findings.shared_nodes))


def test_identical_offline_runs_preserve_all_artifact_bytes(
    findings: Findings,
    graph: tuple[Settings, ResponseCache],
    tmp_path: Path,
) -> None:
    # Repeating a cached run reproduces every artifact byte-for-byte with zero HTTP attempts.
    paths = [tmp_path / "out" / name for name in ["findings.json", "report.html"]]
    before = [path.read_bytes() for path in paths]
    manifest_path = tmp_path / "out/run_manifest.json"
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second = pipeline.run_pipeline(
        graph[0], all_sheets=True, offline=True, output_dir=tmp_path / "out"
    )
    assert second == findings and [path.read_bytes() for path in paths] == before
    json.dumps(json.loads(paths[0].read_text(encoding="utf-8")), allow_nan=False)
    manifest_path = tmp_path / "out/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run"]["execution_mode"] == "offline"
    assert manifest["api"]["data_http_attempts"] == 0
    assert manifest["status_counts"]["coverage"] == second.coverage
    first_manifest["run"].pop("timestamp")
    manifest["run"].pop("timestamp")
    assert manifest == first_manifest


@pytest.mark.parametrize("empty_collections", [False, True])
def test_inline_filter_executes_locally_for_both_tables_and_clear(
    r4_findings: Findings,
    tmp_path: Path,
    empty_collections: bool,
) -> None:
    # A Node harness runs the report's script and checks filtering and clearing on both tables.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node optional for local JS execution; manual browser gate still required")
    for slug, factor in r4_findings.ontology.items():
        r4_findings.ontology[slug] = factor.model_copy(
            update={"description": factor.description + " Collapsed factor description probe."}
        )
    if empty_collections:
        r4_findings.shared_nodes = []
        r4_findings.suppliers = []
    html = _html(r4_findings, tmp_path)
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None
    script = match.group(1)
    entries = _supplier_entries(html)
    texts = [_visible_text(entry) for entry in entries.values()]
    document = _Document()
    document.feed(html)
    disclosure_count = sum("data-disclosure" in attrs for _, attrs in document.tags)
    factor_groups = sum(attrs.get("class") == "factor-additional" for _, attrs in document.tags)
    nested_count = disclosure_count - len(entries)
    assert disclosure_count >= factor_groups
    additional = re.search(r'<div class="factor-additional".*?</div>', _shared_findings(html), re.S)
    description = re.search(
        r'<small class="factor-description">([^<]*Collapsed factor description probe\.)</small>',
        additional.group(0) if additional else "",
    )
    if not empty_collections:
        assert description is not None and factor_groups > 0
    factor_text = _visible_text(description.group(0)) if description else ""
    harness = r"""
const assert = require('node:assert/strict');
const classes = [];
let ready;
const control = () => ({
  value: '', focused: false, hidden: true, disabled: false,
  addEventListener(name, fn) { this[name] = fn; },
  focus() { this.focused = true; },
  scrollIntoView() { this.scrolled = true; }
});
const input = control(), clear = control(), supplierInput = control(), supplierClear = control();
const entityCountry = control(), entityCategory = control();
const supplierCountry = control(), supplierEvidence = control();
const count = {}, empty = {}, supplierCount = {}, supplierEmpty = {};
const disclosure = textContent => ({
  textContent, hidden: false, open: true, summary: control(), dataset: {},
  querySelector(selector) { assert.equal(selector, 'summary'); return this.summary; },
  removeAttribute(name) { assert.equal(name, 'open'); this.open = false; }
});
const rows = EMPTY_COLLECTIONS ? [] : Array.from({length: 24}, (_, i) =>
  disclosure(i === 0 ? 'convergence 测试' : `entity ${i} ${i === 23 ? 'offpage needle' : ''}` +
    (i === 5 ? ' Eberspächer' : '')));
// Synthetic dropdown values: PHL on every third entity, sanctions on the odd ones.
rows.forEach((row, i) => {
  row.dataset = {
    countries: i % 3 ? 'CHN' : 'PHL MYS', categories: i % 2 ? 'sanctions' : 'export'
  };
});
const exhibits = [{textContent:'diagram 测试', hidden: false}];
const suppliers = SUPPLIER_TEXT.map(disclosure);
// DEU on every fifth supplier, not_attempted on every fourth.
suppliers.forEach((supplier, i) => {
  supplier.dataset = {
    countries: i % 5 ? 'JPN' : 'DEU', evidence: i % 4 ? 'assessed' : 'not_attempted'
  };
});
const nested = Array.from({length: NESTED_COUNT}, () => disclosure('route'));
const disclosures = [...rows, ...suppliers, ...nested];
if (rows.length) {
  nested[nested.length - 1].textContent = FACTOR_TEXT;
  rows[23].textContent += ` ${nested[nested.length - 1].textContent}`;
}
const groups = [rows.slice(0, 11), rows.slice(11)].map(children => ({
  hidden: false,
  querySelectorAll(selector) { assert.equal(selector, '[data-filter-row]'); return children; }
}));
const original = [...suppliers], originalRows = [...rows];
const controls = {};
for (const prefix of ['entity', 'supplier']) {
  for (const part of ['pagination', 'previous', 'next', 'page-status', 'page-number']) {
    controls[`${prefix}-${part}`] = control();
  }
}
const bad = () => { throw Error('forbidden mutation or network'); };
for (const element of [...rows, ...exhibits, ...disclosures, ...Object.values(controls)]) {
  Object.defineProperty(element, 'innerHTML', {set: bad});
  element.remove = bad; element.replaceChild = bad; element.insertAdjacentHTML = bad;
}
Object.freeze(suppliers); Object.freeze(rows);
global.fetch = bad;
global.XMLHttpRequest = bad;
const events = {};
global.window = {addEventListener: (name, fn) => { events[name] = fn; }};
global.document = {
  documentElement: {classList: {add: value => classes.push(value),
    toggle(value, force) { const index = classes.indexOf(value);
      if (force && index < 0) classes.push(value);
      if (!force && index >= 0) classes.splice(index, 1); } }},
  addEventListener: (name, fn) => { assert.equal(name, 'DOMContentLoaded'); ready = fn; },
  getElementById: id => ({
    'report-filter': input, 'clear-filter': clear, 'filter-count': count, 'no-match': empty,
    'supplier-filter': supplierInput, 'clear-supplier-filter': supplierClear,
    'supplier-filter-count': supplierCount, 'supplier-no-match': supplierEmpty,
    'entity-country': entityCountry, 'entity-category': entityCategory,
    'supplier-country': supplierCountry, 'supplier-evidence': supplierEvidence,
    ...controls
  })[id],
  querySelectorAll: selector => ({
    '[data-filter-row]': rows, '[data-filter-exhibit]': exhibits,
    '[data-supplier-row]': suppliers, '[data-disclosure]': disclosures,
    '[data-entity-group]': groups
  })[selector],
};
"""
    harness = harness.replace("SUPPLIER_TEXT", json.dumps(texts, ensure_ascii=False))
    harness = harness.replace("NESTED_COUNT", str(nested_count))
    harness = harness.replace("FACTOR_TEXT", json.dumps(factor_text, ensure_ascii=False))
    harness = harness.replace("EMPTY_COLLECTIONS", json.dumps(empty_collections))
    assertions = r"""
assert.deepEqual(classes, ['js']);
assert.ok(disclosures.every(d => d.open));
ready();
assert.deepEqual(classes, ['js', 'js-ready']);
assert.ok(disclosures.every(d => !d.open));
assert.equal(controls['entity-pagination'].hidden, false);
assert.equal(controls['supplier-pagination'].hidden, false);
const visible = elements => elements.filter(e => !e.hidden);
const status = prefix => controls[`${prefix}-page-status`].textContent;
if (EMPTY_COLLECTIONS) {
  assert.equal(status('entity'), 'Showing 0 of 0 shared entity rows');
  assert.equal(status('supplier'), 'Showing 0 of 0 supplier entries');
  for (const prefix of ['entity', 'supplier']) {
    assert.equal(controls[`${prefix}-page-number`].textContent, 'Page 0 of 0');
    assert.equal(controls[`${prefix}-previous`].disabled, true);
    assert.equal(controls[`${prefix}-next`].disabled, true);
    controls[`${prefix}-next`].click();
  }
} else {
  assert.equal(suppliers.length, 50);
  assert.equal(supplierCount.textContent, '50 of 50 supplier entries match');
  assert.equal(count.textContent, '24 of 24 shared entity rows match');
  assert.deepEqual(visible(rows), rows.slice(0, 10));
  assert.deepEqual(visible(suppliers), suppliers.slice(0, 10));
  assert.equal(status('entity'), 'Showing 1–10 of 24 shared entity rows');
  assert.equal(controls['entity-page-number'].textContent, 'Page 1 of 3');
  assert.equal(controls['entity-previous'].disabled, true);
  input.value = 'collapsed factor description probe'; input.input();
  assert.deepEqual(visible(rows), [rows[23]]);
  assert.ok(disclosures.every(d => !d.open));
  supplierInput.value = 'collapsed factor description probe'; supplierInput.input();
  const factorMatches = suppliers.filter(s =>
    s.textContent.includes('Collapsed factor description probe'));
  assert.ok(factorMatches.length > 0 && factorMatches.length < suppliers.length);
  assert.deepEqual(visible(suppliers), factorMatches.slice(0, 10));
  assert.ok(disclosures.every(d => !d.open));
  clear.click(); supplierClear.click();
  suppliers[0].open = true; rows[0].open = true; nested[0].open = true;
  const openStates = disclosures.map(d => d.open);
  controls['entity-next'].click();
  assert.deepEqual(visible(rows), rows.slice(10, 20));
  assert.equal(status('entity'), 'Showing 11–20 of 24 shared entity rows');
  assert.ok(rows[10].summary.focused && rows[10].summary.scrolled);
  controls['entity-next'].click();
  assert.deepEqual(visible(rows), rows.slice(20));
  assert.equal(status('entity'), 'Showing 21–24 of 24 shared entity rows');
  assert.equal(controls['entity-next'].disabled, true);
  assert.equal(groups[0].hidden, true); assert.equal(groups[1].hidden, false);
  controls['entity-next'].click();
  assert.deepEqual(visible(rows), rows.slice(20));
  controls['entity-previous'].click();
  assert.deepEqual(visible(rows), rows.slice(10, 20));
  assert.deepEqual(disclosures.map(d => d.open), openStates);
  input.value = 'needle'; input.input();
  assert.deepEqual(visible(rows), [rows[23]]);
  assert.equal(status('entity'), 'Showing 1–1 of 1 shared entity rows');
  assert.equal(controls['entity-page-number'].textContent, 'Page 1 of 1');
  assert.equal(controls['entity-previous'].disabled, true);
  assert.equal(controls['entity-next'].disabled, true);
  input.value = '测试'; input.input();
  assert.deepEqual(visible(rows), [rows[0]]);
  assert.equal(count.textContent, '1 of 24 shared entity rows match');
  assert.deepEqual(visible(suppliers), suppliers.slice(0, 10));
  input.value='diagram'; input.input(); assert.equal(empty.hidden, false);
  assert.ok(rows.every(r => r.hidden)); assert.equal(exhibits[0].hidden, false);
  assert.ok(groups.every(g => g.hidden));
  assert.equal(status('entity'), 'Showing 0 of 0 shared entity rows');
  const rowStates = rows.map(r => r.hidden);
  const seen = [...visible(suppliers)];
  for (let page = 1; page < 5; page++) {
    controls['supplier-next'].click();
    assert.deepEqual(visible(suppliers), suppliers.slice(page * 10, page * 10 + 10));
    seen.push(...visible(suppliers));
  }
  assert.deepEqual(seen, suppliers);
  assert.equal(status('supplier'), 'Showing 41–50 of 50 supplier entries');
  assert.equal(controls['supplier-next'].disabled, true);
  assert.deepEqual(rows.map(r => r.hidden), rowStates);
  supplierInput.value = '中间节点'; supplierInput.input();
  const matching = suppliers.filter(s => s.textContent.includes('中间节点'));
  assert.deepEqual(visible(suppliers), matching.slice(0, 10));
  assert.ok(matching.length > 0 && matching.length < suppliers.length);
  assert.equal(controls['supplier-previous'].disabled, true);
  assert.deepEqual(rows.map(r => r.hidden), rowStates);
  assert.deepEqual(disclosures.map(d => d.open), openStates);
  supplierInput.value = 'ＩＮＰＵＴ'; supplierInput.input();
  assert.equal(visible(suppliers).length, 10);
  supplierInput.value = '<script>globalThis.executed = true</script>'; supplierInput.input();
  assert.ok(suppliers.some(s => !s.hidden));
  assert.equal(globalThis.executed, undefined);
  supplierInput.value = 'absent unmatched query'; supplierInput.input();
  assert.equal(supplierEmpty.hidden, false);
  assert.equal(supplierCount.textContent, '0 of 50 supplier entries match');
  assert.ok(suppliers.every(s => s.hidden));
  supplierClear.click();
  assert.equal(supplierInput.value, ''); assert.equal(supplierInput.focused, true);
  assert.equal(supplierEmpty.hidden, true);
  assert.equal(supplierCount.textContent, '50 of 50 supplier entries match');
  assert.deepEqual(visible(suppliers), suppliers.slice(0, 10));
  assert.deepEqual(disclosures.map(d => d.open), openStates);
  supplierInput.value = '   '; supplierInput.input(); assert.equal(supplierEmpty.hidden, true);
  input.value='absent'; input.input(); assert.equal(empty.hidden, false);
  clear.click(); assert.equal(input.value, ''); assert.equal(empty.hidden, true);
  assert.deepEqual(visible(rows), rows.slice(0, 10)); assert.equal(exhibits[0].hidden, false);
  assert.equal(input.focused, true);
  // A dropdown matches one data attribute exactly, combines with the keyword, and leaves the
  // other list alone; Clear resets it with the keyword.
  entityCountry.value = 'PHL'; entityCountry.change();
  const inPhl = rows.filter(r => r.dataset.countries.split(' ').includes('PHL'));
  assert.equal(inPhl.length, 8);
  assert.deepEqual(visible(rows), inPhl);
  assert.equal(count.textContent, '8 of 24 shared entity rows match');
  assert.deepEqual(visible(suppliers), suppliers.slice(0, 10));
  entityCategory.value = 'sanctions'; entityCategory.change();
  assert.deepEqual(visible(rows), [rows[3], rows[9], rows[15], rows[21]]);
  assert.equal(empty.hidden, true);
  input.value = 'needle'; input.input();
  assert.ok(rows.every(r => r.hidden)); assert.equal(empty.hidden, false);
  clear.click();
  assert.equal(entityCountry.value, ''); assert.equal(entityCategory.value, '');
  assert.equal(input.value, ''); assert.equal(empty.hidden, true);
  assert.equal(count.textContent, '24 of 24 shared entity rows match');
  // A code must equal a whole value, so "CH" does not pick up "CHN".
  entityCountry.value = 'CH'; entityCountry.change();
  assert.ok(rows.every(r => r.hidden)); assert.equal(empty.hidden, false);
  clear.click();
  supplierCountry.value = 'DEU'; supplierCountry.change();
  const german = suppliers.filter(s => s.dataset.countries === 'DEU');
  assert.equal(german.length, 10);
  assert.deepEqual(visible(suppliers), german);
  assert.equal(supplierCount.textContent, '10 of 50 supplier entries match');
  supplierEvidence.value = 'not_attempted'; supplierEvidence.change();
  assert.deepEqual(visible(suppliers), [suppliers[0], suppliers[20], suppliers[40]]);
  assert.equal(supplierEmpty.hidden, true);
  supplierClear.click();
  assert.equal(supplierCountry.value, ''); assert.equal(supplierEvidence.value, '');
  assert.equal(supplierCount.textContent, '50 of 50 supplier entries match');
  assert.equal(supplierInput.focused, true);
  // Accents are ignored both ways, so a query typed without them still finds the name.
  input.value = 'eberspacher'; input.input(); assert.deepEqual(visible(rows), [rows[5]]);
  input.value = 'EBERSPÄCHER'; input.input(); assert.deepEqual(visible(rows), [rows[5]]);
  clear.click();
  controls['supplier-next'].click();
}
const elements = [...rows, ...suppliers, ...groups];
assert.ok(exhibits.every(e => !e.hidden));
const hiddenBeforePrint = elements.map(e => e.hidden);
const openBeforePrint = disclosures.map(e => e.open);
const supplierPageBeforePrint = status('supplier');
events.beforeprint(); events.beforeprint();
assert.ok(elements.every(e => !e.hidden));
assert.ok(disclosures.every(e => e.open));
events.afterprint(); events.afterprint();
assert.deepEqual(elements.map(e => e.hidden), hiddenBeforePrint);
assert.deepEqual(disclosures.map(e => e.open), openBeforePrint);
assert.equal(status('supplier'), supplierPageBeforePrint);
assert.deepEqual(rows, originalRows); assert.deepEqual(suppliers, original);
"""
    assertions = assertions.replace("EMPTY_COLLECTIONS", json.dumps(empty_collections))
    completed = subprocess.run(
        [node, "-"],
        input=harness + script + assertions,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_filter_dropdowns_match_what_each_card_displays(findings: Findings, tmp_path: Path) -> None:
    # Every card carries the values its dropdowns filter on, and every dropdown choice reaches a
    # card, so no choice can empty the list on its own.
    html = _html(findings, tmp_path)
    document = _Document()
    document.feed(html)
    view = presentation.build_report_view(findings)
    suppliers = [attrs for _, attrs in document.tags if "data-supplier-row" in attrs]
    assert [attrs["id"] for attrs in suppliers] == [row.dom_id for row in view.supplier_rows]
    for attrs, row in zip(suppliers, view.supplier_rows, strict=True):
        assert attrs["data-countries"] == (row.supplier.get("input_country") or "")
        assert attrs["data-evidence"] == (row.supplier.get("coverage_status") or "not_attempted")
    entities = [attrs for _, attrs in document.tags if "data-filter-row" in attrs]
    nodes = [node for group in view.ranked_groups for node in group["nodes"]]
    for attrs, node in zip(entities, nodes, strict=True):
        assert attrs["data-countries"] == " ".join(node["countries"] or [])
        assert attrs["data-categories"] == " ".join(
            presentation.risk_categories(node["severe_factors"], findings.ontology)
        )
    cards = {"supplier": suppliers, "entity": entities}
    keys = {"country": "data-countries", "evidence": "data-evidence", "category": "data-categories"}
    for select_id in ("supplier-country", "supplier-evidence", "entity-country", "entity-category"):
        scope, facet = select_id.split("-")
        assert any(tag == "label" and attrs.get("for") == select_id for tag, attrs in document.tags)
        block = re.search(rf'<select id="{select_id}"[^>]*>(.*?)</select>', html, re.S)
        assert block is not None
        options = re.findall(r'<option value="([^"]*)">([^<]*)</option>', block.group(1))
        assert options[0][0] == "" and options[0][1].startswith("All ") and len(options) > 1
        for value, _ in options[1:]:
            assert any(value in (attrs.get(keys[facet]) or "").split() for attrs in cards[scope])


def test_country_options_name_known_codes_and_keep_unknown_ones() -> None:
    # Dropdown labels name each country and sort by that name; a code pycountry doesn't know
    # stays as it is rather than being guessed.
    assert report.country_options(["KOR", "DEU", "C0", "TWN"]) == [
        ("C0", "C0"),
        ("DEU", "Germany (DEU)"),
        ("KOR", "South Korea (KOR)"),
        ("TWN", "Taiwan (TWN)"),
    ]


def test_ranked_table_identifies_scope_and_leads_with_headline(
    findings: Findings,
    tmp_path: Path,
) -> None:
    # Ranked groups name the supplier list they belong to, and the headline list comes first.
    before = findings.model_dump(mode="json")
    html = _html(findings, tmp_path)
    table = _shared_findings(html)
    expected = [
        " ".join(filter(None, [node["label"], node["translated_label"]]))
        for portfolio in ["list_3", "another_portfolio"]
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] == portfolio
    ]
    name_cells = re.findall(r'<span class="entity-identity">(.*?)</span>', table, re.S)
    assert [_visible_text(cell) for cell in name_cells] == expected
    assert "Screened supplier list" in table and "Supplier list: another_portfolio" in table
    assert findings.model_dump(mode="json") == before  # Presentation cannot rewrite ranking.
    assert "Shared entities within the country limit" in html
    assert "Excluded — no critical or high risk factor" in html
    assert "after_hub_suppression</td>" not in html


def test_json_and_html_use_lf(
    findings: Findings,
    tmp_path: Path,
) -> None:
    # Both artifacts are written with LF line endings on every platform, so their bytes match.
    for name in ["findings.json", "report.html"]:
        content = (tmp_path / "out" / name).read_bytes()
        assert b"\n" in content and b"\r\n" not in content


def test_list_3_only_preserves_pre_r1_convergence_and_coverage(
    graph: tuple[Settings, ResponseCache], tmp_path: Path
) -> None:
    # Frozen from this synthetic graph before the watchlist/control surface was removed. Running
    # list_3 on its own keeps the expected convergence and coverage.
    result = pipeline.run_pipeline(
        graph[0], sheets=["list_3"], offline=True, output_dir=tmp_path / "list-3-only"
    )
    assert len(cast(list[dict[str, Any]], result.suppliers)) == 8
    assert {row["portfolio"] for row in cast(list[dict[str, Any]], result.suppliers)} == {"list_3"}
    assert [
        (node["upstream_id"], node["supplier_count"])
        for node in cast(list[dict[str, Any]], result.shared_nodes)
    ] == [
        ("n-top", 6),
        ("n-boundary", 2),
    ]
    assert cast(dict[str, Any], result.convergence)["list_3"]["funnel"] == {
        "contributing_suppliers": 6,
        "distinct_upstream_entities": 6,
        "shared_nodes": 5,
        "suppressed_hubs": 2,
        "after_hub_suppression": 3,
        "removed_by_severity_filter": 1,
        "after_severity_filter": 2,
    }
    assert result.coverage == {
        "list_3": {"assessed": 6, "partial": 1, "no_data": 0, "error": 0, "not_attempted": 1}
    }


R3_SECTIONS = [
    "report-header",
    "executive-summary",
    "coverage",
    "convergence",
    "supplier-worklist",
    "key-finding",
    "suppliers",
    "flagged-entities",
    "methodology",
]


def _section(html: str, identifier: str) -> str:
    return html.split(f'id="{identifier}"', 1)[1].split("</section>", 1)[0]


def _visible_text(html: str) -> str:
    return " ".join(html_module.unescape(re.sub(r"<[^>]+>", " ", html)).split())


def _executive_cards(html: str) -> dict[str, str]:
    summary = _section(html, "executive-summary")
    return {
        _visible_text(term): _visible_text(value)
        for term, value in re.findall(r"<dt>(.*?)</dt><dd>(.*?)</dd>", summary, re.S)
    }


def test_r3_sections_are_ordered_named_regions(findings: Findings, tmp_path: Path) -> None:
    # Sections are ordered, labelled landmarks under one h1, and headings never skip a level.
    html = _html(findings, tmp_path)
    positions = [html.index(f'id="{identifier}"') for identifier in R3_SECTIONS]
    assert positions == sorted(positions) and len(set(positions)) == len(R3_SECTIONS)
    document = _Document()
    document.feed(html)
    elements = {attrs["id"]: (tag, attrs) for tag, attrs in document.tags if "id" in attrs}
    for identifier in R3_SECTIONS:
        tag, attrs = elements[identifier]
        assert tag == "section" or attrs.get("role") == "region"
        assert elements[attrs["aria-labelledby"]][0] == "h2"
    levels = [int(tag[1]) for tag, _ in document.tags if re.fullmatch(r"h[1-6]", tag)]
    assert levels[0] == 1 and levels.count(1) == 1
    assert all(right <= left + 1 for left, right in zip(levels, levels[1:], strict=False))


def test_executive_cards_derive_from_changed_findings(findings: Findings, tmp_path: Path) -> None:
    # The executive cards are recomputed from the evidence supplied, not pinned copy.
    headline = cast(dict[str, Any], findings.manifest)["headline_portfolio"]
    original = [
        row
        for row in cast(list[dict[str, Any]], findings.suppliers)
        if row["portfolio"] == headline
    ]
    rows = [dict(row) for row in original[:4] + original[:2]]
    for index, row in enumerate(rows):
        row["resolution_status"] = "resolved" if index < 4 else "unmatched"
        row["entity_id"] = f"canonical-{index // 2 if index < 2 else index}" if index < 4 else None
    findings.suppliers = rows + [
        r for r in cast(list[dict[str, Any]], findings.suppliers) if r["portfolio"] != headline
    ]
    findings.shared_nodes = cast(list[dict[str, Any]], findings.shared_nodes)[1:]
    counts = {"assessed": 3, "partial": 2, "no_data": 1, "error": 1, "not_attempted": 2}
    findings.coverage[headline] = counts
    resolved = [r for r in rows if r["resolution_status"] == "resolved"]
    entities = {r["entity_id"] for r in resolved if r["entity_id"]}
    nodes = [
        n for n in cast(list[dict[str, Any]], findings.shared_nodes) if n["portfolio"] == headline
    ]
    cards = _executive_cards(_html(findings, tmp_path))
    assert cards == {
        "Suppliers screened": f"{len(rows)} supplier input rows",
        "Suppliers resolved": (
            f"{len(resolved)} resolved input rows {len(entities)} distinct entities · "
            f"{len(rows) - len(resolved)} unresolved rows"
        ),
        "Shared entities for review": (
            f"{len(nodes)} distinct sub-tier entities Includes possible identity matches"
        ),
        "With upstream evidence": (
            f"{counts['assessed'] + counts['partial']} / {sum(counts.values())} "
            f"supplier input rows {counts['assessed']} assessed · {counts['partial']} partial"
        ),
    }

    summary = _section(_html(findings, tmp_path), "executive-summary")
    assert (
        "Shared entities for review appear upstream of more than one supplier "
        "and meet the stated risk factor rules. Possible identity matches are unconfirmed."
    ) in summary
    assert "<br>" not in summary


def test_coverage_card_uses_all_five_states(findings: Findings, tmp_path: Path) -> None:
    # The coverage ratio counts every outcome in its input-row denominator.
    counts = findings.coverage[cast(dict[str, Any], findings.manifest)["headline_portfolio"]]
    before = _executive_cards(_html(findings, tmp_path))["With upstream evidence"]
    counts["partial"] += 3
    after = _executive_cards(_html(findings, tmp_path))["With upstream evidence"]
    assert before != after
    assert after.startswith(f"{counts['assessed'] + counts['partial']} / {sum(counts.values())} ")
    counts["no_data"] += 2
    changed_denominator = _executive_cards(_html(findings, tmp_path))["With upstream evidence"]
    assert changed_denominator.startswith(
        f"{counts['assessed'] + counts['partial']} / {sum(counts.values())} "
    )
    assert changed_denominator != after


def test_key_finding_follows_changed_rank_order(findings: Findings, tmp_path: Path) -> None:
    # Changing producer rank changes the displayed key finding.
    nodes = [
        n
        for n in cast(list[dict[str, Any]], findings.shared_nodes)
        if n["portfolio"] == cast(dict[str, Any], findings.manifest)["headline_portfolio"]
    ]
    original_label = nodes[0]["label"]
    cast(list[dict[str, Any]], findings.shared_nodes).reverse()
    html = _section(_html(findings, tmp_path), "key-finding")
    assert nodes[-1]["label"] in html and original_label not in html
    assert f"{nodes[-1]['supplier_count']} distinct connected suppliers" in html


# R-5: on a narrow screen the diagram is replaced by a text list of the same suppliers, so the
# shared entity never needs sideways scrolling; print and wide screens keep the diagram.
def test_key_finding_lists_the_diagram_suppliers_for_narrow_screens(
    findings: Findings, tmp_path: Path
) -> None:
    # The narrow-screen text list names exactly the suppliers the diagram draws.
    html = _html(findings, tmp_path)
    section = _section(html, "key-finding")
    svg = re.findall(r"<svg\b.*?</svg>", section, flags=re.S)[0]
    supplier_title = r'<text x="[^"]+" y="[^"]+" text-anchor="middle"><title>(.*?)</title>'
    drawn = [
        html_module.unescape(title.split(" — ")[0]) for title in re.findall(supplier_title, svg)
    ]
    listed = re.search(r'<ul class="exhibit-list">(.*?)</ul>', section, flags=re.S)
    assert drawn and listed is not None
    items = [
        html_module.unescape(re.sub(r"<small>.*?</small>", "", item)).strip()
        for item in re.findall(r"<li>(.*?)</li>", listed.group(1), flags=re.S)
    ]
    assert items == drawn
    assert ".exhibit-list { display: none; }" in html
    narrow = re.search(r"@media screen and \(max-width: 700px\) \{(.*?)\n\}", html, flags=re.S)
    assert narrow is not None
    assert ".svg-wrap { display: none; }" in narrow.group(1)
    assert ".exhibit-list { display: block; }" in narrow.group(1)
    # Worklist names may wrap only between words, never mid-word.
    assert "#supplier-worklist th, #supplier-worklist td { overflow-wrap: break-word; }" in html


def test_key_finding_empty_keeps_every_other_section(findings: Findings, tmp_path: Path) -> None:
    # An empty qualifying queue leaves the rest of the report available.
    findings.shared_nodes = []
    html = _html(findings, tmp_path)
    text = _visible_text(_section(html, "key-finding"))
    assert "No shared upstream entity meets" in text
    assert "the country and risk factor rules within retrieved evidence" in text
    assert "How shared upstream entities were selected" in text
    assert "Coverage and data completeness" in text
    assert not re.search(r"\d", text)
    assert all(f'id="{identifier}"' in html for identifier in R3_SECTIONS)


def test_key_finding_escapes_source_and_preserves_native_script(
    findings: Findings, tmp_path: Path
) -> None:
    # Key-finding labels render as literal text and keep both native and translated script.
    node = next(
        n
        for n in cast(list[dict[str, Any]], findings.shared_nodes)
        if n["portfolio"] == cast(dict[str, Any], findings.manifest)["headline_portfolio"]
    )
    node["label"] = "<script>alert(1)</script>\"&'"
    node["translated_label"] = "原生文字 — 供应链实体"
    html = _section(_html(findings, tmp_path), "key-finding")
    assert node["label"] not in html and "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;&#34;&amp;&#39;" in html
    assert node["translated_label"] in html
    assert node["label"] in html_module.unescape(html)


def test_single_stylesheet_and_script_with_variable_free_css(
    findings: Findings, tmp_path: Path
) -> None:
    # The report stays self-contained, with one stylesheet and one script.
    html = _html(findings, tmp_path)
    document = _Document()
    document.feed(html)
    assert sum(tag == "style" for tag, _ in document.tags) == 1
    assert sum(tag == "script" for tag, _ in document.tags) == 1
    assert all(not ({"href", "src", "xlink:href"} & attrs.keys()) for _, attrs in document.tags)
    css = (
        Path(report.__file__)
        .with_name("templates")
        .joinpath("_report.css.j2")
        .read_text(encoding="utf-8")
    )
    assert "{{" not in css and "{%" not in css


def test_r3_retains_bounded_evidence_wording(findings: Findings, tmp_path: Path) -> None:
    # Required limitations remain visible beside retrieved evidence.
    html = _html(findings, tmp_path)
    for phrase in [
        "lower bounds",
        "within retrieved evidence",
        "within the search limits",
        "not proof of direct supply",
        "Fully assessed",
        "missing information is not interpreted as an absence of risk",
    ]:
        assert phrase in html


def test_r3_has_no_removed_scope_surface(findings: Findings, tmp_path: Path) -> None:
    # Removed product features cannot return through report copy.
    html = _html(findings, tmp_path)
    for phrase in [
        "list_1",
        "list_2",
        "Restricted-party result",
        "Control comparison",
        "designation context",
    ]:
        assert phrase not in html


def test_r3_repeated_render_is_byte_identical(findings: Findings, tmp_path: Path) -> None:
    # Rendering the same Findings twice produces identical HTML bytes.
    first, second = tmp_path / "first.html", tmp_path / "second.html"
    report.render_report(findings, first)
    report.render_report(findings, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.fixture
def r4_findings(findings: Findings) -> Findings:
    source = next(
        row for row in cast(list[dict[str, Any]], findings.suppliers) if row["entity_id"] == "s1"
    )
    source["input_name"] = "<script>globalThis.executed = true</script> Input 原始"
    source["score"] = 272.61
    second = next(
        row for row in cast(list[dict[str, Any]], findings.suppliers) if row["entity_id"] == "s2"
    )
    second["score"] = 88.54
    unresolved = next(
        row for row in cast(list[dict[str, Any]], findings.suppliers) if row["entity_id"] is None
    )
    cast(list[dict[str, Any]], findings.suppliers).extend(
        {**unresolved, "row_number": number, "input_name": f"Input {number}"}
        for number in range(20, 60)
    )
    cast(list[dict[str, Any]], findings.suppliers).sort(
        key=lambda row: (row["portfolio"], row["row_number"])
    )
    components = [
        {
            "hs_code": "0099",
            "departure_countries": ["ZZZ", "AAA"],
            "arrival_countries": [],
            "min_date": "2020-01-03",
            "max_date": "2025-07-04",
        },
        {
            "hs_code": "8542",
            "departure_countries": [],
            "arrival_countries": ["USA", "CHN"],
            "min_date": None,
            "max_date": None,
        },
    ]
    hops: list[dict[str, Any]] = [
        {"entity_id": "middle", "tier": 2, "components": components},
        {"entity_id": "n-top", "tier": 4, "components": components[::-1]},
    ]
    first = {"source_entity_id": "s1", "path_index": 1, "hops": hops}
    different = {
        "source_entity_id": "s1",
        "path_index": 3,
        "hops": [{**hop, "components": hop["components"][::-1]} for hop in hops],
    }
    findings.paths = {
        "entities": {
            "middle": {"label": "中间节点", "translated_label": "Intermediate label"},
            "n-top": {"label": "路径终点", "translated_label": "Target path label"},
        },
        "nodes": {
            "n-top": {
                "retrieved_path_count": 9,
                "stored_path_count": 4,
                "truncated": True,
                "path_observed_supplier_ids": ["s1", "s2"],
                "paths": [
                    first,
                    {**first, "path_index": 2},
                    different,
                    {**first, "source_entity_id": "s2"},
                ],
            },
            "n-boundary": {
                "retrieved_path_count": 0,
                "stored_path_count": 0,
                "truncated": False,
                "path_observed_supplier_ids": [],
                "paths": [],
            },
        },
    }
    return findings


def _supplier_entries(html: str) -> dict[str, str]:
    section = _section(html, "suppliers")
    starts = list(re.finditer(r'<details id="(supplier-\d+)" data-supplier-row', section))
    return {
        match[1]: section[
            match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(section)
        ]
        for index, match in enumerate(starts)
    }


def test_supplier_entries_server_render_all_evidence_in_four_blocks(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # All 50 supplier entries render their evidence in the page, with no client-side fetching.
    before = r4_findings.model_dump()
    html = _html(r4_findings, tmp_path)
    entries = _supplier_entries(html)
    view = presentation.build_report_view(r4_findings)
    rows = view.supplier_rows
    assert len(entries) == len(rows) == 50
    assert list(entries) == [row.dom_id for row in rows]
    for row in rows:
        entry = entries[row.dom_id]
        summary = _visible_text(entry.split("</summary>", 1)[0])
        assert str(row.supplier["input_name"]) in summary
        assert str(row.supplier["resolution_status"]).replace("_", " ") in summary
        # The summary shows the same label as the Coverage section, not the stored status.
        assert (
            view.coverage_labels[cast(str, row.supplier["coverage_status"]) or "not_attempted"]
            in summary
        )
        if row.exposure_count:
            assert (
                f"Connected upstream to {row.exposure_count} of {row.exposure_denominator}"
                in summary
            )
            assert f"{row.path_stored_count} supply chain path records shown" in summary
        elif row.exposure_count is None:
            assert row.exposure_note in summary
            assert "Connected upstream to 0" not in summary
        else:
            assert "No observed upstream connection" in summary
        headings = re.findall(r"<h3>(.*?)</h3>", entry)
        assert headings == [
            "Identity and resolution",
            "Retrieval coverage",
            "Shared upstream connections",
            "Risk factors reported for this supplier",
        ]
        if row.supplier["label"]:
            assert str(row.supplier["label"]) in _visible_text(entry)
        for node in row.exposure_nodes:
            assert str(node.node["label"]) in _visible_text(entry)
            for route in node.routes:
                for hop in route.hops:
                    assert hop.label is not None and hop.label in _visible_text(entry)
                    assert f"Tier {hop.tier}" in _visible_text(entry)
                    for component in hop.components:
                        assert str(component["hs_code"]) in entry
                        for key in ("min_date", "max_date"):
                            if component[key]:
                                assert component[key] in entry
    assert r4_findings.model_dump() == before


def test_supplier_path_counts_coverage_and_components_are_honest(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Path notes keep retrieved totals, stored examples and gaps clearly apart.
    html = _html(r4_findings, tmp_path)
    entries = _supplier_entries(html)
    rows = presentation.build_report_view(r4_findings).supplier_rows
    first = next(row for row in rows if row.supplier["entity_id"] == "s1")
    text = _visible_text(entries[first.dom_id])
    # The ratio is per node, so the sentence must use that node's own supplier count, read from the
    # exposure entry the template renders, not from a same-named node in another portfolio.
    top = next(node for node in first.exposure_nodes if node.node["upstream_id"] == "n-top")
    assert (
        f"Sayari returned 9 supply chain paths to this entity across all "
        f"{top.node['supplier_count']} connected suppliers; this report keeps 4 of them as "
        "examples. 2 of the 6 suppliers have at least one path" in text
    )
    # A node with no retrieved route states that, and still reports supplier coverage.
    assert (
        "Sayari returned no supply chain paths to this entity within the search limits. "
        "None of the 2 suppliers has a path on record." in text
    )
    # A stretch of the sentence with no run-specific value, so it reads the same in every fixture.
    method = (
        "Examples are taken from each connected supplier in rotation, never ranked, "
        "so they cover as many suppliers as possible."
    )
    assert method not in text
    assert _visible_text(_section(html, "methodology")).count(method) == 1
    assert _visible_text(html).count(method) == 1
    # Grouped routes share a trimmed prefix; they are no longer necessarily identical.
    assert "Sayari returned 2 paths that reach this entity by this route." in text
    assert text.count("Observed supply chain path") == 2
    # Supplier coverage is stated positively on every block, not only when one is missing.
    assert "2 of the 6 suppliers have at least one path" in text
    assert "None of the 2 suppliers has a path on record" in text
    assert (
        "No supply chain path to this entity was retrieved for this supplier within the "
        "search limits. That does not mean no path exists."
    ) in text
    assert "Departure countries: ZZZ, AAA → Arrival countries: none returned" in text
    assert "Departure countries: none returned → Arrival countries: USA, CHN" in text
    assert "Shipment observation window: 2020-01-03 – 2025-07-04" in text
    assert "Shipment observation window: none returned – none returned" in text
    for row in rows:
        if row.portfolio != "list_3":
            assert "Paths shown for this entity" not in entries[row.dom_id]


def test_relevance_score_label_value_and_methodology_have_separate_scopes(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Relevance stays an unscaled value and never reads as a confidence percentage.
    html = _html(r4_findings, tmp_path)
    entries = _supplier_entries(html)
    for row in presentation.build_report_view(r4_findings).supplier_rows:
        identity = (
            entries[row.dom_id]
            .split("<h3>Identity and resolution</h3>", 1)[1]
            .split("<h3>Retrieval coverage</h3>", 1)[0]
        )
        scores = re.findall(r"<small data-relevance-score>(.*?)</small>", identity, re.S)
        score = row.supplier["score"]
        assert len(scores) == (0 if score is None else 1)
        if score is not None:
            assert isinstance(score, (int, float))
            assert _visible_text(scores[0]) == f"Relevance score: {score}"
            assert not re.search(r"%|confidence|probability|certainty", scores[0], re.I)
            assert f"{float(score) / 100}" not in _visible_text(identity)
        strength = row.supplier["match_strength"]
        if strength is not None:
            assert f"Match strength: {strength}" in _visible_text(identity)
        else:
            assert "Match strength:" not in identity
        assert "confidence:" not in _visible_text(identity).lower()
    # The identity-uncertainty block was removed from the methodology. The honesty rules bind the
    # qualification, not the section carrying it, so the remaining scope rule for this section is
    # simply that it never reintroduces score language.
    methodology = _visible_text(_section(html, "methodology"))
    assert not re.search(r"%|confidence|probability|certainty", methodology, re.I)


def test_supplier_path_wording_and_no_r6_recommendations(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Supplier paths use bounded trade-evidence wording and offer no invented advice.
    html = _html(r4_findings, tmp_path)
    for phrase in (
        "supplies",
        "supplier of",
        "sells to",
        "sold to",
        "purchases from",
        "buys from",
        "direct supplier",
        "contractual",
        "contract with",
        "sources from",
        "ships to",
    ):
        assert phrase not in html.lower()
    text = _visible_text(_section(html, "suppliers")).lower()
    for phrase in (
        "audit this",
        "block this",
        "block supplier",
        "escalate this",
        "escalate to",
        "we recommend",
        "recommended action",
        "next step",
        "coming soon",
        "to be added",
        "not yet available",
    ):
        assert phrase not in text
    for phrase in (
        "supply chain paths for this supplier",
        "sayari tier",
        "within the search limits",
        "observed supply chain path",
        "shipment observation window",
    ):
        assert phrase in text
    assert presentation.MAX_EXHIBIT_SUPPLIERS == 8


def test_supplier_disclosures_are_open_without_js_and_script_has_safe_single_head_entry(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Supplier disclosures remain available when JavaScript is disabled.
    html = _html(r4_findings, tmp_path)
    document = _Document()
    document.feed(html)
    disclosures = [(tag, attrs) for tag, attrs in document.tags if "data-disclosure" in attrs]
    assert len(disclosures) > 50
    assert all(tag == "details" and "open" in attrs for tag, attrs in disclosures)
    supplier_attrs = [attrs for _, attrs in document.tags if "data-supplier-row" in attrs]
    assert len(supplier_attrs) == 50
    assert all(attrs["data-supplier-row"] is None for attrs in supplier_attrs)
    assert all(re.fullmatch(r"supplier-\d+", (attrs["id"] or "")) for attrs in supplier_attrs)
    assert all("data-filter-row" not in attrs for attrs in supplier_attrs)
    entity_attrs = [attrs for _, attrs in document.tags if "data-filter-row" in attrs]
    assert len(entity_attrs) == len(r4_findings.shared_nodes)
    assert all("data-disclosure" in attrs and "open" in attrs for attrs in entity_attrs)
    assert all(tag == "details" for tag, attrs in document.tags if "data-filter-row" in attrs)
    assert all("hidden" not in attrs for attrs in [*entity_attrs, *supplier_attrs])
    pagers = [attrs for tag, attrs in document.tags if tag == "nav"]
    assert {attrs["aria-label"] for attrs in pagers} == {
        "Shared-entity pages",
        "Supplier evidence pages",
    }
    assert all("hidden" in attrs for attrs in pagers)
    ids = {attrs["id"] for _, attrs in document.tags if "id" in attrs}
    page_buttons = [
        attrs for tag, attrs in document.tags if tag == "button" and "aria-controls" in attrs
    ]
    assert len(page_buttons) == 4
    assert all(
        attrs["type"] == "button" and attrs["aria-controls"] in ids for attrs in page_buttons
    )
    assert ".filter, .pagination { display: none; }" in html
    for guard in (
        ".js details[data-disclosure] > .disclosure-body { display: none }",
        ".js-ready details[data-disclosure][open] > .disclosure-body { display: block }",
    ):
        assert guard in html
    assert html.index("<script>") < html.index("</head>")
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None
    script = match.group(1)
    assert script.strip().startswith("document.documentElement.classList.add('js');")
    assert html.count("<script>") == html.count("<style>") == 1
    for forbidden in (
        "innerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "fetch(",
        "XMLHttpRequest",
        "url(",
        "<mark",
        "replaceChild",
        ".remove(",
    ):
        assert forbidden not in html
    section = _section(html, "suppliers")
    assert (
        "No supplier entry in this report matched the current search and filters. This report "
        "contains only "
        "the supplier input rows screened in this run — screening a new entity requires a "
        "new screening run."
    ) in _visible_text(section)
    assert 'id="supplier-no-match" class="notice" hidden' in section
    for control in ("supplier-filter-count",):
        attrs = next(attrs for _, attrs in document.tags if attrs.get("id") == control)
        assert attrs["role"] == "status" and attrs["aria-live"] == "polite"


def test_supplier_hop_entity_and_factor_payloads_remain_literal_escaped_text(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Untrusted supplier, hop and factor text renders only as escaped text.
    payload = """原始<script>alert("x")</script> & ' </text><image href="bad">"""
    source = next(
        row for row in cast(list[dict[str, Any]], r4_findings.suppliers) if row["entity_id"] == "s1"
    )
    source["input_name"] = source["label"] = source["translated_label"] = payload
    profile: Any = source["profile"]
    profile["risk_factors"][0]["factor"] = payload
    entities: Any = cast(dict[str, Any], r4_findings.paths)["entities"]
    entities["middle"]["label"] = payload
    entities["middle"]["translated_label"] = payload
    cast(list[dict[str, Any]], r4_findings.shared_nodes)[0]["label"] = payload
    html = _html(r4_findings, tmp_path)
    assert payload in _visible_text(_section(html, "suppliers"))
    assert payload not in html
    assert "&lt;script&gt;" in html and "&lt;image href=&#34;bad&#34;&gt;" in html
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None
    script = match.group(1)
    assert "alert" not in script and "原始" not in script
    document = _Document()
    document.feed(html)
    for tag, attrs in document.tags:
        assert tag != "image"
        assert not any(
            key.startswith("on") or key in {"src", "href", "xlink:href"} for key in attrs
        )
        if "data-supplier-row" in attrs:
            assert re.fullmatch(r"supplier-\d+", (attrs["id"] or ""))
    assert html.count("<script>") == 1


@pytest.mark.parametrize("headline", ["list_3", "custom_headline"])
def test_report_list_labels_follow_role_without_exposing_headline_key(
    findings: Findings, tmp_path: Path, headline: str
) -> None:
    # Rename the actual evidence scope; a hardcoded list_3 alias cannot satisfy this. Supplier lists
    # are labelled by role, so the internal headline key never appears.
    payload = findings.model_dump_json().replace("list_3", headline)
    renamed = Findings.model_validate_json(payload)
    before = renamed.model_dump()
    html = _html(renamed, tmp_path)
    assert headline not in html
    assert "Screened supplier list" in html
    assert "Supplier list: another_portfolio" in html
    assert renamed.model_dump() == before
    assert renamed.manifest["headline_portfolio"] == headline


def test_psa_factors_are_separate_on_every_factor_surface(
    findings: Findings, tmp_path: Path
) -> None:
    # Possible-identity factors stay distinct from network signals on every factor surface.
    from sayari_poc.models import OntologyFactor

    slugs = ["psa_misleading_name", "no_prefix"]
    for slug, kind, label in [
        (slugs[0], "network", "Published network signal"),
        (slugs[1], "psa", "Published possible identity signal"),
    ]:
        findings.ontology[slug] = OntologyFactor.model_validate(
            {
                "id": slug,
                "label": label,
                "description": "Published definition for " + label,
                "level": "high",
                "risk_type": kind,
                "categories": [],
            }
        )
    for node in cast(list[dict[str, Any]], findings.shared_nodes):
        node["severe_factors"] = list(slugs)
        node["other_factors"] = ["unknown_hs_signal", "psa_unknown_hs_signal"]
    convergence = cast(dict[str, Any], findings.convergence)
    convergence["list_3"]["suppressed_hubs"][0]["severe_factors"] = list(slugs)
    for supplier in cast(list[dict[str, Any]], findings.suppliers):
        if supplier["profile"]:
            supplier["profile"]["risk_factors"] = [
                {"factor": slug, "level": "high", "value": True} for slug in slugs
            ]
    before = findings.model_dump()
    html = _html(findings, tmp_path)
    surfaces = [
        _section(html, "key-finding"),
        _shared_findings(html),
        _section(html, "suppliers"),
    ]
    for surface in surfaces:
        assert surface is not None
        content = surface if isinstance(surface, str) else surface.group(1)
        lists = re.findall(r'<ul class="factor-list[^"]*">(.*?)</ul>', content, re.S)
        assert lists
        for factor_list in lists:
            items = re.findall(r"<li>(.*?)</li>", factor_list, re.S)
            names = [
                re.findall(r'<span class="factor-label">(.*?)</span>', item)[0] for item in items
            ]
            assert names in [
                ["Published network signal", "Published possible identity signal"],
                ["unknown_hs_signal", "psa_unknown_hs_signal"],
            ]
            for name, item in zip(names, items, strict=True):
                badges = re.findall(r'<span class="factor-badge">(.*?)</span>', item)
                expected_badges = {
                    "unknown_hs_signal": ["Unclassified"],
                    "psa_unknown_hs_signal": [
                        "Unclassified",
                        "Possible identity involvement — unconfirmed",
                    ],
                    "Published network signal": [
                        "Network risk",
                        "Possible identity involvement — unconfirmed",
                        "high",
                    ],
                    "Published possible identity signal": [
                        "Possible identity match — unconfirmed",
                        "high",
                    ],
                }
                assert badges == expected_badges[name]
                if "Published" in name:
                    # Level and risk type are on the badge row, asserted above.
                    assert "Sayari severity" not in item
                else:
                    assert "No published definition in the committed ontology snapshot." in item
                    assert '<span class="factor-badge">Unclassified</span>' in item
                assert "<code>" not in item
        additions = re.findall(r'<div class="factor-additional">(.*?)</div>', content, re.S)
        for group in additions:
            document = _Document()
            document.feed("<div>" + group + "</div>")
            children = [
                node
                for node, depth in zip(document.tags, document.depths, strict=True)
                if depth == 1
            ]
            assert [tag for tag, _ in children] == ["p", "ul"]
            count = re.search(r'<span class="factor-count">(\d+)</span>', group)
            assert count is not None and int(count[1]) == group.count("<li>")
            assert "<small" in group and "<details" not in group
        assert content.count("Possible-identity factors do not confirm") == content.count(
            'class="factor-section"'
        )
        assert content.count("Possible identity involvement is unconfirmed;") == content.count(
            'class="factor-section"'
        )
        if content != _shared_findings(html):
            assert not additions
    assert "Published network signal" in html
    assert "Published possible identity signal" in html
    assert "Published definition for Published possible identity signal" in html
    assert "No glossary entry" not in html
    assert (
        "potential risk through entity similarities that suggest a relationship "
        "but lack definitive proof" in html
    )
    assert findings.model_dump() == before


def test_reading_guide_defines_terms_and_no_local_jargon_in_authored_copy(
    findings: Findings, tmp_path: Path
) -> None:
    # The reading guide defines source terms, and the authored copy uses no internal jargon.
    html = _html(findings, tmp_path)
    guide = html.split('<div class="reading-guide">', 1)[1].split("</div>", 1)[0]
    # ISO/IEC Directives Part 2 makes "Terms and definitions" the normative clause-3 heading; this
    # block is a definitions list, so it carries that title.
    assert "Terms and definitions" in guide
    terms = re.findall(r"<strong>(.*?)</strong>", guide)
    assert terms == [
        "Entity:",
        "Match strength:",
        "Upstream:",
        "Tier:",
        "Risk factor:",
        "Seed risk:",
        "Network risk:",
        "PSA risk (Possibly-Same-As):",
        "Level:",
        "Shared sub-tier supplier:",
        "HS code:",
    ]
    assert "original identifier for it instead of a name" in guide
    body = _visible_text(html.split("<body>", 1)[1])
    assert not re.search(r"\b(retained|convergence|qualifying|hub|funnel|portfolio)\b", body, re.I)
    document = _Document()
    document.feed(html)
    assert all("open" in attrs for tag, attrs in document.tags if tag == "details")


def test_customer_report_omits_identifiers_and_duplicate_input_metadata(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Customer-facing copy leaves out canonical identifiers and duplicate input aliases.
    identifiers: set[str] = set()

    def collect_ids(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.endswith("_id") and isinstance(item, str):
                    identifiers.add(item)
                collect_ids(item)
        elif isinstance(value, list):
            for item in value:
                collect_ids(item)

    collect_ids(r4_findings.model_dump())
    payload = r4_findings.model_dump_json()
    hidden_ids = []
    for position, identifier in enumerate(sorted(identifiers)):
        sentinel = f"hidden_entity_identifier_{position}"
        payload = payload.replace(json.dumps(identifier), json.dumps(sentinel))
        hidden_ids.append(sentinel)
    renamed = Findings.model_validate_json(payload)
    for node in cast(list[dict[str, Any]], renamed.shared_nodes):
        for supplier in node["suppliers"]:
            supplier["input_names"] = ["Hidden duplicate input alias"]
    before = renamed.model_dump()
    html = _html(renamed, tmp_path)
    assert all(identifier not in html for identifier in hidden_ids)
    assert "Hidden duplicate input alias" not in html
    assert "input names:" not in html.lower()
    assert "entity id" not in html.lower()
    assert "also appear in the data export" in html
    table = _shared_findings(html)
    entries = re.findall(
        r'<details data-filter-row class="entity-card" data-disclosure open[^>]*>(.*?)</summary>',
        table,
        re.S,
    )
    assert len(entries) == len(renamed.shared_nodes)
    assert all('class="entity-identity"' in entry for entry in entries)
    assert all('class="entity-countries"' in entry for entry in entries)
    assert all('class="entity-metrics"' in entry for entry in entries)
    assert all(
        "distinct suppliers" in entry and "critical/high factor" in entry for entry in entries
    )
    assert "Reported risk factors" in table and "Connected supplier evidence" in table
    assert len(_supplier_entries(html)) == len(renamed.suppliers) == 50
    for node in cast(list[dict[str, Any]], renamed.shared_nodes):
        assert node["label"] in _visible_text(table)
        for slug in [*node["severe_factors"], *node["other_factors"]]:
            if "_" in slug:
                assert slug not in html
    assert renamed.model_dump() == before


def test_published_factor_text_is_escaped_and_fallback_requires_absence(
    findings: Findings, tmp_path: Path
) -> None:
    # Published labels are escaped, and the fallback appears only when the text is truly absent.
    from sayari_poc.models import OntologyFactor

    for slug, label, description in [
        ("known", "<Published & label>", "Published <definition> & details"),
        ("blank", "Published without definition", ""),
    ]:
        findings.ontology[slug] = OntologyFactor(
            id=slug,
            label=label,
            description=description,
            categories=[],
            level="high",
            risk_type="network",
        )
    node = next(node for node in findings.shared_nodes if node["portfolio"] == "list_3")
    node["severe_factors"] = ["known", "blank", "genuinely_missing_factor"]
    html = _html(findings, tmp_path)
    key = _section(html, "key-finding")
    assert "&lt;Published &amp; label&gt;" in key
    assert "Published &lt;definition&gt; &amp; details" in key
    assert "Published without definition" in key
    assert "genuinely_missing_factor" in key
    assert "Genuinely missing factor" not in key
    assert key.count("No published definition in the committed ontology snapshot.") == 2
    assert '<span class="factor-badge">Unclassified</span>' in key


def test_methodology_states_the_selection_rule_and_never_a_score(
    findings: Findings, tmp_path: Path
) -> None:
    # The methodology explains selection by published level without inventing a score.
    before = findings.model_dump()
    html = _html(findings, tmp_path)
    text = _visible_text(_section(html, "methodology"))
    # The rule itself: Sayari's own published levels, and only the two that qualify.
    assert "Sayari’s own published critical and high levels" in text
    # A network value must never read as a score, whatever else the section drops.
    assert "distance to the nearest risk target" in text
    assert "not a score" in text
    # PSA moved here from the removed identity block, and these badges appear on hundreds of
    # factors, so the section must still define them.
    assert "PSA means “possibly the same as”" in text
    assert "never a confirmed one" in text
    # The honesty rules bind the score qualification regardless of which section carries it.
    # Asserted on the delivered artifact: the synthetic fixture renders no candidates.
    delivered = _visible_text(
        (Path(__file__).resolve().parents[2] / "docs/report.html").read_text(encoding="utf-8")
    )
    assert "relevance score, not a calibrated confidence or probability" in delivered
    assert "Possible identity matches are unconfirmed" in delivered
    assert findings.model_dump() == before


@pytest.mark.parametrize("status", ["weak", "error", "no_match", "not_attempted"])
def test_unadjudicated_candidates_keep_response_order_and_escape_labels(
    r4_findings: Findings, tmp_path: Path, status: str
) -> None:
    # Candidate review keeps the original response positions and escapes source labels.
    row = cast(dict[str, Any], r4_findings.suppliers[0])
    row.update(
        resolution_status=status,
        candidate_count=3,
        candidate_errors={"1": "Malformed alternate candidate"},
        candidates=[
            {
                "label": "First <candidate>",
                "match_strength": {"value": "weak"},
                "score": 88.5,
                "translated_label": None,
                "countries": ["USA", "JPN"],
            },
            {
                "label": "Later & stronger",
                "match_strength": {"value": "strong"},
                "score": 272.6,
                "translated_label": "原始候補",
                "countries": [],
            },
        ],
    )
    before = r4_findings.model_dump()
    html = _html(r4_findings, tmp_path)
    entry = next(
        value for value in _supplier_entries(html).values() if "First &lt;candidate&gt;" in value
    )
    table = re.search(r'<table class="candidate-table">(.*?)</table>', entry, re.S)
    assert table is not None
    text = _visible_text(table[1])
    assert text.index("First <candidate>") < text.index("Later & stronger")
    assert "<tr><td>0</td>" in table[1] and "<tr><td>2</td>" in table[1]
    assert "USA, JPN" in text and "None returned" in text
    assert "88.5" in text and "272.6" in text and "%" not in text
    assert "原始候補" in text
    assert "Unadjudicated resolution candidates" in entry
    assert "without local re-ranking" in entry
    assert "not a calibrated confidence" in entry
    assert "Candidate at response index 1: Malformed alternate candidate" in entry
    assert "hidden" not in table[1]
    assert r4_findings.model_dump() == before


@pytest.mark.parametrize(
    "status, expected",
    [
        ("no_match", "No candidates returned by the successful resolution request"),
        ("error", "Resolution failed; no validated candidate evidence"),
        ("not_attempted", "No validated candidate evidence retained"),
    ],
)
def test_empty_adjudication_states_are_distinct(
    r4_findings: Findings, tmp_path: Path, status: str, expected: str
) -> None:
    # An empty success, a failure and an unattempted resolution each get their own wording.
    r4_findings.suppliers = [r4_findings.suppliers[0]]
    row = cast(dict[str, Any], r4_findings.suppliers[0])
    row.update(resolution_status=status, candidate_count=0, candidates=[], candidate_errors={})
    html = _html(r4_findings, tmp_path)
    assert expected in html
    assert 'class="candidate-table"' not in html
    assert ("No candidates returned by the successful resolution request" in html) == (
        status == "no_match"
    )
    assert ("Sayari returned 0 possible records" in html) == (status == "no_match")
    assert ("Returned-candidate count unavailable" in html) == (status != "no_match")


def test_resolved_rows_do_not_offer_identity_adjudication(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Accepted identities get no table of unadjudicated candidates.
    r4_findings.suppliers = [
        row for row in r4_findings.suppliers if row["resolution_status"] == "resolved"
    ]
    assert "Unadjudicated resolution candidates" not in _html(r4_findings, tmp_path)


def test_r19_factor_cards_have_one_flat_additional_disclosure(
    findings: Findings, tmp_path: Path
) -> None:
    # Factor cards keep their additional evidence without nesting one disclosure in another.
    html = _html(findings, tmp_path)
    cards = re.split(
        r'<details data-filter-row class="entity-card" data-disclosure open[^>]*>',
        _shared_findings(html),
    )[1:]
    assert cards
    for card in cards:
        assert card.count("<details") == 0
        assert "Critical and high factors (" in card
        assert 'class="factor-section"' in card
        if 'class="factor-additional"' in card:
            assert "Additional reported factors (" in card
    for section in ("key-finding", "suppliers"):
        assert "factor-group" not in _section(html, section)


@pytest.mark.parametrize(
    "slug,kind,expected",
    [
        ("psa_seed", "seed", ["Seed risk", "Possible identity involvement — unconfirmed"]),
        ("psa_network", "network", ["Network risk", "Possible identity involvement — unconfirmed"]),
        ("psa_psa", "psa", ["Possible identity match — unconfirmed"]),
        ("psa_missing", None, ["Unclassified", "Possible identity involvement — unconfirmed"]),
        ("seed", "seed", ["Seed risk"]),
        ("network", "network", ["Network risk"]),
        ("no_prefix", "psa", ["Possible identity match — unconfirmed"]),
        ("missing", None, ["Unclassified"]),
    ],
)
def test_r19_badges_use_prefix_first_and_one_card_caveat(
    findings: Findings, tmp_path: Path, slug: str, kind: str | None, expected: list[str]
) -> None:
    # Factor badges keep uncertainty visible, and each card carries at most one caveat.
    from sayari_poc.models import OntologyFactor

    if kind is not None:
        findings.ontology[slug] = OntologyFactor.model_validate(
            {
                "id": slug,
                "label": "Badge probe",
                "description": "Published definition",
                "level": "critical",
                "risk_type": kind,
                "categories": [],
            }
        )
    for node in findings.shared_nodes:
        node["severe_factors"] = [slug]
        node["other_factors"] = [slug]
    card = re.split(
        r'<details data-filter-row class="entity-card" data-disclosure open[^>]*>',
        _shared_findings(_html(findings, tmp_path)),
    )[1]
    badges = re.findall(r'<span class="factor-badge">(.*?)</span>', card)
    assert badges == (expected + (["critical"] if kind else [])) * 2
    assert card.count('class="factor-note"') == int(
        any("unconfirmed" in badge for badge in expected)
    )
    if slug.startswith("psa_"):
        assert "this entity" not in badges and "via relationship" not in badges
    # Level and risk type are carried by the badge row, asserted above.
    assert "Sayari severity" not in card and "risk type:" not in card
    if kind is None:
        assert slug in card
        assert "No published definition in the committed ontology snapshot." in card


def test_r19_flat_factors_preserve_order_and_published_fields(
    findings: Findings, tmp_path: Path
) -> None:
    # Flattened factors keep their source order, published fields and uncertainty.
    from sayari_poc.models import OntologyFactor

    slugs = ["second", "psa_first", "unknown"]
    for slug, kind in [("second", "network"), ("psa_first", "seed")]:
        findings.ontology[slug] = OntologyFactor.model_validate(
            {
                "id": slug,
                "label": slug + " label",
                "description": slug + " definition",
                "level": "high",
                "risk_type": kind,
                "categories": [],
            }
        )
    for node in findings.shared_nodes:
        node["severe_factors"] = list(slugs)
        node["other_factors"] = []
    key = _section(_html(findings, tmp_path), "key-finding")
    # The narrow-screen supplier list (R-5) also uses list items; only factors are checked here.
    key = re.sub(r'<ul class="exhibit-list">.*?</ul>', "", key, flags=re.S)
    items = re.findall(r"<li>(.*?)</li>", key, re.S)
    assert [re.findall(r'<span class="factor-label">(.*?)</span>', item)[0] for item in items] == [
        "second label",
        "psa_first label",
        "unknown",
    ]
    assert (
        '<span class="factor-badge">Possible identity involvement — unconfirmed</span>' in items[1]
    )
    assert "Sayari severity" not in items[1]
    assert "psa_first definition" in items[1]
    assert "No published definition in the committed ontology snapshot." in items[2]


def test_r19_exposure_headings_show_connections_without_rank(
    r4_findings: Findings, tmp_path: Path
) -> None:
    # Exposure headings describe the connected suppliers and never show a rank.
    view = presentation.build_report_view(r4_findings)
    html = _html(r4_findings, tmp_path)
    entries = _supplier_entries(html)
    for row in view.supplier_rows:
        headings = [_visible_text(x) for x in re.findall(r"<h4>(.*?)</h4>", entries[row.dom_id])]
        assert not any("Rank" in h for h in headings)
        for node in row.exposure_nodes:
            assert (
                f"{node.node['label']} — {node.node['supplier_count']} connected suppliers"
                in headings
            )
    assert presentation.build_report_view(r4_findings) == view


@pytest.mark.parametrize("stored,retrieved", [(0, 0), (4, 4), (4, 9)])
def test_r19_path_ratios_only_explain_truncation(
    r4_findings: Findings, tmp_path: Path, stored: int, retrieved: int
) -> None:
    # Path ratios explain only how the examples were truncated, without repeating the methodology.
    paths = cast(dict[str, Any], r4_findings.paths)["nodes"]["n-top"]
    paths.update(
        stored_path_count=stored, retrieved_path_count=retrieved, truncated=stored < retrieved
    )
    if stored == 0:
        paths.update(paths=[], path_observed_supplier_ids=[])
    html = _html(r4_findings, tmp_path)
    text = _visible_text(_section(html, "suppliers"))
    assert "Paths shown for this entity:" not in text
    if stored < retrieved:
        # The ratio is node-level; the sentence must say so, because the routes rendered beneath it
        # are filtered to the one supplier whose entry this is.
        top = next(
            node
            for row in presentation.build_report_view(r4_findings).supplier_rows
            for node in row.exposure_nodes
            if node.node["upstream_id"] == "n-top"
        )
        assert (
            f"Sayari returned {retrieved} supply chain paths to this entity across all "
            f"{top.node['supplier_count']} connected suppliers; this report keeps {stored} "
            "of them as examples." in text
        )
        # The zero-path branch must never pick up the retrieved-count tail.
        assert "no supply chain paths to this entity across" not in text
    elif stored:
        # Nothing was truncated, so the line says so rather than repeating the count.
        assert "and this report keeps all of them." in text
        assert f"{stored} of {retrieved}" not in text
    else:
        assert "Sayari returned no supply chain paths to this entity" in text
        assert "That does not mean no path exists." in text
    # A stretch of the sentence with no run-specific value, so it reads the same in every fixture.
    method = (
        "Examples are taken from each connected supplier in rotation, never ranked, "
        "so they cover as many suppliers as possible."
    )
    assert method not in text
    assert _visible_text(_section(html, "methodology")).count(method) == 1


def test_display_limit_is_named_when_a_supplier_loses_the_round_robin(
    r4_findings: Findings, tmp_path: Path
) -> None:
    """Name the display limit for a supplier that has evidence but no stored example.

    Stored paths are capped at PATH_MAX_PER_NODE and shared across a node's suppliers, so a
    node with more connected suppliers than slots leaves some of them showing a count and no
    route. That is a display limit, and it must never read as absent retrieved evidence.
    """

    def node_block(entry: str, label: str) -> str:
        # Isolate one exposure entry, because a supplier card carries several.
        return next(block for block in entry.split("<h4>") if label in block)

    node = cast(dict[str, Any], r4_findings.paths)["nodes"]["n-top"]
    # Keep s2 credited with retrieved evidence while removing its stored example.
    node["paths"] = [path for path in node["paths"] if path["source_entity_id"] != "s2"]
    assert "s2" in node["path_observed_supplier_ids"]

    entries = _supplier_entries(_html(r4_findings, tmp_path))
    rows = {
        row.supplier["entity_id"]: row
        for row in presentation.build_report_view(r4_findings).supplier_rows
    }
    # Take the heading label from the ranked node itself, which is what the <h4> renders.
    top_label = next(
        node.node["label"]
        for node in rows["s2"].exposure_nodes
        if node.node["upstream_id"] == "n-top"
    )
    loser = _visible_text(node_block(entries[rows["s2"].dom_id], top_label))
    winner = _visible_text(node_block(entries[rows["s1"].dom_id], top_label))
    display_limit = (
        "This supplier’s supply chain paths to this entity are not among the examples kept, "
        "which are shared across all"
    )

    assert display_limit in loser
    # The supplier does have retrieved evidence, so the absent-route sentence must not appear.
    assert "That does not mean no path exists." not in loser
    assert "Observed supply chain path" not in loser
    # A supplier that kept an example still renders its route and no display-limit notice.
    assert display_limit not in winner
    assert "Observed supply chain path" in winner


def test_profile_values_with_uninterpreted_units_are_retained_but_not_displayed(
    findings: Findings, tmp_path: Path
) -> None:
    # Values with no interpreted unit stay in the evidence but are left out of the display.
    for index, supplier in enumerate(cast(list[dict[str, Any]], findings.suppliers)):
        supplier["psa_risky"] = [True, False, None][index % 3]
        if supplier["profile"]:
            supplier["profile"]["risk_factors"] = [
                {"factor": f"ordinary_factor_{i}", "level": "relevant", "value": value}
                for i, value in enumerate([True, False, 0, 1, 2.5, "source value"])
            ]
    before = findings.model_dump()
    html = _html(findings, tmp_path)
    section = _section(html, "suppliers")
    text = _visible_text(section)
    assert not re.search(r"\b(?:True|False)\b", text)
    assert "Signal reported as present" in text and "Signal reported as absent" in text
    # Sayari publishes no unit for these, so they are left unstated rather than shown as bare
    # numbers a reader cannot interpret.
    assert "Sayari-reported value" not in text
    for index in range(6):
        items = re.findall(
            rf'<li><span class="factor-label">ordinary_factor_{index}</span>(.*?)</li>',
            section,
            re.S,
        )
        assert items
        assert all(('class="factor-value"' in item) == (index < 2) for item in items)
    assert findings.model_dump() == before
    method = _visible_text(_section(html, "methodology"))
    assert "Every other value is shown exactly as Sayari returned it" not in method
    assert "Values without interpreted units" in method
    assert "At least one factor above comes from one of those unconfirmed records" in text
    assert "No factor above comes from one of those unconfirmed records" in text
    assert "Sayari did not classify every factor" in text


def test_r19_factor_sections_state_their_distinct_scope(findings: Findings, tmp_path: Path) -> None:
    # The supplier and upstream factor sections each state their own scope.
    html = _html(findings, tmp_path)
    assert (
        "Reported for this upstream entity. Critical and high factors are shown first, "
        "followed by other returned factors."
    ) in _visible_text(_shared_findings(html))
    text = _visible_text(_section(html, "suppliers"))
    assert "Risk factors reported for this supplier" in text
    assert (
        "Every risk factor Sayari reported for this supplier, at all severity levels. "
        "Factors Sayari reported as not present are left out."
    ) in text


def test_r19_profile_maximum_label_is_sayari_reported_not_a_selection_rule(
    findings: Findings, tmp_path: Path
) -> None:
    # A profile's maximum level is shown as Sayari reported it, not as a selection decision.
    for row in cast(list[dict[str, Any]], findings.suppliers):
        if row["profile"]:
            row["profile"]["max_level"] = "relevant"
    text = _visible_text(_section(_html(findings, tmp_path), "suppliers"))
    assert "Most severe level Sayari reported on any single factor: relevant" in text
    assert "under the report rules" not in text
    assert "none assigned" not in text


def test_readability_outer_accordion_and_plain_factor_sections(
    findings: Findings, tmp_path: Path
) -> None:
    # Outer disclosures keep factor sections readable, with no nested cards.
    html = _html(findings, tmp_path)
    cards = re.findall(r"<details data-filter-row[^>]*>(.*?)</details>", html, re.S)
    assert len(cards) == len(findings.shared_nodes)
    assert all("<details" not in card for card in cards)
    assert all("Critical and high factors (" in card for card in cards)
    assert "Selected by the review rules" not in html
    # Source details render unconditionally now that the toggle is gone; the rule that hid them went
    # with it, so evidence can no longer be hidden behind a control.
    assert 'class="source-meta"' in html
    assert ".js .source-meta" not in html
    assert "show-source-details" not in html


@pytest.mark.parametrize(
    "slug,kind,label,expected,forbidden",
    [
        (
            "psa_trade",
            "network",
            "Trade (May Include 'PSA' Path)",
            ["Network risk", "May include PSA links", "high"],
            "Possible identity match",
        ),
        (
            "psa_goods",
            "seed",
            "Goods (May Include 'PSA' Path)",
            ["Seed risk", "May include PSA links", "high"],
            "Possible identity match",
        ),
        (
            "psa_match",
            "psa",
            "Possible match",
            ["Possible identity match — unconfirmed", "high"],
            "Network risk",
        ),
        ("basel_aml", "seed", "Basel AML Index", ["Country indicator", "high"], "this entity"),
    ],
)
def test_readability_distinguishes_psa_paths_and_identity(
    findings: Findings,
    tmp_path: Path,
    slug: str,
    kind: str,
    label: str,
    expected: list[str],
    forbidden: str,
) -> None:
    # The copy tells a possible-identity match apart from uncertainty along a path.
    from sayari_poc.models import OntologyFactor

    findings.ontology[slug] = OntologyFactor.model_validate(
        {
            "id": slug,
            "label": label,
            "description": "Published definition",
            "risk_type": kind,
            "level": "high",
            "categories": [],
        }
    )
    for node in findings.shared_nodes:
        node["severe_factors"] = [slug]
        node["other_factors"] = []
    key = _section(_html(findings, tmp_path), "key-finding")
    assert re.findall(r'<span class="factor-badge">(.*?)</span>', key) == expected
    assert forbidden not in key
    if "May include PSA links" in expected:
        assert "may rest on a company match Sayari has not confirmed" in key
        assert "matched entity's risk" not in key


@pytest.mark.parametrize(
    "slug,kind,value,expected",
    [
        ("network_probe", "network", 4, "Risk-network distance: 4 links"),
        ("network_probe", "network", 1, "Risk-network distance: 1 link"),
        ("network_probe", "network", 0, "Risk-network distance: 0 links"),
        ("network_probe", "network", True, "Signal reported as present"),
        ("network_probe", "network", False, "Signal reported as absent"),
        ("network_probe", "network", 2.5, None),
        ("network_probe", "network", "4", None),
        ("basel_aml", "seed", 6.78, "Jurisdiction AML index: 6.78 / 10"),
        ("cpi_score", "seed", 77, "Country corruption perceptions index: 77 / 100"),
        ("unknown_probe", None, 4, None),
    ],
)
def test_readability_profile_value_semantics_and_source_levels(
    findings: Findings,
    tmp_path: Path,
    slug: str,
    kind: str | None,
    value: object,
    expected: str | None,
) -> None:
    # Displayed values use their published meanings and keep their source levels.
    from sayari_poc.models import OntologyFactor

    if kind:
        findings.ontology[slug] = OntologyFactor.model_validate(
            {
                "id": slug,
                "label": "Value probe",
                "description": "Published definition",
                "risk_type": kind,
                "level": "high",
                "categories": [],
            }
        )
    for row in cast(list[dict[str, Any]], findings.suppliers):
        if row["profile"]:
            row["profile"]["risk_factors"] = [{"factor": slug, "level": "elevated", "value": value}]
    html = _html(findings, tmp_path)
    section = _section(html, "suppliers")
    if expected is None:
        # No published unit, so no value sentence renders at all.
        assert "Sayari-reported value" not in _visible_text(section)
        assert 'class="factor-value"' not in section
    else:
        assert expected in _visible_text(section)
    assert '<span class="factor-badge">elevated</span>' in section
    # The badge row is the only carrier of level and risk type.
    assert "Sayari severity" not in section and "API value" not in section
    assert "Profile-reported level" not in section
    if kind:
        assert "published level for this factor is high" in section
    if value is True or value is False or isinstance(value, str) or kind is None:
        assert "Risk-network distance:" not in section
    if slug == "basel_aml":
        assert "0 = lower risk; 10 = higher risk" in section
    if slug == "cpi_score":
        assert "0 = highly corrupt; 100 = very clean" in section


@pytest.mark.parametrize(
    "status,count,risky",
    [
        ("available", 1, True),
        ("available", 0, False),
        ("available", 3, None),
        ("unavailable", None, None),
        ("not_resolved", None, None),
    ],
)
def test_readability_identity_summary_keeps_count_presence_and_availability_separate(
    findings: Findings, tmp_path: Path, status: str, count: int | None, risky: bool | None
) -> None:
    # The identity summary keeps the reported count, risky matches and availability separate.
    for row in cast(list[dict[str, Any]], findings.suppliers):
        row.update(psa_status=status, psa_count=count, psa_risky=risky)
    text = _visible_text(_section(_html(findings, tmp_path), "suppliers"))
    assert "Possible identity match status:" not in text
    assert "a possible identity match carries a risk factor" not in text
    assert "Other records that may be the same company" in text
    if status == "not_resolved":
        assert "Not checked — this supplier was not matched to a Sayari record" in text
    elif status == "unavailable":
        assert "Not available — this supplier’s company profile could not be retrieved" in text
    else:
        assert f"Sayari holds {count} other record" in text
        assert "Sayari has not confirmed they are the same" in text
        if risky is True:
            assert "At least one factor above comes from one of those unconfirmed records" in text
        elif risky is False:
            assert "No factor above comes from one of those unconfirmed records" in text
            assert "does not mean this supplier is free of risk" in text
        else:
            assert "Sayari did not classify every factor" in text
            assert "not evidence of no risk" in text


@pytest.mark.parametrize("variant", ["original", "critical", "unavailable"])
def test_worklist_agrees_with_its_rollup_and_keeps_evidence_beside_every_count(
    findings: Findings, tmp_path: Path, variant: str
) -> None:
    """Assert the table against the rollup that fed it, never against literals.

    The counts are pinned by tests/unit/test_rollups.py against hand-counted evidence;
    here the question is only whether the section renders what was computed.
    """
    manifest = cast(dict[str, Any], findings.manifest)
    headline = manifest["headline_portfolio"]
    worklist = cast(dict[str, Any], findings.convergence[headline])["suppliers_ranked"]
    if variant == "critical":
        worklist[0]["critical_factors"] = 1
        worklist[0]["severe_factors"] = 2
    elif variant == "unavailable":
        worklist[0]["coverage"] = "error"
        for key in (
            "upstream",
            "flagged",
            "severe_factors",
            "critical_factors",
            "elevated_factors",
        ):
            worklist[0][key] = None
    html = _html(findings, tmp_path)
    section = _section(html, "supplier-worklist")
    text = _visible_text(section)

    # The table shows a bounded sample and says how many of how many it is showing.
    shown = worklist[:10]
    assert f"Showing the {len(shown)} most connected of {len(worklist)} screened suppliers" in text
    table = re.search(r"<tbody>(.*?)</tbody>", section, re.S)
    assert table
    rendered_rows = re.findall(r"<tr>(.*?)</tr>", table[1], re.S)
    assert len(rendered_rows) == len(shown)
    for rendered, record in zip(rendered_rows, shown, strict=True):
        cells = [_visible_text(cell) for cell in re.findall(r"<td>(.*?)</td>", rendered, re.S)]
        counts = [
            "Unavailable" if record[key] is None else str(record[key])
            for key in ("upstream", "flagged", "severe_factors", "elevated_factors")
        ]
        if record["critical_factors"]:
            counts[2] += f" includes {record['critical_factors']} critical"
        assert cells == [
            str(record["label"]),
            presentation.COVERAGE_LABELS[str(record["coverage"])],
            *counts,
        ]
        assert ('class="bar"' in rendered) == (record["flagged"] is not None)
        if record["flagged"] is not None:
            assert f'style="width: {record["flagged_share"]}%"' in rendered

    # The honesty wording that makes the counts readable is present.
    assert "Incomplete evidence may understate the full upstream picture" in text
    assert "do not necessarily represent findings directly associated with the supplier" in text
    assert "All counts are minimums based on the evidence retrieved" in text
    assert "No critical-level factors were identified in this run" not in text
    assert "should not be interpreted as an absence of risk" in text
    if variant == "critical":
        assert "includes 1 critical" in text


def test_breakdown_bars_are_decorative_and_never_exceed_their_track(
    findings: Findings, tmp_path: Path
) -> None:
    # Worklist bars are decorative and stay within the zero-to-100 display range.
    section = _section(_html(findings, tmp_path), "supplier-worklist")
    # Every bar width is a whole percent inside the track.
    widths = [int(value) for value in re.findall(r"width: (\d+)%", section)]
    assert widths and all(0 <= width <= 100 for width in widths)
    # The track is hidden from assistive technology; the number beside it carries the value.
    assert section.count('class="bar"') == section.count('aria-hidden="true"')
    # None of the report's document-wide JavaScript hooks may appear here, or these rows would
    # silently join the shared-entity filter and its ten-row pager.
    for hook in (
        "data-filter-row",
        "data-supplier-row",
        "data-filter-exhibit",
        "data-disclosure",
        "data-entity-group",
    ):
        assert hook not in section
