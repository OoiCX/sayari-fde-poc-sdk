"""Render the self-contained HTML report from Findings."""

from datetime import UTC, datetime
from pathlib import Path

import pycountry
from jinja2 import Environment, PackageLoader, select_autoescape

from sayari_poc.models import Findings
from sayari_poc.presentation import (
    build_report_view,
    exception_note,
    risk_categories,
    supplier_list_label,
)

# The SDK defaults resolution to a ten-entry page and we send neither limit nor offset, so a full
# page is a lower bound on candidates. See sayari/resolution/client.py: ResolutionClient.resolution.
RESOLUTION_PAGE_MAX = 10

# Spell out the months here instead of using strftime, whose month and day names follow the process
# locale and would make the rendered bytes depend on the machine.
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def report_datetime(value: object) -> str:
    """Render a stored ISO-8601 instant as "24 September 2026, 01:52 UTC".

    The stored form carries microseconds and a numeric offset, which reads as machine output and
    invites a reader in another zone to conclude the date is wrong. The instant is never
    converted to the host's local zone: that would make the rendered bytes depend on the machine
    and break byte-identical replay, so UTC is stated explicitly instead.
    """
    if not isinstance(value, str):
        return "" if value is None else str(value)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        # Show a timestamp we can't parse as it is; a wrong date is worse than a raw one.
        return value
    # Convert to UTC so any recorded offset renders as one comparable instant.
    moment = moment.astimezone(UTC) if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    # Build the day number ourselves, because %-d doesn't work on Windows.
    return f"{moment.day} {MONTH_NAMES[moment.month - 1]} {moment.year}, {moment:%H:%M} UTC"


def country_options(codes: list[str]) -> list[tuple[str, str]]:
    """Pair each ISO alpha-3 code with a readable name, sorted by that name.

    Cards show only codes, so the dropdown names them: "South Korea (KOR)". A code pycountry does
    not know is shown as it is rather than guessed. pycountry is pinned in requirements.lock, so
    the names, and therefore the rendered bytes, don't change between installs.
    """
    options = []
    for code in codes:
        country = pycountry.countries.get(alpha_3=code)
        name = (getattr(country, "common_name", None) or country.name) if country else None
        options.append((code, f"{name} ({code})" if name else code))
    return sorted(options, key=lambda option: (option[1].casefold(), option[0]))


def render_report(findings: Findings, out: Path) -> None:
    """Render HTML without retrieving additional evidence.

    Read packaged templates and create the output directory and file.
    Autoescaping applies to all templates, including SVG and .j2 files.
    """
    view = build_report_view(findings)
    # The HTML and SVG template filenames end in .j2, so extension-based autoescaping would miss
    # them. Turn escaping on for every packaged template.
    environment = Environment(
        loader=PackageLoader("sayari_poc", "templates"),
        autoescape=select_autoescape(default=True),
    )
    environment.globals["resolution_page_max"] = RESOLUTION_PAGE_MAX
    environment.filters["report_datetime"] = report_datetime
    environment.filters["supplier_list_label"] = supplier_list_label
    environment.filters["exception_note"] = exception_note
    environment.filters["country_options"] = country_options
    environment.filters["risk_categories"] = lambda slugs: risk_categories(slugs, view.glossary)

    def factor_psa_scope(slug: str) -> str:
        """Locate published possible-identity uncertainty."""
        factor = view.glossary.get(slug)
        # A published PSA type describes a possible match to another entity.
        if factor is not None and factor.risk_type == "psa":
            return "identity"
        # The published "path may include" wording can apply to network and seed trade factors.
        if factor is not None and (
            "network risk path may include" in factor.description.casefold()
            or ("may include" in factor.label.casefold() and "psa" in factor.label.casefold())
        ):
            # This permits identity links on paths without showing that a retrieved path has one.
            return "path"
        # A psa_ prefix without a usable published explanation can't locate the uncertainty.
        if slug.startswith("psa_"):
            return "unspecified"
        return "none"

    def factor_badges(slug: str, profile_level: str | None = None) -> list[str]:
        """Describe source signal type and level without an entity score."""
        factor = view.glossary.get(slug)
        scope = factor_psa_scope(slug)
        if factor is None:
            badges = ["Unclassified"]
        # These two jurisdiction indices describe the country, not allegations about the entity.
        elif slug in ("basel_aml", "cpi_score"):
            badges = ["Country indicator"]
        elif factor.risk_type == "network":
            badges = ["Network risk"]
        elif factor.risk_type == "psa":
            badges = ["Possible identity match — unconfirmed"]
        else:
            badges = ["Seed risk"]
        # The path badge is added alongside the signal type badge, not instead of it.
        if scope == "path":
            badges.append("May include PSA links")
        elif scope == "unspecified":
            badges.append("Possible identity involvement — unconfirmed")
        # Profiles carry their own level; factors seen only upstream use the ontology level.
        level = profile_level or (factor.level if factor is not None else None)
        if level is not None:
            badges.append(level)
        return badges

    def factor_value(slug: str, value: object) -> str:
        """Explain values only when their meaning is published."""
        # Test identity, not equality: True == 1 in Python, but a boolean signal isn't the number 1.
        if value is True:
            return "Signal reported as present"
        if value is False:
            # Live profiles omit false factors; if test evidence has one, show it as reported.
            return "Signal reported as absent"
        factor = view.glossary.get(slug)
        # Only explain numbers whose units are published; other values stay in Findings.
        if isinstance(value, (int, float)) and factor is not None:
            # The Basel factor publishes a 0-to-10 index for the jurisdiction.
            if slug == "basel_aml" and 0 <= value <= 10:
                return f"Jurisdiction AML index: {value} / 10 (0 = lower risk; 10 = higher risk)"
            # CPI runs on a different scale, and in the opposite direction, to the Basel index.
            if slug == "cpi_score" and 0 <= value <= 100:
                return (
                    f"Country corruption perceptions index: {value} / 100 "
                    "(0 = highly corrupt; 100 = very clean)"
                )
            # Only a whole, nonnegative network value can be a relationship distance.
            if (
                factor.risk_type == "network"
                and value >= 0
                and (isinstance(value, int) or value.is_integer())
            ):
                distance = int(value)
                unit = "link" if distance == 1 else "links"
                # This distance is in Sayari's risk network, not along the retrieved trade path.
                return (
                    f"Risk-network distance: {distance} {unit} to the nearest risk target "
                    "(Sayari-reported)"
                )
        # Sayari publishes no unit for any other value, so leave it out rather than show a bare
        # number the reader can't use. Only the branches above have a published meaning.
        return ""

    environment.filters["factor_psa_scope"] = factor_psa_scope
    environment.filters["factor_badges"] = factor_badges
    environment.filters["factor_value"] = factor_value
    html = environment.get_template("report.html.j2").render(**view.context())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
