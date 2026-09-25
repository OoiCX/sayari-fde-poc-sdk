-- The warehouse is a rebuildable artifact. The loader wraps DDL and loads in one transaction.
DROP TABLE IF EXISTS supply_paths;
DROP TABLE IF EXISTS supplier_upstream;
DROP TABLE IF EXISTS risk_factors;
DROP TABLE IF EXISTS upstream_entities;
DROP TABLE IF EXISTS suppliers;

CREATE TABLE suppliers (
    portfolio         TEXT NOT NULL,
    row_number        INTEGER NOT NULL,
    entity_id         TEXT,
    input_name        TEXT NOT NULL,
    label             TEXT,
    translated_label  TEXT,
    match_strength    TEXT,
    resolution_status TEXT NOT NULL,
    coverage_status   TEXT, -- NULL means retrieval was not attempted for this row.
    PRIMARY KEY (portfolio, row_number)
);

CREATE TABLE upstream_entities (
    upstream_id      TEXT PRIMARY KEY,
    label            TEXT,
    translated_label TEXT,
    countries        TEXT[],
    country_count    INTEGER NOT NULL
);

CREATE TABLE supplier_upstream (
    supplier_id TEXT NOT NULL,
    upstream_id TEXT NOT NULL,
    PRIMARY KEY (supplier_id, upstream_id)
);

CREATE TABLE risk_factors (
    entity_id     TEXT NOT NULL,
    factor        TEXT NOT NULL,
    source        TEXT NOT NULL,
    level         TEXT,
    -- NULL status means classification has not run; unresolved retains NULL source fields.
    is_severe     BOOLEAN,
    ontology_level TEXT,
    risk_type      TEXT,
    ontology_status TEXT,
    PRIMARY KEY (entity_id, factor)
);

CREATE TABLE supply_paths (
    source_entity_id   TEXT NOT NULL,
    path_index         INTEGER NOT NULL,
    hop_position       INTEGER NOT NULL,
    tier               INTEGER NOT NULL,
    entity_id          TEXT NOT NULL,
    terminal_entity_id TEXT NOT NULL,
    components         STRUCT(hs_code TEXT, departure_countries TEXT[],
                              arrival_countries TEXT[], min_date TEXT,
                              max_date TEXT)[] NOT NULL,
    PRIMARY KEY (source_entity_id, path_index, hop_position)
);
