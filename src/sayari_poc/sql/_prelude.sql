-- Scoping shared by every convergence query, defined once and prepended by the loader.
--
-- Table macros rather than views: a view is reported by SHOW TABLES, which the warehouse
-- schema tests pin to exactly the five real tables, and a view cannot take the portfolio
-- and threshold parameters these definitions need. The parameter is named
-- target_portfolio because a macro parameter shadows any column of the same name.

-- Edges from this portfolio's resolved suppliers only: a weak row carries a candidate ID
-- but was never retrieved, so without this guard two weak rows could fabricate convergence.
CREATE OR REPLACE MACRO scoped(target_portfolio) AS TABLE
    SELECT DISTINCT su.upstream_id, su.supplier_id
    FROM supplier_upstream su
    JOIN suppliers s
      ON s.entity_id = su.supplier_id
     AND s.portfolio = target_portfolio
     AND s.resolution_status = 'resolved';

-- An entity reached by more than one distinct supplier. The loader omits self-edges, so
-- a traversal root cannot qualify by reaching itself.
CREATE OR REPLACE MACRO shared(target_portfolio) AS TABLE
    SELECT upstream_id, COUNT(DISTINCT supplier_id) AS supplier_count
    FROM scoped(target_portfolio)
    GROUP BY upstream_id
    HAVING COUNT(DISTINCT supplier_id) > 1;

-- One identity per canonical supplier; MIN makes the label choice deterministic when two
-- workbook rows resolve to the same entity under different input names.
CREATE OR REPLACE MACRO supplier_identity(target_portfolio) AS TABLE
    SELECT s.entity_id, MIN(s.label) AS label, MIN(s.translated_label) AS translated_label,
           LIST(DISTINCT s.input_name ORDER BY s.input_name) AS input_names
    FROM suppliers s
    WHERE s.portfolio = target_portfolio AND s.resolution_status = 'resolved'
    GROUP BY s.entity_id;

-- The suppliers attached to each shared entity, ordered by canonical ID for stable bytes.
CREATE OR REPLACE MACRO connections(target_portfolio) AS TABLE
    SELECT sc.upstream_id,
           LIST(STRUCT_PACK(entity_id := s.entity_id, label := s.label,
                            translated_label := s.translated_label,
                            input_names := s.input_names) ORDER BY s.entity_id) AS suppliers
    FROM scoped(target_portfolio) sc
    JOIN supplier_identity(target_portfolio) s ON s.entity_id = sc.supplier_id
    GROUP BY sc.upstream_id;

-- Traversal-reported factors only: profile evidence stays with the supplier it describes.
-- Unresolved factors have NULL is_severe and fall into other_factors, never severe_factors.
CREATE OR REPLACE MACRO upstream_factors() AS TABLE
    SELECT r.entity_id,
           LIST(DISTINCT r.factor ORDER BY r.factor)
               FILTER (WHERE r.is_severe) AS severe_factors,
           LIST(DISTINCT r.factor ORDER BY r.factor)
               FILTER (WHERE NOT COALESCE(r.is_severe, FALSE)) AS other_factors
    FROM risk_factors r
    WHERE r.source = 'upstream'
    GROUP BY r.entity_id;
