-- Recomputes the published funnel over one scoped population, without factor fan-out.
WITH stages AS (
    SELECT sh.upstream_id, u.country_count,
           EXISTS (
               SELECT 1 FROM risk_factors r
               WHERE r.entity_id = sh.upstream_id AND r.is_severe
                 AND r.source = 'upstream'
           ) AS qualifies
    FROM shared($portfolio) sh
    JOIN upstream_entities u USING (upstream_id)
)
SELECT (SELECT COUNT(DISTINCT supplier_id) FROM scoped($portfolio)) AS contributing_suppliers,
       (SELECT COUNT(DISTINCT upstream_id) FROM scoped($portfolio)) AS distinct_upstream_entities,
       COUNT(*) AS shared_nodes,
       COUNT(*) FILTER (WHERE country_count > $hub_max_countries) AS suppressed_hubs,
       COUNT(*) FILTER (WHERE country_count <= $hub_max_countries) AS after_hub_suppression,
       COUNT(*) FILTER (WHERE country_count <= $hub_max_countries AND NOT qualifies)
           AS removed_by_severity_filter,
       COUNT(*) FILTER (WHERE country_count <= $hub_max_countries AND qualifies)
           AS after_severity_filter
FROM stages;
