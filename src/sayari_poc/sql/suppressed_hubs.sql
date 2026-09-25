-- A deterministic bounded sample of the shared entities the country rule excludes.
-- Suppression precedes severity: broad-only hubs must remain in this evidence.
SELECT $portfolio AS portfolio, sh.upstream_id, sh.supplier_count,
       u.label, u.translated_label, u.countries, u.country_count, c.suppliers,
       COALESCE(f.severe_factors, []::TEXT[]) AS severe_factors,
       COALESCE(f.other_factors, []::TEXT[]) AS other_factors
FROM shared($portfolio) sh
JOIN upstream_entities u USING (upstream_id)
JOIN connections($portfolio) c USING (upstream_id)
LEFT JOIN upstream_factors() f ON f.entity_id = sh.upstream_id
WHERE u.country_count > $hub_max_countries
ORDER BY u.country_count DESC, sh.supplier_count DESC, sh.upstream_id ASC
LIMIT $limit;
