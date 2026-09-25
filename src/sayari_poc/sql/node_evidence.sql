-- One ranked row per surviving shared entity: identity, connected suppliers, and the
-- traversal-reported factors behind its selection.
--
-- This was previously two queries whose rows were stitched together in Python. They
-- always corresponded 1:1 because both applied the same country and severity filters and
-- the same ordering; the column list below is their union, in the order the stitched
-- record produced, so serialized Findings bytes are unchanged.
SELECT sh.upstream_id, sh.supplier_count, u.label, u.translated_label,
       u.countries, u.country_count, $portfolio AS portfolio, c.suppliers,
       COALESCE(f.severe_factors, []::TEXT[]) AS severe_factors,
       COALESCE(f.other_factors, []::TEXT[]) AS other_factors
FROM shared($portfolio) sh
JOIN upstream_entities u USING (upstream_id)
JOIN connections($portfolio) c USING (upstream_id)
LEFT JOIN upstream_factors() f ON f.entity_id = sh.upstream_id
-- Suppress only country_count > threshold; equality survives.
WHERE u.country_count <= $hub_max_countries
-- Qualify only traversal-reported factors with a published critical or high level.
  AND EXISTS (
      SELECT 1 FROM risk_factors r
      WHERE r.entity_id = sh.upstream_id AND r.is_severe
        AND r.source = 'upstream'
  )
-- Most connected suppliers first, then fewest countries, then canonical ID for stable ties.
ORDER BY sh.supplier_count DESC, u.country_count ASC, sh.upstream_id ASC;
