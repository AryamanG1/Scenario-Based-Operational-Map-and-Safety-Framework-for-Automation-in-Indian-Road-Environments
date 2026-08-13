# Dashboard Road Data: provenance and regeneration

## Source

`dashboard/app.js`'s `ROADS` array is **not hand-picked**. Every entry comes from
a live query against OpenStreetMap's Overpass API, run once and pasted into the
file as a static array — the dashboard itself still has to work offline over
`file://` (the same constraint that governs `pipeline_stats.js` and
`carla_live.js`; see `dashboard/app.js`'s module docstring), so the data is baked
in at generation time rather than fetched at page load.

16 roads are currently baked in, each a single continuous polyline of real OSM
node coordinates.

## Regenerating it

1. Query Overpass for every named trunk/primary/secondary/tertiary road inside a
   bounding box matching the dashboard's map view (`map.setView([30.7333,
   76.7794], 13)` in `app.js`):

   ```
   [out:json][timeout:60];
   way["highway"~"^(trunk|primary|secondary|tertiary)$"]["name"]
     (30.715,76.760,30.760,76.812);
   out geom;
   ```

   POST that to `https://overpass-api.de/api/interpreter` as the `data` parameter.
2. Group the returned ways by `tags.name` — OSM splits a single named road into
   many short segments (one per intersection), so a real road comes back as
   dozens of disconnected pieces.
3. Chain each name's segments into one continuous path by repeatedly joining
   whichever unused segment's start/end point is closest to the current path's
   current endpoint (a ~40 m snap tolerance absorbs the minor
   floating-point/digitizing offsets between segments that should connect).
4. Drop any road whose chained length is under 400 m (junk/fragment names).
5. Sort by length and keep the rest. **This is what determines which roads appear
   at all** — no road name is chosen by hand.

The current data is 15 `arterial` roads and 1 `local`. Note that `road_type` is
**display-only** — `app.js` reads it in exactly one place, the map popup's "Type"
row; it feeds neither `calculateODDScore()` nor the polyline styling. The
one-shot generation script was not preserved, so the exact OSM `highway`-tag →
`road_type` mapping it applied is not recorded here; a regeneration should pick
its own mapping and document it. (The earlier hand-typed data also used a
`highway` value, which no longer appears in the generated set — nothing reads it,
so nothing broke.)

## The correction this replaced (project history)

An earlier pass hand-typed 12 road names with 3–4 guessed lat/lng points each.
Two problems, both visible on the rendered map:

- **Straight diagonal lines cutting across city blocks.** Three or four points
  cannot follow a real street's curvature, so every "road" rendered as a
  polyline that ignored the actual street grid underneath it.
- **Two roads placed several kilometres from their real locations.** The guessed
  coordinates for those names were simply wrong, and they rendered as
  disconnected strays at the edge of the map.

Sourcing every road from one bounding-box query eliminates both at once: no
guessed coordinates, and nothing outside the map's intended viewing area. This
is the same class of correction as the IDD-Lite class-mapping bug in
[`DATASET_NOTES.md`](DATASET_NOTES.md) — plausible-looking scaffold data that
direct inspection showed to be wrong.

## Known limitation

The bounding box was chosen to match the dashboard's existing map center/zoom,
not surveyed against which roads look best on screen. Some names from the earlier
hand-typed list (e.g. "Sector 17 Ring Road") do not exist in OSM under that name
and therefore no longer appear. If a future regeneration needs a different area,
widen or move the bbox in step 1 and re-run.

The ODD-readiness **scores** painted onto these roads remain synthetic demo
values from `app.js`'s `calculateODDScore()` — only the road geometry and names
are real. The panel that shows real pipeline output is "Real Pipeline Results",
fed by `dashboard/pipeline_stats.js` (see
[`modules/decision/feasibility_map.md`](modules/decision/feasibility_map.md)).
