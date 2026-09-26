WinterRadar — 3-hour retention / 30-minute archive chunks

Replace/add

.github/workflows/mrms.yml

.github/workflows/mrms_archive.yml

src/archive_mrms_history.py

src/publish_live_history.py

src/stage_history_for_pages.py

Architecture

Live workflow: every 5 minutes; latest MRMS scan and current products.

Archive workflow: every 5 minutes; retains 3 hours but processes only one 30-minute oldest-missing chunk per run.

RAP: hourly/cached by the existing historical phase helper.

Dual-pol: remains in the live scientific pipeline every 5 minutes; historical dual-pol/fusion persistence is not silently fabricated by this change.

MRMS history: full-CONUS QC reflectivity plus per-scan phase and precipitation type.

GitHub Releases: persistent archive storage.

GitHub Pages: same-origin staged history for the browser.

Important fixes

Live Web Mercator projection now runs after all fusion/diagnostic source images exist.

Live installs scipy, required by download_rap_profile.py.

Archive honors the live completion watermark and does not compete for the newest scan.

Archive works oldest-first through the 3-hour retention window in 30-minute chunks.

Archive persists mrms_history.json and mrms_archive_status.json into the daily Release; the previous runner-local manifest would disappear when the Actions runner ended.

Exact per-scan phase and precipitation-type products are retained instead of creating an additional current snapshot every archive run.

Archive has the keep_grib parameter required by the phase helper handoff.
