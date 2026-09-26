WinterRadar 3-Hour Split Workflow

Files

.github/workflows/mrms_live.yml — current/live products every 5 minutes and Pages deployment.

.github/workflows/mrms_archive.yml — rolling 3-hour history repair every 5 minutes.

src/publish_live_history.py — stores the newest live radar/phase/precipitation-type WebPs in the daily GitHub Release and writes mrms_live_latest.json.

src/stage_history_for_pages.py — retrieves the release manifest and staged history WebPs for the Pages build.

src/archive_mrms_history.py — 3-hour rolling archive/backfill plus mrms_history.json and mrms_archive_status.json.

Data flow

Live workflow downloads and calculates the newest MRMS/RAP products.

Live publishes the newest timestamped browser rasters to the daily mrms-YYYYMMDD release.

Live writes mrms_live_latest.json as the completion watermark.

Archive scans the last 3 hours but will not process observations newer than that live watermark. This prevents a current scan from being downloaded twice by the two workflows.

Archive fills missing older radar timestamps and missing per-scan phase/precipitation-type products.

Archive writes mrms_history.json and mrms_archive_status.json back into the release.

The next live Pages build retrieves those release assets and stages them under site/history/.

GitHub Releases are persistent; the GitHub Actions runner filesystem is not.

Release storage

A 3-hour window is roughly 90 MRMS observations at a ~2-minute upstream cadence. With radar + phase + precipitation-type assets, this remains well below GitHub's 1,000-assets-per-release limit. GitHub currently allows up to 1,000 assets per release and each individual asset may be up to 2 GiB.
