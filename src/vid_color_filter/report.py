"""HTML report generator with annotation JS for calibration.

Generates self-contained HTML pages (no external CSS/JS dependencies)
with dark-themed UI and localStorage-based annotation support.
"""

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def generate_pair_report(
    scores: Dict[str, Any],
    pair_dir: Union[str, Path],
    n_frames: int = 5,
) -> None:
    """Create report.html in pair_dir for a single video pair.

    Parameters
    ----------
    scores : dict
        Must contain: video_pair_id, global_shift_score, local_diff_score,
        temporal_instability, mask_coverage_ratio.
    pair_dir : str or Path
        Directory where report.html will be written.
    n_frames : int
        Number of sampled frames to display in the comparison tabs.
    """
    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)

    vid_id = scores["video_pair_id"]
    global_shift = scores["global_shift_score"]
    local_diff = scores["local_diff_score"]
    temporal = scores["temporal_instability"]
    mask_cov = scores["mask_coverage_ratio"]

    ls_key = f"calibration_annotation_{vid_id}"

    # Build frame tabs
    frame_buttons = []
    frame_panels = []
    for i in range(n_frames):
        active_btn = ' class="tab-btn active"' if i == 0 else ' class="tab-btn"'
        display = "block" if i == 0 else "none"
        frame_buttons.append(
            f'<button{active_btn} onclick="showFrame({i})">Frame {i}</button>'
        )
        frame_panels.append(f"""<div class="frame-panel" id="frame-{i}" style="display:{display}">
  <div class="frame-row">
    <div class="frame-col"><h4>Source</h4><img src="src_frame_{i}.png" alt="Source frame {i}"></div>
    <div class="frame-col"><h4>Filtered</h4><img src="filt_frame_{i}.png" alt="Filtered frame {i}"></div>
  </div>
</div>""")

    frame_buttons_html = "\n".join(frame_buttons)
    frame_panels_html = "\n".join(frame_panels)

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Report: {html.escape(vid_id)}</title>
<style>
body {{
  background: #1a1a2e;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0; padding: 20px;
  line-height: 1.6;
}}
a {{ color: #5dade2; }}
h1, h2, h3 {{ color: #f0f0f0; }}
.back-link {{ margin-bottom: 20px; display: inline-block; }}
table {{
  border-collapse: collapse; width: 100%; margin: 16px 0;
}}
th, td {{
  border: 1px solid #333; padding: 10px 14px; text-align: left;
}}
th {{ background: #16213e; }}
tr:nth-child(even) {{ background: #16213e; }}
img {{ max-width: 100%; height: auto; border: 1px solid #333; }}
.frame-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.frame-col {{ flex: 1; min-width: 300px; }}
.tab-bar {{ display: flex; gap: 4px; margin: 12px 0; flex-wrap: wrap; }}
.tab-btn {{
  background: #16213e; color: #e0e0e0; border: 1px solid #333;
  padding: 8px 16px; cursor: pointer; border-radius: 4px 4px 0 0;
}}
.tab-btn.active {{ background: #0f3460; border-bottom-color: #0f3460; }}
.section {{ margin: 32px 0; }}
.heatmap-section img, .mask-section img {{ max-width: 600px; }}
.temporal-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.temporal-col {{ flex: 1; min-width: 300px; text-align: center; }}
.annotation-bar {{
  position: sticky; top: 0; background: #0f3460; padding: 12px 20px;
  display: flex; align-items: center; gap: 16px; z-index: 100;
  border-bottom: 2px solid #333; margin: -20px -20px 20px -20px;
}}
.annotation-bar button {{
  padding: 8px 24px; border: none; border-radius: 4px;
  font-size: 16px; cursor: pointer; font-weight: bold;
}}
.btn-pass {{ background: #27ae60; color: #fff; }}
.btn-pass.selected {{ box-shadow: 0 0 0 3px #2ecc71; }}
.btn-fail {{ background: #c0392b; color: #fff; }}
.btn-fail.selected {{ box-shadow: 0 0 0 3px #e74c3c; }}
#annotation-status {{ font-style: italic; color: #aaa; }}
</style>
</head>
<body>

<div class="annotation-bar">
  <a class="back-link" href="../index.html">&larr; Back to Index</a>
  <button class="btn-pass" onclick="annotate('pass')">Pass</button>
  <button class="btn-fail" onclick="annotate('fail')">Fail</button>
  <span id="annotation-status">Not annotated</span>
</div>

<h1>Video Pair: {html.escape(vid_id)}</h1>

<h2>Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Global Shift Score</td><td>{global_shift}</td></tr>
  <tr><td>Local Diff Score</td><td>{local_diff}</td></tr>
  <tr><td>Temporal Instability</td><td>{temporal}</td></tr>
  <tr><td>Mask Coverage Ratio</td><td>{mask_cov}</td></tr>
</table>

<div class="section">
  <h2>Frame Comparison</h2>
  <div class="tab-bar">
    {frame_buttons_html}
  </div>
  {frame_panels_html}
</div>

<div class="section heatmap-section">
  <h2>Heatmap</h2>
  <img src="heatmap.png" alt="Color difference heatmap">
</div>

<div class="section mask-section">
  <h2>Mask</h2>
  <img src="mask.png" alt="Detected mask">
</div>

<div class="section">
  <h2>Temporal Maps</h2>
  <div class="temporal-row">
    <div class="temporal-col">
      <h3>Median</h3>
      <img src="temporal_median.png" alt="Temporal median map">
    </div>
    <div class="temporal-col">
      <h3>IQR</h3>
      <img src="temporal_iqr.png" alt="Temporal IQR map">
    </div>
  </div>
</div>

<script>
(function() {{
  var LS_KEY = {json.dumps(ls_key)};

  function showFrame(idx) {{
    var panels = document.querySelectorAll('.frame-panel');
    var btns = document.querySelectorAll('.tab-btn');
    for (var i = 0; i < panels.length; i++) {{
      panels[i].style.display = (i === idx) ? 'block' : 'none';
      btns[i].classList.toggle('active', i === idx);
    }}
  }}
  window.showFrame = showFrame;

  function annotate(value) {{
    localStorage.setItem(LS_KEY, value);
    updateStatus();
  }}
  window.annotate = annotate;

  function updateStatus() {{
    var val = localStorage.getItem(LS_KEY);
    var el = document.getElementById('annotation-status');
    if (val) {{
      el.textContent = 'Annotated: ' + val;
    }} else {{
      el.textContent = 'Not annotated';
    }}
    document.querySelector('.btn-pass').classList.toggle('selected', val === 'pass');
    document.querySelector('.btn-fail').classList.toggle('selected', val === 'fail');
  }}

  updateStatus();
}})();
</script>

</body>
</html>"""

    (pair_dir / "report.html").write_text(report_html, encoding="utf-8")


def generate_index_page(
    cases: List[Dict[str, Any]],
    reports_dir: Union[str, Path],
) -> None:
    """Create index.html in reports_dir listing all video pairs.

    Parameters
    ----------
    cases : list of dict
        Each dict has: video_pair_id, global_shift_score, local_diff_score,
        temporal_instability.
    reports_dir : str or Path
        Directory where index.html will be written.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Sort by global_shift_score descending
    sorted_cases = sorted(cases, key=lambda c: c["global_shift_score"], reverse=True)

    # Build table rows
    rows = []
    for c in sorted_cases:
        vid_id = html.escape(str(c["video_pair_id"]))
        rows.append(
            f'<tr data-id="{vid_id}">'
            f'<td><a href="{vid_id}/report.html">{vid_id}</a></td>'
            f'<td>{c["global_shift_score"]}</td>'
            f'<td>{c["local_diff_score"]}</td>'
            f'<td>{c["temporal_instability"]}</td>'
            f'<td class="anno-cell">-</td>'
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    # JSON list of video pair IDs for JS
    vid_ids_json = json.dumps([c["video_pair_id"] for c in sorted_cases])

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calibration Index</title>
<style>
body {{
  background: #1a1a2e;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0; padding: 20px;
  line-height: 1.6;
}}
a {{ color: #5dade2; }}
h1 {{ color: #f0f0f0; }}
table {{
  border-collapse: collapse; width: 100%; margin: 16px 0;
}}
th, td {{
  border: 1px solid #333; padding: 10px 14px; text-align: left;
}}
th {{
  background: #16213e; cursor: pointer; user-select: none;
}}
th:hover {{ background: #0f3460; }}
tr:nth-child(even) {{ background: #16213e; }}
.toolbar {{
  display: flex; align-items: center; gap: 16px; margin: 16px 0;
  flex-wrap: wrap;
}}
.progress {{ font-size: 18px; }}
.btn-export {{
  background: #2980b9; color: #fff; border: none; padding: 10px 20px;
  border-radius: 4px; cursor: pointer; font-size: 14px;
}}
.btn-export:hover {{ background: #3498db; }}
.anno-pass {{ color: #2ecc71; font-weight: bold; }}
.anno-fail {{ color: #e74c3c; font-weight: bold; }}
</style>
</head>
<body>

<h1>Calibration Reports</h1>

<div class="toolbar">
  <span class="progress" id="progress">Annotated: 0 / {len(sorted_cases)}</span>
  <button class="btn-export" onclick="exportAnnotations()">Export annotations.json</button>
</div>

<table id="cases-table">
  <thead>
    <tr>
      <th data-col="0" data-type="string" onclick="sortTable(0, 'string')">Video Pair ID</th>
      <th data-col="1" data-type="number" onclick="sortTable(1, 'number')">Global Shift Score</th>
      <th data-col="2" data-type="number" onclick="sortTable(2, 'number')">Local Diff Score</th>
      <th data-col="3" data-type="number" onclick="sortTable(3, 'number')">Temporal Instability</th>
      <th data-col="4" data-type="string" onclick="sortTable(4, 'string')">Annotation</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<script>
(function() {{
  var VIDEO_IDS = {vid_ids_json};
  var sortDir = {{}};

  function refreshAnnotations() {{
    var annotated = 0;
    var rows = document.querySelectorAll('#cases-table tbody tr');
    rows.forEach(function(row) {{
      var vid = row.getAttribute('data-id');
      var key = 'calibration_annotation_' + vid;
      var val = localStorage.getItem(key);
      var cell = row.querySelector('.anno-cell');
      if (val === 'pass') {{
        cell.textContent = 'pass';
        cell.className = 'anno-cell anno-pass';
        annotated++;
      }} else if (val === 'fail') {{
        cell.textContent = 'fail';
        cell.className = 'anno-cell anno-fail';
        annotated++;
      }} else {{
        cell.textContent = '-';
        cell.className = 'anno-cell';
      }}
    }});
    document.getElementById('progress').textContent =
      'Annotated: ' + annotated + ' / ' + VIDEO_IDS.length;
  }}

  function sortTable(colIdx, colType) {{
    var table = document.getElementById('cases-table');
    var tbody = table.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var dir = sortDir[colIdx] === 'asc' ? 'desc' : 'asc';
    sortDir[colIdx] = dir;

    rows.sort(function(a, b) {{
      var aVal = a.cells[colIdx].textContent.trim();
      var bVal = b.cells[colIdx].textContent.trim();
      if (colType === 'number') {{
        aVal = parseFloat(aVal) || 0;
        bVal = parseFloat(bVal) || 0;
      }}
      if (aVal < bVal) return dir === 'asc' ? -1 : 1;
      if (aVal > bVal) return dir === 'asc' ? 1 : -1;
      return 0;
    }});

    rows.forEach(function(row) {{ tbody.appendChild(row); }});
  }}
  window.sortTable = sortTable;

  function exportAnnotations() {{
    var data = {{}};
    VIDEO_IDS.forEach(function(vid) {{
      var key = 'calibration_annotation_' + vid;
      var val = localStorage.getItem(key);
      if (val) data[vid] = val;
    }});
    var blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.json';
    a.click();
    URL.revokeObjectURL(url);
  }}
  window.exportAnnotations = exportAnnotations;

  document.addEventListener('visibilitychange', function() {{
    if (!document.hidden) refreshAnnotations();
  }});

  refreshAnnotations();
}})();
</script>

</body>
</html>"""

    (reports_dir / "index.html").write_text(index_html, encoding="utf-8")


def generate_error_report(
    video_pair_id: str,
    error_message: str,
    pair_dir: Union[str, Path],
) -> None:
    """Create a placeholder report.html for a failed visualization.

    Parameters
    ----------
    video_pair_id : str
        Identifier for the video pair.
    error_message : str
        Error message to display.
    pair_dir : str or Path
        Directory where report.html will be written.
    """
    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)

    safe_id = html.escape(video_pair_id)
    safe_msg = html.escape(error_message)

    error_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Error: {safe_id}</title>
<style>
body {{
  background: #1a1a2e;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0; padding: 20px;
  line-height: 1.6;
}}
a {{ color: #5dade2; }}
h1 {{ color: #f0f0f0; }}
.error-box {{
  background: #3b1111;
  border: 2px solid #e74c3c;
  color: #ff6b6b;
  padding: 20px;
  border-radius: 8px;
  margin: 24px 0;
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-word;
}}
.back-link {{ margin-bottom: 20px; display: inline-block; }}
</style>
</head>
<body>

<a class="back-link" href="../index.html">&larr; Back to Index</a>

<h1>Error: {safe_id}</h1>

<p>Visualization generation failed for this video pair.</p>

<div class="error-box">{safe_msg}</div>

</body>
</html>"""

    (pair_dir / "report.html").write_text(error_html, encoding="utf-8")
