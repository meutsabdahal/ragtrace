from __future__ import annotations
import json
from dataclasses import asdict
from ragtrace.session import TraceSession


def _session_to_dict(session: TraceSession) -> dict:
    return {
        "session_id": session.session_id,
        "query": session.query,
        "total_latency_ms": round(session.total_latency_ms, 1),
        "retrieval_spans": [
            {
                "query": s.query,
                "chunks": s.chunks,
                "scores": [round(sc, 4) for sc in s.scores],
                "k_requested": s.k_requested,
                "k_returned": s.k_returned,
                "latency_ms": round(s.latency_ms, 1),
                "diagnosis": s.diagnosis,
            }
            for s in session.retrieval_spans
        ],
        "generation_spans": [
            {
                "prompt": s.prompt,
                "response": s.response,
                "model": s.model,
                "prompt_tokens": s.prompt_tokens,
                "response_tokens": s.response_tokens,
                "latency_ms": round(s.latency_ms, 1),
                "diagnosis": s.diagnosis,
            }
            for s in session.generation_spans
        ],
    }


def render_html(session: TraceSession) -> str:
    data = json.dumps(_session_to_dict(session), indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Trace — {session.query[:60]}</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#1a1a1a;background:#fff}}
  h1{{font-size:18px;font-weight:600}}
  .query{{background:#f5f5f5;border-radius:8px;padding:10px 14px;font-size:14px;margin:12px 0 24px}}
  .section{{border:1px solid #e5e5e5;border-radius:8px;margin-bottom:16px;overflow:hidden}}
  .section-header{{background:#fafafa;border-bottom:1px solid #e5e5e5;padding:10px 16px;font-size:13px;font-weight:600;display:flex;gap:12px;align-items:center}}
  .latency{{font-size:11px;color:#999;font-weight:400}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{text-align:left;padding:8px 12px;border-bottom:1px solid #e5e5e5;font-weight:500;color:#666}}
  td{{padding:8px 12px;border-bottom:1px solid #f0f0f0;vertical-align:top}}
  .score-high{{color:#166534;font-weight:500}}
  .score-mid{{color:#92400e;font-weight:500}}
  .score-low{{color:#991b1b;font-weight:500}}
  .diagnosis{{padding:10px 16px;font-size:12px;color:#555;border-top:1px solid #f0f0f0}}
  .warn{{color:#92400e}}
  .ok{{color:#166534}}
  .response{{padding:12px 16px;font-size:13px;line-height:1.6;white-space:pre-wrap}}
  .meta{{padding:8px 16px;font-size:11px;color:#999;border-bottom:1px solid #f0f0f0}}
</style>
</head>
<body>
<h1>RAG Trace</h1>
<div class="query"><strong>Query:</strong> {session.query}</div>
<div id="root"></div>
<script>
const data = {data};

function score_class(s) {{
  if (s >= 0.7) return 'score-high';
  if (s >= 0.5) return 'score-mid';
  return 'score-low';
}}

function score_signal(s) {{
  if (s >= 0.7) return '✓';
  if (s >= 0.5) return '⚠';
  return '✗';
}}

const root = document.getElementById('root');

data.retrieval_spans.forEach((span, i) => {{
  const sec = document.createElement('div');
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-header">
      Retrieval ${{i + 1}}
      <span class="latency">${{span.latency_ms}}ms · k=${{span.k_returned}}/${{span.k_requested}}</span>
    </div>
    <table>
      <tr><th>Chunk</th><th>Score</th><th></th></tr>
      ${{span.chunks.map((c, j) => `
        <tr>
          <td>${{c.substring(0, 120)}}${{c.length > 120 ? '...' : ''}}</td>
          <td class="${{score_class(span.scores[j])}}">${{span.scores[j].toFixed(2)}}</td>
          <td class="${{score_class(span.scores[j])}}">${{score_signal(span.scores[j])}}</td>
        </tr>
      `).join('')}}
    </table>
    <div class="diagnosis">
      ${{span.diagnosis.map(d => `<div class="${{d.includes('healthy') ? 'ok' : 'warn'}}">${{d}}</div>`).join('')}}
    </div>
  `;
  root.appendChild(sec);
}});

data.generation_spans.forEach((span, i) => {{
  const sec = document.createElement('div');
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-header">
      Generation ${{i + 1}}
      <span class="latency">${{span.latency_ms}}ms · model=${{span.model}}</span>
    </div>
    <div class="meta">Prompt tokens: ${{span.prompt_tokens}} · Response tokens: ${{span.response_tokens}}</div>
    <div class="response">${{span.response}}</div>
    <div class="diagnosis">
      ${{span.diagnosis.map(d => `<div class="${{d.includes('healthy') ? 'ok' : 'warn'}}">${{d}}</div>`).join('')}}
    </div>
  `;
  root.appendChild(sec);
}});

const footer = document.createElement('div');
footer.style = 'font-size:11px;color:#aaa;margin-top:16px';
footer.textContent = `Total latency: ${{data.total_latency_ms}}ms · Session: ${{data.session_id}}`;
root.appendChild(footer);
</script>
</body>
</html>"""
