"""Self-contained interactive candle chart for the local Streamlit app."""

from __future__ import annotations

import json

import pandas as pd


def chart_html(history: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame) -> str:
    visible_history = history.tail(90)
    actual_close = dict(zip(actual["date"], actual["close"]))
    rows = [
        {
            "date": str(row.date), "kind": "known",
            "o": float(row.open), "h": float(row.high),
            "l": float(row.low), "c": float(row.close),
        }
        for row in visible_history.itertuples()
    ]
    rows += [
        {
            "date": str(row.date), "kind": "forecast",
            "o": float(row.open), "h": float(row.high),
            "l": float(row.low), "c": float(row.close),
            "actual": float(actual_close[row.date]) if row.date in actual_close else None,
        }
        for row in forecast.itertuples()
    ]
    dataset = json.dumps({"rows": rows, "cutoffIndex": len(visible_history) - 1}, ensure_ascii=False)
    template = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8" />
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: transparent; color: #17372d; font: 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .card { background: #fff; border: 1px solid #dae5df; border-radius: 18px; padding: 16px 18px 10px; overflow: hidden; }
  .top { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
  .title { font-size: 16px; font-weight: 720; letter-spacing: -.02em; }
  .hint { color: #647b71; font-size: 12px; margin-top: 2px; }
  .controls { display: flex; gap: 5px; align-items: center; }
  button { font: inherit; background: #f2f7f4; border: 1px solid #d8e5dd; border-radius: 8px; color: #214637; min-width: 35px; height: 32px; cursor: pointer; }
  button:hover { background: #e3f1e9; border-color: #a8cdb9; }
  button:focus-visible { outline: 2px solid #138769; outline-offset: 2px; }
  #reset { padding: 0 10px; }
  #view { padding: 0 10px; }
  .chart-wrap { position: relative; height: 478px; }
  canvas { display: block; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }
  #tooltip { display: none; position: absolute; top: 18px; left: 70px; z-index: 3; pointer-events: none; background: rgba(15,38,31,.94); color: white; border-radius: 9px; padding: 9px 11px; line-height: 1.55; box-shadow: 0 6px 20px #123b2c25; white-space: nowrap; }
  .bottom { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; padding: 3px 4px 0; color: #5b7467; font-size: 12px; }
  .legend { display: flex; flex-wrap: wrap; gap: 15px; align-items: center; }
  .swatch { display: inline-block; width: 16px; height: 3px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }
  .dashed { background: repeating-linear-gradient(90deg,#586ab5 0 5px,transparent 5px 8px); }
  .solid-known { background: #138769; } .solid-actual { background: #e88643; }
  #range { color: #46655a; font-variant-numeric: tabular-nums; }
</style></head><body>
<div class="card">
  <div class="top">
    <div><div class="title">Đường giá và nến dự báo</div><div class="hint">Lăn chuột để phóng to · kéo để di chuyển · trục giá tự co theo vùng đang xem</div></div>
    <div class="controls"><button id="zoomOut" title="Thu nhỏ">−</button><button id="zoomIn" title="Phóng to">+</button><button id="reset">Mặc định</button><button id="view">Nến</button></div>
  </div>
  <div class="chart-wrap"><canvas id="chart"></canvas><div id="tooltip"></div></div>
  <div class="bottom"><div class="legend"><span><i class="swatch solid-known"></i>Nến đã biết</span><span><i class="swatch dashed"></i>Kronos giả định</span><span><i class="swatch solid-actual"></i>Đóng cửa thực tế</span></div><span id="range"></span></div>
</div>
<script>
const data = __DATA__;
const rows = data.rows, cutoff = data.cutoffIndex;
const canvas = document.getElementById('chart'), ctx = canvas.getContext('2d');
const tip = document.getElementById('tooltip'), range = document.getElementById('range');
let count = Math.min(rows.length, 55), start = Math.max(0, rows.length-count), candleMode = true;
let dragging = false, dragX = 0, dragStart = 0, hover = -1;
const pad = {left: 72, right: 20, top: 20, bottom: 38};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const money = v => Number(v).toFixed(2);

function size() {
  const box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(box.width*dpr));
  canvas.height = Math.max(1, Math.floor(box.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
function geometry() {
  const w=canvas.clientWidth, h=canvas.clientHeight;
  return {w,h,pw:w-pad.left-pad.right,ph:h-pad.top-pad.bottom};
}
function visibleRange() {
  const end=Math.min(rows.length, start+count), values=[];
  for(let i=start;i<end;i++) {
    const r=rows[i]; values.push(r.o,r.h,r.l,r.c);
    if(r.actual!==null && r.actual!==undefined) values.push(r.actual);
  }
  let lo=Math.min(...values), hi=Math.max(...values);
  if(!isFinite(lo)||!isFinite(hi)) {lo=0;hi=1;}
  const p=Math.max((hi-lo)*.08, hi*.002, .05);
  return {lo:lo-p,hi:hi+p,end};
}
function niceStep(span) {
  const raw=span/5, pow=Math.pow(10,Math.floor(Math.log10(raw)));
  for(const factor of [1,2,5,10]) if(factor*pow>=raw) return factor*pow;
  return pow*10;
}
function draw() {
  const {w,h,pw,ph}=geometry(), {lo,hi,end}=visibleRange();
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#ffffff';ctx.fillRect(0,0,w,h);
  const x=i=>pad.left+(i-start+.5)*pw/count;
  const y=v=>pad.top+(hi-v)*ph/(hi-lo);
  ctx.font='12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
  ctx.textBaseline='middle';
  const step=niceStep(hi-lo);
  for(let tick=Math.ceil(lo/step)*step;tick<=hi;tick+=step) {
    const yy=y(tick); ctx.strokeStyle='#e4ece7';ctx.lineWidth=1;ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(pad.left,yy);ctx.lineTo(w-pad.right,yy);ctx.stroke();
    ctx.fillStyle='#587066';ctx.textAlign='right';ctx.fillText(money(tick),pad.left-10,yy);
  }
  ctx.fillStyle='#587066';ctx.textAlign='left';ctx.fillText('nghìn VND',pad.left,10);
  const labelEvery=Math.max(1,Math.ceil(count/7));
  for(let i=start;i<end;i++) {
    if((i-start)%labelEvery===0 || i===end-1) {
      ctx.fillStyle='#667b71';ctx.textAlign='center';
      ctx.fillText(rows[i].date.slice(5),x(i),h-18);
    }
  }
  if(cutoff>=start && cutoff<end) {
    const boundary=x(cutoff)+pw/count*.5;
    ctx.strokeStyle='#7c9488';ctx.lineWidth=1.5;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.moveTo(boundary,pad.top);ctx.lineTo(boundary,h-pad.bottom);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#496559';ctx.textAlign='left';ctx.fillText('Ngày chốt',Math.min(boundary+5,w-95),pad.top+12);
  }
  if(candleMode) {
    const body=Math.max(3,Math.min(13,pw/count*.62));
    for(let i=start;i<end;i++) {
      const r=rows[i], xx=x(i), up=r.c>=r.o, forecast=r.kind==='forecast';
      ctx.strokeStyle=forecast?'#586ab5':up?'#14876a':'#d37863';
      ctx.fillStyle=forecast?'rgba(88,106,181,.11)':up?'#83bba3':'#e8a999';
      ctx.lineWidth=forecast?1.6:1.2;ctx.setLineDash(forecast?[3,2]:[]);
      ctx.beginPath();ctx.moveTo(xx,y(r.h));ctx.lineTo(xx,y(r.l));ctx.stroke();
      const top=Math.min(y(r.o),y(r.c)), height=Math.max(2,Math.abs(y(r.o)-y(r.c)));
      ctx.fillRect(xx-body/2,top,body,height);ctx.strokeRect(xx-body/2,top,body,height);
    }
    ctx.setLineDash([]);
  } else {
    ctx.strokeStyle='#138769';ctx.lineWidth=2.2;ctx.setLineDash([]);ctx.beginPath();
    let begun=false;
    for(let i=start;i<end && i<=cutoff;i++) {const r=rows[i]; if(!begun){ctx.moveTo(x(i),y(r.c));begun=true;}else ctx.lineTo(x(i),y(r.c));}
    ctx.stroke();
  }
  // The model path is dashed even when hypothetical forecast candles are shown.
  ctx.strokeStyle='#586ab5';ctx.lineWidth=2.5;ctx.setLineDash([7,5]);ctx.beginPath();
  let started=false;
  for(let i=Math.max(start,cutoff);i<end;i++) {
    if(!started){ctx.moveTo(x(i),y(rows[i].c));started=true;}else ctx.lineTo(x(i),y(rows[i].c));
  }
  ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#e88643';ctx.lineWidth=2.8;ctx.beginPath();started=false;
  for(let i=Math.max(start,cutoff);i<end;i++) {
    const v=i===cutoff?rows[i].c:rows[i].actual;
    if(v===null||v===undefined) continue;
    if(!started){ctx.moveTo(x(i),y(v));started=true;}else ctx.lineTo(x(i),y(v));
  }
  ctx.stroke();
  if(hover>=start&&hover<end) {
    ctx.strokeStyle='#a4b9ad';ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(x(hover),pad.top);ctx.lineTo(x(hover),h-pad.bottom);ctx.stroke();ctx.setLineDash([]);
  }
  range.textContent=rows[start].date+' → '+rows[end-1].date;
}
function zoom(factor, anchor=.5) {
  const newCount=clamp(Math.round(count*factor),12,rows.length);
  const target=start+count*anchor;
  start=clamp(Math.round(target-newCount*anchor),0,rows.length-newCount);
  count=newCount;draw();
}
function pointerIndex(clientX) {
  const rect=canvas.getBoundingClientRect(), {pw}=geometry();
  return clamp(Math.floor(start+(clientX-rect.left-pad.left)/pw*count),start,Math.min(rows.length-1,start+count-1));
}
canvas.addEventListener('wheel',e=>{e.preventDefault();const rect=canvas.getBoundingClientRect(),{pw}=geometry();const anchor=clamp((e.clientX-rect.left-pad.left)/pw,0,1);zoom(e.deltaY < 0 ? .8 : 1.25,anchor);},{passive:false});
canvas.addEventListener('pointerdown',e=>{dragging=true;dragX=e.clientX;dragStart=start;canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';});
canvas.addEventListener('pointerup',e=>{dragging=false;canvas.style.cursor='crosshair';});
canvas.addEventListener('pointermove',e=>{
  if(dragging){const {pw}=geometry();start=clamp(Math.round(dragStart-(e.clientX-dragX)/pw*count),0,rows.length-count);draw();return;}
  hover=pointerIndex(e.clientX);const r=rows[hover],rect=canvas.getBoundingClientRect();
  tip.innerHTML='<b>'+r.date+'</b> · '+(r.kind==='known'?'Đã biết':'Kronos giả định')+'<br>Mở '+money(r.o)+' · Cao '+money(r.h)+'<br>Thấp '+money(r.l)+' · Đóng '+money(r.c)+(r.actual==null?'':'<br>Đóng thực tế '+money(r.actual));
  tip.style.display='block';tip.style.left=clamp(e.clientX-rect.left+14,74,Math.max(74,rect.width-230))+'px';draw();
});
canvas.addEventListener('pointerleave',()=>{if(!dragging){hover=-1;tip.style.display='none';draw();}});
document.getElementById('zoomIn').onclick=()=>zoom(.8);
document.getElementById('zoomOut').onclick=()=>zoom(1.25);
document.getElementById('reset').onclick=()=>{count=Math.min(rows.length,55);start=Math.max(0,rows.length-count);draw();};
document.getElementById('view').onclick=e=>{candleMode=!candleMode;e.target.textContent=candleMode?'Nến':'Đường';draw();};
new ResizeObserver(size).observe(canvas);size();
</script></body></html>"""
    return template.replace("__DATA__", dataset)
