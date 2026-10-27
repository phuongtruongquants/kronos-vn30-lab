"""Interactive price and volume chart for independent Kronos sample paths."""

from __future__ import annotations

import json

import pandas as pd


COLORS = ["#e24e45", "#9b5de5", "#f49b3f", "#09a6a6", "#d15c90", "#5968d9", "#9b7d1f"]


def chart_html(history: pd.DataFrame, forecasts: pd.DataFrame, actual: pd.DataFrame) -> str:
    visible_history = history.tail(90)
    actual_by_date = {str(row.date): row for row in actual.itertuples()}
    paths = []
    for index, (seed, frame) in enumerate(forecasts.groupby("seed", sort=True)):
        frame = frame.sort_values("date")
        paths.append({
            "seed": int(seed), "color": COLORS[index % len(COLORS)],
            "points": [{"date": str(row.date), "o": float(row.open), "h": float(row.high),
                        "l": float(row.low), "c": float(row.close), "v": max(0, float(row.volume))}
                       for row in frame.itertuples()],
        })
    future_dates = [point["date"] for point in paths[0]["points"]]
    rows = [
        {"date": str(row.date), "o": float(row.open), "h": float(row.high),
         "l": float(row.low), "c": float(row.close), "v": float(row.volume), "kind": "known"}
        for row in visible_history.itertuples()
    ]
    for date in future_dates:
        actual_row = actual_by_date.get(date)
        rows.append({"date": date, "kind": "future", "actual": None if actual_row is None else {
            "o": float(actual_row.open), "h": float(actual_row.high),
            "l": float(actual_row.low), "c": float(actual_row.close),
            "v": float(actual_row.volume),
        }})
    dataset = json.dumps({"rows": rows, "paths": paths, "cutoff": len(visible_history) - 1}, ensure_ascii=False)
    template = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;background:transparent;color:#18352c;font:13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.card{background:white;border:1px solid #dce6df;border-radius:18px;padding:15px 18px 11px;overflow:hidden}
.head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
.title{font-size:16px;font-weight:720}.hint{font-size:12px;color:#647b71;margin-top:2px}
.controls{display:flex;gap:5px}button{font:inherit;background:#f2f7f4;border:1px solid #d8e5dd;border-radius:8px;color:#214637;min-width:35px;height:32px;cursor:pointer}button:hover{background:#e4f1e9}button:focus-visible{outline:2px solid #138769;outline-offset:2px}#reset,#view{padding:0 10px}
.wrap{height:548px;position:relative}canvas{display:block;width:100%;height:100%;cursor:crosshair;touch-action:none}
#tip{display:none;position:absolute;z-index:2;pointer-events:none;background:#142a25f2;color:white;border-radius:9px;padding:9px 11px;line-height:1.55;box-shadow:0 6px 22px #123b2c35;white-space:nowrap;font-size:12px}
.bottom{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding:4px;color:#61766c;font-size:12px}.legend{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.sw{display:inline-block;width:17px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}#range{font-variant-numeric:tabular-nums}
</style></head><body><div class="card">
<div class="head"><div><div class="title">Giá đóng cửa và khối lượng · nhiều đường Kronos</div><div class="hint">Lăn chuột để phóng to · kéo để di chuyển · trục tung tự đổi theo vùng xem</div></div><div class="controls"><button id="minus" title="Thu nhỏ">−</button><button id="plus" title="Phóng to">+</button><button id="reset">Mặc định</button><button id="view">Nến</button></div></div>
<div class="wrap"><canvas id="chart"></canvas><div id="tip"></div></div>
<div class="bottom"><div class="legend" id="legend"></div><span id="range"></span></div>
</div><script>
const data=__DATA__, rows=data.rows, paths=data.paths, cutoff=data.cutoff;
const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip'),range=document.getElementById('range');
const legend=document.getElementById('legend');
legend.innerHTML='<span><i class="sw" style="background:#2858c8"></i>Giá/khối lượng thực tế</span>'+paths.map(p=>'<span><i class="sw" style="background:'+p.color+'"></i>Seed '+p.seed+'</span>').join('')+'<span>Vùng tô = khoảng các mẫu, không phải độ tin cậy</span>';
let count=Math.min(rows.length,45),start=Math.max(0,rows.length-count),candleMode=false,dragging=false,dragX=0,dragStart=0,hover=-1;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));const money=x=>Number(x).toFixed(2);const compact=x=>x>=1e6?(x/1e6).toFixed(1)+'tr':x>=1e3?(x/1e3).toFixed(0)+'k':Math.round(x).toString();
function geom(){const w=canvas.clientWidth,h=canvas.clientHeight,left=75,right=24;return {w,h,left,right,pw:w-left-right,pt:27,pb:352,vt:396,vb:h-35};}
function x(i,g){return g.left+(i-start+.5)*g.pw/count;}
function size(){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.floor(rect.width*dpr));canvas.height=Math.max(1,Math.floor(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function niceStep(span){const raw=span/5,pow=Math.pow(10,Math.floor(Math.log10(raw)));for(const k of [1,2,5,10])if(k*pow>=raw)return k*pow;return pow*10;}
function scales(){const end=Math.min(rows.length,start+count),prices=[],volumes=[];
 for(let i=start;i<end;i++){const r=rows[i],a=r.kind==='known'?r:r.actual;if(a){prices.push(a.o,a.h,a.l,a.c);volumes.push(a.v);}if(i>cutoff)for(const p of paths){const v=p.points[i-cutoff-1];if(v){prices.push(v.o,v.h,v.l,v.c);volumes.push(v.v);}}}
 let lo=Math.min(...prices),hi=Math.max(...prices);if(!isFinite(lo)||!isFinite(hi)){lo=0;hi=1;}const pad=Math.max((hi-lo)*.07,hi*.002,.05);return {lo:lo-pad,hi:hi+pad,vmax:Math.max(1,...volumes)*1.13,end};}
function draw(){const g=geom(),s=scales();ctx.clearRect(0,0,g.w,g.h);ctx.fillStyle='#fff';ctx.fillRect(0,0,g.w,g.h);
 const yp=v=>g.pt+(s.hi-v)*(g.pb-g.pt)/(s.hi-s.lo),yv=v=>g.vb-Math.max(0,v)/s.vmax*(g.vb-g.vt);
 ctx.font='12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.textBaseline='middle';
 const step=niceStep(s.hi-s.lo);for(let t=Math.ceil(s.lo/step)*step;t<=s.hi;t+=step){const yy=yp(t);ctx.strokeStyle='#e1e9e5';ctx.lineWidth=1;ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(g.left,yy);ctx.lineTo(g.w-g.right,yy);ctx.stroke();ctx.fillStyle='#536b60';ctx.textAlign='right';ctx.fillText(money(t),g.left-10,yy);}
 ctx.fillStyle='#536b60';ctx.textAlign='left';ctx.fillText('Giá · nghìn VND',g.left,11);ctx.fillText('Khối lượng',g.left,g.vt-15);
 for(let t=0;t<s.vmax;t+=niceStep(s.vmax)){const yy=yv(t);ctx.strokeStyle='#edf1ee';ctx.beginPath();ctx.moveTo(g.left,yy);ctx.lineTo(g.w-g.right,yy);ctx.stroke();ctx.fillStyle='#72857b';ctx.textAlign='right';ctx.fillText(compact(t),g.left-10,yy);}
 const every=Math.max(1,Math.ceil(count/8));for(let i=start;i<s.end;i++)if((i-start)%every===0||i===s.end-1){ctx.fillStyle='#657d70';ctx.textAlign='center';ctx.fillText(rows[i].date.slice(5),x(i,g),g.h-17);}
 if(cutoff>=start&&cutoff<s.end){const bx=x(cutoff,g)+g.pw/count*.5;ctx.strokeStyle='#8ca299';ctx.lineWidth=1.3;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(bx,g.pt);ctx.lineTo(bx,g.vb);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#64796f';ctx.textAlign='left';ctx.fillText('Ngày chốt',Math.min(bx+5,g.w-100),g.pt+11);}
 // Observed volume bars and prices.
 const bw=Math.max(2,Math.min(11,g.pw/count*.65));for(let i=start;i<s.end;i++){const r=rows[i],a=r.kind==='known'?r:r.actual;if(!a)continue;const xx=x(i,g);ctx.fillStyle=r.kind==='known'?'#446bdd75':'#2858c8aa';ctx.fillRect(xx-bw/2,yv(a.v),bw,g.vb-yv(a.v));}
 if(candleMode){for(let i=start;i<s.end;i++){const r=rows[i],a=r.kind==='known'?r:r.actual;if(!a)continue;const xx=x(i,g),up=a.c>=a.o;ctx.strokeStyle=up?'#2d69c8':'#6b8fd9';ctx.fillStyle=up?'#8bafe1':'#d2ddf2';ctx.lineWidth=1.1;ctx.beginPath();ctx.moveTo(xx,yp(a.h));ctx.lineTo(xx,yp(a.l));ctx.stroke();const top=Math.min(yp(a.o),yp(a.c));ctx.fillRect(xx-bw/2,top,bw,Math.max(2,Math.abs(yp(a.o)-yp(a.c))));}}
 else{ctx.strokeStyle='#2858c8';ctx.lineWidth=2.35;ctx.setLineDash([]);ctx.beginPath();let started=false;for(let i=start;i<s.end;i++){const r=rows[i],a=r.kind==='known'?r:r.actual;if(!a)continue;if(!started){ctx.moveTo(x(i,g),yp(a.c));started=true;}else ctx.lineTo(x(i,g),yp(a.c));}ctx.stroke();}
 // Sample spread only, without probabilistic interpretation.
 if(paths.length>1&&s.end>cutoff+1){const first=Math.max(start,cutoff+1),last=s.end-1;ctx.beginPath();let begun=false;for(let i=first;i<=last;i++){const vals=paths.map(p=>p.points[i-cutoff-1]?.c).filter(Number.isFinite);if(!vals.length)continue;const yy=yp(Math.max(...vals));if(!begun){ctx.moveTo(x(i,g),yy);begun=true;}else ctx.lineTo(x(i,g),yy);}for(let i=last;i>=first;i--){const vals=paths.map(p=>p.points[i-cutoff-1]?.c).filter(Number.isFinite);if(vals.length)ctx.lineTo(x(i,g),yp(Math.min(...vals)));}ctx.closePath();ctx.fillStyle='#df6b5620';ctx.fill();}
 for(const [j,p] of paths.entries()){ctx.strokeStyle=p.color;ctx.lineWidth=j===0?2.6:1.7;ctx.setLineDash(j===0?[7,4]:[4,4]);ctx.beginPath();let begun=false;for(let i=Math.max(start,cutoff);i<s.end;i++){const val=i===cutoff?rows[i].c:p.points[i-cutoff-1]?.c;if(!Number.isFinite(val))continue;if(!begun){ctx.moveTo(x(i,g),yp(val));begun=true;}else ctx.lineTo(x(i,g),yp(val));}ctx.stroke();ctx.beginPath();begun=false;for(let i=Math.max(start,cutoff+1);i<s.end;i++){const v=p.points[i-cutoff-1]?.v;if(!Number.isFinite(v))continue;if(!begun){ctx.moveTo(x(i,g),yv(v));begun=true;}else ctx.lineTo(x(i,g),yv(v));}ctx.stroke();ctx.setLineDash([]);}
 if(hover>=start&&hover<s.end){ctx.strokeStyle='#9db1aa';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x(hover,g),g.pt);ctx.lineTo(x(hover,g),g.vb);ctx.stroke();ctx.setLineDash([]);}
 range.textContent=rows[start].date+' → '+rows[s.end-1].date;
}
function zoom(factor,anchor=1){const next=clamp(Math.round(count*factor),12,rows.length),target=start+count*anchor;start=clamp(Math.round(target-next*anchor),0,rows.length-next);count=next;draw();}
function indexAt(clientX){const r=canvas.getBoundingClientRect(),g=geom();return clamp(Math.floor(start+(clientX-r.left-g.left)/g.pw*count),start,Math.min(rows.length-1,start+count-1));}
canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),g=geom();zoom(e.deltaY<0?.8:1.25,clamp((e.clientX-r.left-g.left)/g.pw,0,1));},{passive:false});
canvas.addEventListener('pointerdown',e=>{dragging=true;dragX=e.clientX;dragStart=start;canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';});
canvas.addEventListener('pointerup',()=>{dragging=false;canvas.style.cursor='crosshair';});
canvas.addEventListener('pointermove',e=>{if(dragging){const g=geom();start=clamp(Math.round(dragStart-(e.clientX-dragX)/g.pw*count),0,rows.length-count);draw();return;}hover=indexAt(e.clientX);const r=rows[hover],a=r.kind==='known'?r:r.actual,rect=canvas.getBoundingClientRect();let html='<b>'+r.date+'</b>';if(a)html+='<br>Thực tế: '+money(a.c)+' · KL '+compact(a.v);if(hover>cutoff)for(const p of paths){const v=p.points[hover-cutoff-1];if(v)html+='<br><span style="color:'+p.color+'">Seed '+p.seed+': '+money(v.c)+' · KL '+compact(v.v)+'</span>';}tip.innerHTML=html;tip.style.display='block';tip.style.left=clamp(e.clientX-rect.left+15,80,Math.max(80,rect.width-210))+'px';tip.style.top=clamp(e.clientY-rect.top+12,8,Math.max(8,rect.height-140))+'px';draw();});
canvas.addEventListener('pointerleave',()=>{if(!dragging){hover=-1;tip.style.display='none';draw();}});
document.getElementById('plus').onclick=()=>zoom(.8);document.getElementById('minus').onclick=()=>zoom(1.25);document.getElementById('reset').onclick=()=>{count=Math.min(rows.length,45);start=Math.max(0,rows.length-count);draw();};document.getElementById('view').onclick=e=>{candleMode=!candleMode;e.target.textContent=candleMode?'Đường':'Nến';draw();};
new ResizeObserver(size).observe(canvas);size();
</script></body></html>"""
    return template.replace("__DATA__", dataset)
