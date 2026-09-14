/* ═══════════════════════════════════════════════════════════════
   AEROS — AISI Radial Gauge & Trend Sparkline
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  // Must match backend aisi_severity_category(): 0-2 / 2-5 / 5-8 / 8-10.
  // (Old frontend used 0-3 for green, which hid Mild 2-3 events.)
  // Neobrutalism palette on the light card: green / primary-yellow /
  // warning-amber / danger-red, ink needle + ticks.
  const AISI_ZONES = [
    { lo: 0,  hi: 2,  color: '#16A34A', label: 'Well-Mixed' },
    { lo: 2,  hi: 5,  color: '#FDC800', label: 'Mild Inversion' },
    { lo: 5,  hi: 8,  color: '#D97706', label: 'Moderate Inversion' },
    { lo: 8,  hi: 10, color: '#DC2626', label: 'Severe Inversion' },
  ];

  // Gauge geometry: 120° sweep centred on vertical
  const START_DEG = -60;
  const END_DEG = 60;
  const CX = 110;
  const CY = 118;
  const R = 82;

  function polar(deg) {
    const rad = deg * Math.PI / 180;
    return {
      x: CX + R * Math.sin(rad),
      y: CY - R * Math.cos(rad),
    };
  }

  function arcPath(lo, hi) {
    const p1 = polar(lo);
    const p2 = polar(hi);
    const large = (hi - lo) > 180 ? 1 : 0;
    return `M ${p1.x.toFixed(2)} ${p1.y.toFixed(2)} ` +
           `A ${R} ${R} 0 ${large} 1 ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }

  class AISIGauge {
    constructor(svgEl) {
      this.svg = svgEl;
      this.lastValue = 0;
      this._build();
    }

    _build() {
      const ns = 'http://www.w3.org/2000/svg';
      // Zone arcs
      for (const z of AISI_ZONES) {
        const loDeg = START_DEG + (z.lo / 10) * (END_DEG - START_DEG);
        const hiDeg = START_DEG + (z.hi / 10) * (END_DEG - START_DEG);
        const path = document.createElementNS(ns, 'path');
        path.setAttribute('d', arcPath(loDeg, hiDeg));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', z.color);
        path.setAttribute('stroke-width', '16');
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('opacity', '0.35');
        this.svg.appendChild(path);
      }
      // Tick marks 0..10
      for (let i = 0; i <= 10; i++) {
        const deg = START_DEG + (i / 10) * (END_DEG - START_DEG);
        const p = polar(deg);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', p.x); line.setAttribute('y1', p.y);
        line.setAttribute('x2', CX + (R - 8) * Math.sin(deg * Math.PI / 180));
        line.setAttribute('y2', CY - (R - 8) * Math.cos(deg * Math.PI / 180));
        line.setAttribute('stroke', 'rgba(28,41,60,0.4)');
        line.setAttribute('stroke-width', '1');
        this.svg.appendChild(line);
      }
      // Needle pivot (ink — the card is light, white would vanish)
      const pivot = document.createElementNS(ns, 'circle');
      pivot.setAttribute('cx', CX);
      pivot.setAttribute('cy', CY);
      pivot.setAttribute('r', '6');
      pivot.setAttribute('fill', '#1C293C');
      pivot.setAttribute('filter', 'drop-shadow(0 0 4px rgba(28,41,60,0.4))');
      this.svg.appendChild(pivot);

      // Needle marker (ink; glow takes the active zone color in update())
      this.needle = document.createElementNS(ns, 'g');
      const needleLine = document.createElementNS(ns, 'line');
      needleLine.setAttribute('x1', CX);
      needleLine.setAttribute('y1', CY - 8);
      needleLine.setAttribute('x2', CX);
      needleLine.setAttribute('y2', CY - R + 16);
      needleLine.setAttribute('stroke', '#1C293C');
      needleLine.setAttribute('stroke-width', '3');
      needleLine.setAttribute('stroke-linecap', 'round');
      this.needle.appendChild(needleLine);
      this.svg.appendChild(this.needle);

      this.needle.style.transformOrigin = `${CX}px ${CY}px`;
    }

    update(value, color) {
      const v = Utils.clamp(value, 0, 10);
      this.lastValue = v;
      const rot = START_DEG + (v / 10) * (END_DEG - START_DEG);
      this.needle.style.transition = 'transform 0.6s cubic-bezier(0.3,0.8,0.3,1)';
      this.needle.style.transform = `rotate(${rot}deg)`;
      this.needle.style.transformOrigin = `${CX}px ${CY}px`;
      this.needle.setAttribute('stroke', color || '#fff');
      this.needle.style.filter = `drop-shadow(0 0 ${8 + v * 2}px ${color || '#fff'})`;
    }
  }

  function drawSparkline(canvas, history, color) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth || 300;
    const h = canvas.height = canvas.clientHeight || 36;
    ctx.clearRect(0, 0, w, h);

    const values = (history || []).map((p) => Number(p.aisi || 0));
    if (values.length < 2) return;

    const min = Math.min(...values, 0);
    const max = Math.max(...values, 10);
    const span = Math.max(max - min, 1);

    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / span) * (h - 6) - 3;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color || '#432DD7';
    ctx.lineWidth = 2;
    ctx.shadowColor = color || '#432DD7';
    ctx.shadowBlur = 0;
    ctx.stroke();
  }

  global.AISIGauge = AISIGauge;
  global.drawSparkline = drawSparkline;
})(window);