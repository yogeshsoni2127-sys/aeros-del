/* ═══════════════════════════════════════════════════════════════
   AEROS — Cursor-reactive background pattern controller
   Lerped (buttery) pointer tracking → CSS vars --mx/--my.
   Spotlight dot-matrix + contour rings + warm glow follow cursor.
   GPU-cheap: only transforms + CSS vars, single rAF loop.
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  function init() {
    var root = document.documentElement;
    var body = document.body;
    var glow = document.getElementById('cursorGlow');
    var dot = document.getElementById('cursorDot');
    var staticGrid = document.getElementById('bg-static-grid');

    // Respect reduced motion + touch (no hover pattern there).
    var reduceMotion = false;
    try {
      reduceMotion =
        global.matchMedia &&
        global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) { /* ignore */ }
    var coarsePointer = false;
    try {
      coarsePointer =
        global.matchMedia && global.matchMedia('(pointer: coarse)').matches;
    } catch (e) { /* ignore */ }
    if (reduceMotion || coarsePointer) return;
    if (!glow && !dot) return;

    var tx = global.innerWidth * 0.5;   // target (instant)
    var ty = global.innerHeight * 0.38;
    var cx = tx, cy = ty;               // lerped (rendered)
    var gx = tx, gy = ty;               // glow lags a touch more
    var active = false;
    var rafId = 0;
    var LERP = 0.16;                    // cursor pattern follow speed
    var LERP_GLOW = 0.09;               // glow trails softer
    var PARALLAX = 10;                  // static grid drift (px max)

    function setVars(x, y) {
      root.style.setProperty('--mx', x.toFixed(1) + 'px');
      root.style.setProperty('--my', y.toFixed(1) + 'px');
    }
    setVars(cx, cy);

    function frame() {
      cx += (tx - cx) * LERP;
      cy += (ty - cy) * LERP;
      gx += (tx - gx) * LERP_GLOW;
      gy += (ty - gy) * LERP_GLOW;
      // Snap when close to avoid endless sub-pixel drift.
      if (Math.abs(tx - cx) < 0.05) cx = tx;
      if (Math.abs(ty - cy) < 0.05) cy = ty;

      setVars(cx, cy);

      if (glow) {
        glow.style.transform =
          'translate3d(' + gx.toFixed(1) + 'px,' + gy.toFixed(1) + 'px,0) translate(-50%,-50%)';
      }
      if (dot) {
        dot.style.transform =
          'translate3d(' + cx.toFixed(1) + 'px,' + cy.toFixed(1) + 'px,0)';
      }
      if (staticGrid) {
        // Gentle parallax opposite the cursor for depth.
        var nx = cx / Math.max(1, global.innerWidth) - 0.5;
        var ny = cy / Math.max(1, global.innerHeight) - 0.5;
        staticGrid.style.transform =
          'translate3d(' + (-nx * PARALLAX).toFixed(2) + 'px,' +
          (-ny * PARALLAX).toFixed(2) + 'px,0)';
      }

      if (active) {
        rafId = global.requestAnimationFrame(frame);
      } else {
        rafId = 0;
      }
    }

    function kick() {
      if (!rafId) rafId = global.requestAnimationFrame(frame);
    }

    function show() {
      root.style.setProperty('--glow-opacity', '1');
      active = true;
      kick();
    }

    function hide() {
      root.style.setProperty('--glow-opacity', '0');
      active = false;
    }

    document.addEventListener('pointermove', function (e) {
      if (e.pointerType === 'touch') return;
      tx = e.clientX;
      ty = e.clientY;
      show();
      kick();
    }, { passive: true });

    document.addEventListener('pointerenter', show);
    document.addEventListener('pointerleave', hide);
    document.addEventListener('blur', hide);
    document.documentElement.addEventListener('mouseleave', hide);

    // Idle fade: if the mouse stops, keep pattern but soften glow.
    // (No timer needed — the lerp settles naturally. Opacity stays.)

    // Hover boost on interactive elements → wider spotlight + ring.
    var HOVER_SEL =
      'a, button, input, summary, .station-row, .stable-row, ' +
      '.alert-card, .card, .toggle, .map-overlay';
    document.addEventListener('pointerover', function (e) {
      try {
        if (e.target && e.target.closest && e.target.closest(HOVER_SEL)) {
          if (body.getAttribute('data-hover') !== 'true') {
            body.setAttribute('data-hover', 'true');
          }
        } else if (body.getAttribute('data-hover') === 'true') {
          body.removeAttribute('data-hover');
        }
      } catch (err) { /* ignore */ }
    }, { passive: true });

    // Scroll reveal: calm Warp-like entrance for cards/panels.
    try {
      var revealEls = document.querySelectorAll(
        '.side-panel > .card, .ticker-bar'
      );
      revealEls.forEach(function (el) { el.classList.add('reveal'); });
      if ('IntersectionObserver' in global) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (en.isIntersecting) {
              en.target.classList.add('is-visible');
              io.unobserve(en.target);
            }
          });
        }, { threshold: 0.08 });
        revealEls.forEach(function (el) { io.observe(el); });
        // Above-the-fold should appear immediately.
        global.requestAnimationFrame(function () {
          revealEls.forEach(function (el) {
            var r = el.getBoundingClientRect();
            if (r.top < global.innerHeight && r.bottom > 0) {
              el.classList.add('is-visible');
            }
          });
        });
      } else {
        revealEls.forEach(function (el) { el.classList.add('is-visible'); });
      }
    } catch (e) { /* reveal is progressive enhancement */ }

    global.addEventListener('resize', kick, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
