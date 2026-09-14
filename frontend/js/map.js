/* ═══════════════════════════════════════════════════════════════
   AEROS — MapLibre GL Map Controller
   Base map + spatial AQI layers (heatmap, stations, plumes, fires).
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  const DELHI_CENTER = [77.2090, 28.6139];
  // Readable first: 'light' (Voyager) is the default — the Neobrutalism
  // theme is a warm light surface. Users can switch to dark anytime; the
  // choice persists in localStorage.
  const BASE_STYLES = {
    light: {
      tiles: ['https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png'],
      attr: '© OpenStreetMap contributors © CARTO',
    },
    dark: {
      tiles: ['https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'],
      attr: '© OpenStreetMap contributors © CARTO',
    },
  };

  function maptilerTiles(key) {
    return [`https://api.maptiler.com/maps/darkmatter/{z}/{x}/{y}.png?key=${key}`];
  }

  function savedBasemap() {
    try {
      const v = localStorage.getItem('aeros-basemap');
      if (v === 'dark' || v === 'light') return v;
    } catch (e) { /* private mode */ }
    return 'light';
  }
  const FIRE_REGIONS = [
    { name: 'Punjab', colors: '#ff3838', lat: 30.8, lon: 75.4 },
    { name: 'Haryana', colors: '#ffb800', lat: 29.6, lon: 76.4 },
  ];

  class AeriMap {
    constructor(containerId, maptilerKey) {
      this.container = document.getElementById(containerId);
      this.onStationClick = null;
      this.currentHour = 0;
      this._forecasts = {};
      // Basemap: saved choice (default light = warm surface). MapTiler key,
      // when configured, upgrades the dark style only.
      this.baseStyle = savedBasemap();
      document.body.dataset.basemap = this.baseStyle;
      let tiles = BASE_STYLES[this.baseStyle].tiles;
      let attribution = BASE_STYLES[this.baseStyle].attr;
      if (maptilerKey && this.baseStyle === 'dark') {
        tiles = maptilerTiles(maptilerKey);
        attribution = '© MapTiler © OpenStreetMap contributors';
      }
      this._maptilerKey = maptilerKey || null;
      this.map = new maplibregl.Map({
        container: this.container,
        style: {
          version: 8,
          sources: {
            base: {
              type: 'raster',
              tiles: tiles,
              tileSize: 256,
              attribution: attribution,
            },
          },
          layers: [{
            id: 'base',
            type: 'raster',
            source: 'base',
          }],
        },
        center: DELHI_CENTER,
        zoom: 8.2,
        attributionControl: true,
      });

      this.map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');
      this.map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');

      // If tiles are blocked (offline venue, firewall), say so instead of
      // showing a black rectangle — dots + labels still work on the
      // fallback background.
      this.map.on('error', (e) => {
        const src = (e && e.sourceId) || '';
        if (src === 'base') {
          const el = document.getElementById('mapStatus');
          if (el) el.textContent = 'basemap tiles blocked (offline?) — stations still live';
        }
      });

      this.map.on('load', () => this._onLoad());
    }

    setBasemap(name) {
      if (!BASE_STYLES[name]) return;
      this.baseStyle = name;
      try { localStorage.setItem('aeros-basemap', name); } catch (e) { /* ignore */ }
      document.body.dataset.basemap = name;
      document.querySelectorAll('[data-base-btn]').forEach((b) => {
        b.classList.toggle('active', b.dataset.baseBtn === name);
      });
      if (!this.map.isStyleLoaded()) return;
      let tiles = BASE_STYLES[name].tiles;
      let attribution = BASE_STYLES[name].attr;
      if (this._maptilerKey && name === 'dark') {
        tiles = maptilerTiles(this._maptilerKey);
        attribution = '© MapTiler © OpenStreetMap contributors';
      }
      // Re-add raster at the bottom (before the heat layer when present).
      if (this.map.getLayer('base')) this.map.removeLayer('base');
      if (this.map.getSource('base')) this.map.removeSource('base');
      this.map.addSource('base', {
        type: 'raster', tiles: tiles, tileSize: 256, attribution: attribution,
      });
      const before = this.map.getLayer('pm25-heat') ? 'pm25-heat' : undefined;
      this.map.addLayer({ id: 'base', type: 'raster', source: 'base' }, before);
      // Keep place labels legible on either basemap.
      if (this.map.getLayer('station-labels')) {
        const dark = name === 'dark';
        this.map.setPaintProperty('station-labels', 'text-color',
          dark ? 'rgba(255,255,255,0.88)' : 'rgba(16,20,46,0.92)');
        this.map.setPaintProperty('station-labels', 'text-halo-color',
          dark ? 'rgba(5,8,25,0.9)' : 'rgba(255,255,255,0.85)');
      }
    }

    _onLoad() {
      this._initSources();
      this._bindEvents();
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = 'map ready · live spatial overlay';
    }

    _initSources() {
      // Heatmap source (station points; weight metric switchable)
      this.map.addSource('pm25-heat', { type: 'geojson', data: emptyFC() });
      this.heatMetric = 'pm25';
      this.map.addLayer({
        id: 'pm25-heat',
        type: 'heatmap',
        source: 'pm25-heat',
        paint: {
          'heatmap-weight': heatWeightExpr('pm25'),
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 2.4],
          'heatmap-color': HEAT_RAMP,
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 12, 9, 34],
          'heatmap-opacity': 0.75,
        },
      });

      // Stations (circles colored by AQI)
      this.map.addSource('stations', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'stations-glow',
        type: 'circle',
        source: 'stations',
        paint: {
          'circle-radius': 12,
          'circle-color': ['get', 'color'],
          'circle-opacity': 0.25,
          'circle-blur': 1,
        },
      });
      this.map.addLayer({
        id: 'stations',
        type: 'circle',
        source: 'stations',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 5, 10, 8],
          'circle-color': ['get', 'color'],
          'circle-stroke-color': '#1C293C',
          'circle-stroke-width': 1.6,
        },
      });
      // Readable place labels — the "black map, can't read locations"
      // complaint was missing text: dots alone say nothing.
      this.map.addLayer({
        id: 'station-labels',
        type: 'symbol',
        source: 'stations',
        layout: {
          'text-field': ['get', 'name'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 7, 9, 10, 11],
          'text-offset': [0, 1.15],
          'text-anchor': 'top',
          'text-allow-overlap': false,
          'text-ignore-placement': false,
        },
        paint: {
          'text-color': this.baseStyle === 'dark'
            ? 'rgba(255,255,255,0.88)' : 'rgba(16,20,46,0.92)',
          'text-halo-color': this.baseStyle === 'dark'
            ? 'rgba(5,8,25,0.9)' : 'rgba(255,255,255,0.85)',
          'text-halo-width': 1.4,
        },
      });

      // Fire hotspots (pulsing red)
      this.map.addSource('fires', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'fires-halo',
        type: 'circle',
        source: 'fires',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'frp'], 0, 6, 80, 16],
          'circle-color': '#ff3838',
          'circle-opacity': 0.35,
          'circle-blur': 0.8,
        },
      });
      this.map.addLayer({
        id: 'fires',
        type: 'circle',
        source: 'fires',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'frp'], 0, 3, 80, 11],
          'circle-color': '#ff3838',
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1,
        },
      });

      // Plume trajectories
      this.map.addSource('plumes', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'plumes-dash',
        type: 'line',
        source: 'plumes',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'estimated_contribution_pm25'], 0, '#16A34A', 5, '#D97706', 15, '#DC2626'],
          'line-width': 2.2,
          'line-opacity': 0.85,
          'line-dasharray': [2, 1.4],
        },
      });

      // Delhi domain ring approximate boundary
      this.map.addSource('delhi-ring', {
        type: 'geojson',
        data: this._delhiRingGeoJSON(),
      });
      this.map.addLayer({
        id: 'delhi-ring',
        type: 'line',
        source: 'delhi-ring',
        paint: {
          // Ink dash: boundary reads via rhythm on either basemap.
          'line-color': 'rgba(28,41,60,0.55)',
          'line-width': 2,
          'line-dasharray': [1, 1],
        },
      });
    }

    _bindEvents() {
      this.map.on('click', 'stations', (e) => {
        const prop = (e.features[0] || {}).properties || {};
        if (this.onStationClick) this.onStationClick(prop.id || prop.name);
        this._flyTo(e.lngLat.lng, e.lngLat.lat);
      });

      this.map.on('mouseenter', 'stations', () => {
        this.map.getCanvas().style.cursor = 'pointer';
      });
      this.map.on('mouseleave', 'stations', () => {
        this.map.getCanvas().style.cursor = '';
      });

      this.map.on('mousemove', 'fires', (e) => {
        const p = e.features[0].properties;
        this._tooltip(`Fire @ ${p.latitude.toFixed?.(2) ?? p.lat ?? ''} · FRP ${p.frp} MW`);
      });
      this.map.on('mouseleave', 'fires', () => this._clearTooltip());
    }

    _flyTo(lon, lat) {
      this.map.flyTo({ center: [lon, lat], zoom: 9.5, duration: 800 });
    }

    _tooltip(text) {
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = text;
    }
    _clearTooltip() {
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = 'hover a fire marker for details';
    }

    /* ── Public update API ─────────────────────────────────────── */
    update(forecastGeojson, stationsGeojson, firesGeojson, plumesGeojson) {
      if (!this.map.isStyleLoaded()) return;
      this._updateSource('stations',
        stationColorsRemapped(stationsGeojson || emptyFC()));
      this._updateSource('pm25-heat', forecastGeojson || stationsGeojson || emptyFC());
      this._updateSource('fires', firesGeojson || emptyFC());
      this._updateSource('plumes', plumesGeojson || emptyFC());
    }

    setTime(hour) {
      this.currentHour = hour;
      // Stations layer can reflect forecast hour if forecasts provided
      const fc = this._forecasts;
      if (!fc || !Object.keys(fc).length) return;
      const features = [];
      for (const id in fc) {
        const f = fc[id];
        if (!f || !f.pm25 || hour >= f.pm25.length) continue;
        const st = this._stationLookup && this._stationLookup[id];
        if (!st) continue;
        features.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [st.longitude, st.latitude] },
          properties: {
            id: id,
            name: st.short_name,
            pm25: f.pm25[hour],
            pm10: (f.pm10 || [])[hour] || 0,
            no2: (f.no2 || [])[hour] || 0,
            o3: (f.o3 || [])[hour] || 0,
            aqi: (f.aqi || [])[hour] || 0,
            category: (f.category || [])[hour] || 'Unknown',
            color: stationColor((f.colors || [])[hour] || '#808080'),
          },
        });
      }
      this._updateSource('stations', { type: 'FeatureCollection', features });
      this._updateSource('pm25-heat', { type: 'FeatureCollection', features });
    }

    setForecasts(forecasts, stationLookup) {
      this._forecasts = forecasts || {};
      this._stationLookup = stationLookup || null;
    }

    toggleLayer(id, visible) {
      const layerIds = {
        heat: 'pm25-heat',
        fires: ['fires', 'fires-halo'],
        plumes: 'plumes-dash',
        stations: ['stations', 'stations-glow', 'station-labels'],
      };
      const targets = layerIds[id];
      if (!targets) return;
      const args = Array.isArray(targets) ? targets : [targets];
      args.forEach((lid) => {
        if (this.map.getLayer(lid)) this.map.setLayoutProperty(lid, 'visibility', visible ? 'visible' : 'none');
      });
    }

    // Switch the heatmap weight between PM2.5 / PM10 / NO2 / O3.
    setHeatMetric(metric) {
      if (!HEAT_METRICS[metric]) return;
      this.heatMetric = metric;
      if (this.map.isStyleLoaded() && this.map.getLayer('pm25-heat')) {
        this.map.setPaintProperty('pm25-heat', 'heatmap-weight',
          heatWeightExpr(metric));
      }
      document.querySelectorAll('[data-heat-metric]').forEach((b) => {
        b.classList.toggle('active', b.dataset.heatMetric === metric);
      });
    }

    /* ── Helpers ───────────────────────────────────────────────── */
    _updateSource(id, data) {
      const src = this.map.getSource(id);
      if (src) src.setData(data);
    }

    _delhiRingGeoJSON() {
      const features = [];
      const R = 0.75; // deg approx
      const coords = [];
      for (let i = 0; i <= 48; i++) {
        const a = (i / 48) * Math.PI * 2;
        coords.push([
          DELHI_CENTER[0] + R * Math.cos(a) * 1.15,
          DELHI_CENTER[1] + R * Math.sin(a) * 0.92,
        ]);
      }
      features.push({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords },
        properties: { name: 'Delhi NCR approx boundary' },
      });
      return { type: 'FeatureCollection', features };
    }
  }

  function emptyFC() {
    return { type: 'FeatureCollection', features: [] };
  }

  // Heat metric config: (property, weight stops, label). Darker ramp
  // throughout so low densities stay visible on the light basemap.
  const HEAT_METRICS = {
    pm25: { prop: 'pm25', stops: [0, 0, 150, 0.6, 500, 1], label: 'PM2.5' },
    pm10: { prop: 'pm10', stops: [0, 0, 250, 0.6, 800, 1], label: 'PM10' },
    no2:  { prop: 'no2',  stops: [0, 0, 80, 0.6, 300, 1],  label: 'NO₂' },
    o3:   { prop: 'o3',   stops: [0, 0, 100, 0.6, 250, 1], label: 'O₃' },
  };
  const HEAT_RAMP = [
    'interpolate', ['linear'], ['heatmap-density'],
    0, 'rgba(0,0,0,0)',
    0.12, 'rgba(0,77,141,0.45)',
    0.3, 'rgba(0,112,209,0.55)',
    0.48, 'rgba(213,59,0,0.65)',
    0.66, 'rgba(200,27,58,0.8)',
    0.84, 'rgba(153,0,76,0.9)',
    1, 'rgba(80,0,20,0.95)',
  ];

  function heatWeightExpr(metric) {
    const cfg = HEAT_METRICS[metric] || HEAT_METRICS.pm25;
    return ['interpolate', ['linear'], ['get', cfg.prop]].concat(cfg.stops);
  }

  // Display override for the light theme: Moderate-yellow (#ffff00)
  // and Satisfactory-mint (#9cff9c) wash out on warm white, so they
  // render as warning-amber / success-green. Duplicated here so the map
  // module never depends on utils.js load order.
  const DISPLAY_COLOR_MAP = {
    '#ffff00': '#D97706',
    '#9cff9c': '#16A34A',
  };

  function stationColor(hex) {
    if (typeof hex === 'string') {
      const mapped = DISPLAY_COLOR_MAP[hex.toLowerCase()];
      if (mapped) return mapped;
    }
    return hex;
  }

  function stationColorsRemapped(fc) {
    if (!fc || !Array.isArray(fc.features)) return fc;
    for (const feat of fc.features) {
      if (feat && feat.properties && feat.properties.color != null) {
        feat.properties.color = stationColor(feat.properties.color);
      }
    }
    return fc;
  }

  global.AeriMap = AeriMap;
  global.DELHI_CENTER = DELHI_CENTER;
})(window);