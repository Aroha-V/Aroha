'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

// ─── Risk colour scheme ────────────────────────────────────────────────────
const RISK_COLORS = {
  HIGH: { border: 'border-red-500/30', text: 'text-red-400', badge: 'bg-red-500/20 border-red-500/40 text-red-400', fill: '#ef4444cc', stroke: '#dc2626' },
  MEDIUM: { border: 'border-amber-500/30', text: 'text-amber-400', badge: 'bg-amber-500/20 border-amber-500/40 text-amber-400', fill: '#f59e0bcc', stroke: '#d97706' },
  LOW: { border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400', fill: '#10b981cc', stroke: '#059669' },
  NONE: { border: 'border-slate-500/30', text: 'text-slate-400', badge: 'bg-slate-500/20 border-slate-500/40 text-slate-400', fill: '#94a3b8aa', stroke: '#64748b' },
};

// Convert display state name → API slug (lowercase)
function toApiSlug(name) {
  return name.toLowerCase();
}

// Beautiful distinct palette for the 36 states/UTs (shown when not in risk-mode)
const STATE_PALETTE = [
  '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#f97316', '#eab308',
  '#84cc16', '#22c55e', '#10b981', '#14b8a6', '#06b6d4', '#3b82f6',
  '#6366f1', '#a855f7', '#d946ef', '#fb923c', '#fbbf24', '#a3e635',
  '#4ade80', '#34d399', '#2dd4bf', '#38bdf8', '#60a5fa', '#818cf8',
  '#c084fc', '#f472b6', '#fb7185', '#fdba74', '#fde68a', '#bef264',
  '#86efac', '#6ee7b7', '#5eead4', '#7dd3fc', '#93c5fd', '#a5b4fc',
];

// All mainland Indian states & UTs (excluding island territories)
const EXCLUDED = new Set([
  'Andaman and Nicobar Islands',
  'Andaman & Nicobar Islands',
  'Lakshadweep',
]);

const DISPLAY_STATES = [
  'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
  'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
  'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
  'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
  'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
  'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Chandigarh',
  'Dadra and Nagar Haveli and Daman and Diu', 'Puducherry',
];

// ─── Status badge helper ─────────────────────────────────────────────────────
function StatusBadge({ status }) {
  if (!status || status === '—') return <span className="text-slate-400 dark:text-slate-500 text-xs font-mono">—</span>;
  const s = status.toLowerCase();
  let cls = 'text-xs font-mono font-semibold px-2 py-0.5 rounded border ';
  if (s.includes('control')) cls += 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400';
  else if (s.includes('surveillance')) cls += 'bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400';
  else if (s.includes('active') || s.includes('outbreak')) cls += 'bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400';
  else cls += 'bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300';
  return <span className={cls}>{status}</span>;
}

// ─── Mercator projection ────────────────────────────────────────────────────
// India bounding box: lon 68–98°E, lat 8–37.5°N
const INDIA_LON_MIN = 67.5, INDIA_LON_MAX = 98.5;
const INDIA_LAT_MIN = 7.5, INDIA_LAT_MAX = 37.8;

function project(lon, lat, width, height, padding = 16) {
  const w = width - padding * 2;
  const h = height - padding * 2;
  // Mercator y
  const latRad = (lat * Math.PI) / 180;
  const y0 = Math.log(Math.tan(Math.PI / 4 + (INDIA_LAT_MIN * Math.PI) / 360));
  const y1 = Math.log(Math.tan(Math.PI / 4 + (INDIA_LAT_MAX * Math.PI) / 360));
  const yMerc = Math.log(Math.tan(Math.PI / 4 + latRad / 2));
  const x = padding + ((lon - INDIA_LON_MIN) / (INDIA_LON_MAX - INDIA_LON_MIN)) * w;
  const y = padding + (1 - (yMerc - y0) / (y1 - y0)) * h;
  return [x, y];
}

function geojsonToSVGPath(geometry, width, height) {
  const convert = (ring) =>
    ring.map(([lon, lat], i) => {
      const [x, y] = project(lon, lat, width, height);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ') + ' Z';

  if (geometry.type === 'Polygon') {
    return geometry.coordinates.map(convert).join(' ');
  }
  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates.flatMap((poly) => poly.map(convert)).join(' ');
  }
  return '';
}

// ─── Geometry helpers ──────────────────────────────────────────────────────────

// Shoelace formula centroid (lon/lat coords, returns [lon,lat])
function polygonCentroid(ring) {
  let area = 0, cx = 0, cy = 0;
  const n = ring.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const cross = xi * yj - xj * yi;
    area += cross;
    cx += (xi + xj) * cross;
    cy += (yi + yj) * cross;
  }
  area /= 2;
  if (Math.abs(area) < 1e-12) {
    // degenerate — fall back to simple average
    return [
      ring.reduce((s, c) => s + c[0], 0) / n,
      ring.reduce((s, c) => s + c[1], 0) / n,
    ];
  }
  return [cx / (6 * area), cy / (6 * area)];
}

// Shoelace signed area (lon/lat)
function polygonArea(ring) {
  let area = 0;
  const n = ring.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    area += ring[i][0] * ring[j][1];
    area -= ring[j][0] * ring[i][1];
  }
  return Math.abs(area) / 2;
}

// Pick the largest sub-polygon ring of a geometry
function largestRing(geometry) {
  if (geometry.type === 'Polygon') return geometry.coordinates[0];
  if (geometry.type === 'MultiPolygon') {
    let best = [], bestA = 0;
    for (const poly of geometry.coordinates) {
      const a = polygonArea(poly[0]);
      if (a > bestA) { bestA = a; best = poly[0]; }
    }
    return best;
  }
  return null;
}

// Manual label nudges (in lon/lat degrees) for states whose centroid
// falls outside the polygon (thin/concave shapes).
const LABEL_NUDGE = {
  'Kerala':         [0.5, 1.5],
  'Goa':            [0.3, 0.3],
  'Manipur':        [0.2, 0.2],
  'Nagaland':       [0.5, 0.5],
  'Mizoram':        [0.3, 0.5],
  'Tripura':        [-0.3, 0.4],
  'Sikkim':         [0.5, 0.5],
  'Chandigarh':     [0.8, 0.5],
  'Puducherry':     [0.5, 1.0],
  'Delhi':          [0.3, 0.3],
  'Haryana':        [-0.3, 0],
  'Punjab':         [-0.3, 0],
  'Uttarakhand':    [0, 0.5],
  'Himachal Pradesh': [0, 0.5],
};

// Split a state name into at most two display lines
function splitLabel(raw) {
  const s = raw
    .replace('Dadra and Nagar Haveli and Daman and Diu', 'D&NH &\nD&D')
    .replace('Jammu and Kashmir', 'Jammu &\nKashmir')
    .replace('Arunachal Pradesh', 'Arunachal\nPradesh')
    .replace('Himachal Pradesh', 'Himachal\nPradesh')
    .replace('Madhya Pradesh', 'Madhya\nPradesh')
    .replace('Andhra Pradesh', 'Andhra\nPradesh')
    .replace('Uttar Pradesh', 'Uttar\nPradesh')
    .replace('West Bengal', 'West\nBengal')
    .replace('Tamil Nadu', 'Tamil\nNadu');
  return s.split('\n');
}

// ─── Zoom controls ──────────────────────────────────────────────────────────
const MIN_SCALE = 1;
const MAX_SCALE = 8;
const ZOOM_STEP = 0.5;

// ─── Map Component ──────────────────────────────────────────────────────────
function IndiaMap({ geoFeatures, stateApiData, selectedState, onSelectState, showRisk }) {
  const [tooltip, setTooltip] = useState(null);
  const [transform, setTransform] = useState({ scale: 1, tx: 0, ty: 0 });
  const svgRef = useRef(null);
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const W = 620, H = 720;

  // ── clamp pan so map never drifts completely off screen ──
  const clampTranslate = useCallback((tx, ty, scale) => {
    const maxTx = (scale - 1) * W * 0.6;
    const maxTy = (scale - 1) * H * 0.6;
    return [
      Math.max(-maxTx, Math.min(maxTx, tx)),
      Math.max(-maxTy, Math.min(maxTy, ty)),
    ];
  }, []);

  // ── zoom towards a point (svgX, svgY) in SVG viewBox coords ──
  const zoomAt = useCallback((svgX, svgY, delta) => {
    setTransform((prev) => {
      const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, prev.scale + delta));
      if (newScale === prev.scale) return prev;
      // Keep the cursor point fixed
      const factor = newScale / prev.scale;
      const newTx = svgX + factor * (prev.tx - svgX);
      const newTy = svgY + factor * (prev.ty - svgY);
      const [tx, ty] = clampTranslate(newTx, newTy, newScale);
      return { scale: newScale, tx, ty };
    });
  }, [clampTranslate]);

  // ── button zoom (towards centre) ──
  const zoom = useCallback((dir) => zoomAt(W / 2, H / 2, dir * ZOOM_STEP), [zoomAt]);
  const resetZoom = useCallback(() => setTransform({ scale: 1, tx: 0, ty: 0 }), []);

  // ── mouse-wheel zoom ──
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const { scale, tx, ty } = transform;
    // Convert mouse pos to SVG viewBox coords
    const mx = ((e.clientX - rect.left) / rect.width) * W;
    const my = ((e.clientY - rect.top) / rect.height) * H;
    // Undo current transform to get pre-transform SVG point
    const svgX = (mx - tx) / scale;
    const svgY = (my - ty) / scale;
    const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
    zoomAt(svgX * scale + tx, svgY * scale + ty, delta);
  }, [transform, zoomAt]);

  // ── drag to pan ──
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    isDragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setTooltip(null);
  }, []);

  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    // Convert pixel delta → SVG unit delta
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const svgDx = (dx / rect.width) * W;
    const svgDy = (dy / rect.height) * H;
    setTransform((prev) => {
      const [tx, ty] = clampTranslate(prev.tx + svgDx, prev.ty + svgDy, prev.scale);
      return { ...prev, tx, ty };
    });
  }, [clampTranslate]);

  const onMouseUp = useCallback(() => { isDragging.current = false; }, []);

  // Attach wheel listener (non-passive to allow preventDefault)
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  const handleMouseMoveState = useCallback((e, name) => {
    if (isDragging.current) return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const records = stateApiData[name] || [];
    const totalCases = records.reduce((s, r) => s + (r.no_of_cases || 0), 0);
    setTooltip({
      x: e.clientX - rect.left + 14,
      y: e.clientY - rect.top - 8,
      name,
      totalCases,
      count: records.length,
    });
  }, [stateApiData]);

  const { scale, tx, ty } = transform;

  return (
    <div className="relative w-full h-full select-none">
      {/* ── Zoom Controls (top-right) ── */}
      <div className="absolute top-3 right-3 z-20 flex flex-col gap-1.5" style={{ filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.18))' }}>
        <button
          onClick={() => zoom(1)}
          title="Zoom In"
          className="w-9 h-9 flex items-center justify-center rounded-lg bg-white/90 dark:bg-slate-700/90 backdrop-blur-sm border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-200 text-lg font-bold hover:bg-blue-600 hover:text-white hover:border-blue-500 transition-all duration-150 shadow-md"
        >
          +
        </button>
        <button
          onClick={() => zoom(-1)}
          title="Zoom Out"
          className="w-9 h-9 flex items-center justify-center rounded-lg bg-white/90 dark:bg-slate-700/90 backdrop-blur-sm border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-200 text-lg font-bold hover:bg-blue-600 hover:text-white hover:border-blue-500 transition-all duration-150 shadow-md"
        >
          −
        </button>
        <button
          onClick={resetZoom}
          title="Reset Zoom"
          className="w-9 h-9 flex items-center justify-center rounded-lg bg-white/90 dark:bg-slate-700/90 backdrop-blur-sm border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 text-sm font-bold hover:bg-slate-700 hover:text-white hover:border-slate-600 transition-all duration-150 shadow-md"
        >
          ⟳
        </button>
        <div className="w-9 h-7 flex items-center justify-center rounded-md bg-slate-800/80 backdrop-blur-sm text-blue-300 font-mono text-[10px] font-bold">
          {scale.toFixed(1)}×
        </div>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
        style={{
          background: 'linear-gradient(135deg,#dbeafe 0%,#e0f2fe 50%,#bfdbfe 100%)',
          borderRadius: 12,
          filter: 'drop-shadow(0 2px 16px rgba(30,58,138,0.10))',
          cursor: isDragging.current ? 'grabbing' : scale > 1 ? 'grab' : 'default',
        }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { isDragging.current = false; setTooltip(null); }}
      >
        <text x={W / 2} y={H - 10} textAnchor="middle" fontSize="9" fontFamily="monospace" fill="#93c5fd" fillOpacity="0.7">
          India — IDSP Disease Surveillance Map
        </text>
        <g
          transform={`translate(${tx.toFixed(2)},${ty.toFixed(2)}) scale(${scale.toFixed(4)})`}
          style={{ transformOrigin: '0 0', transition: isDragging.current ? 'none' : 'transform 0.18s cubic-bezier(0.25,0.46,0.45,0.94)' }}
        >
          {geoFeatures.map((feature, idx) => {
            const name = feature.properties?.st_nm || feature.properties?.ST_NM || feature.properties?.NAME_1 || feature.properties?.name || '';
            const isSelected = selectedState === name;
            if (EXCLUDED.has(name)) return null;
            const pathD = geojsonToSVGPath(feature.geometry, W, H);
            if (!pathD) return null;
            const records = stateApiData[name] || [];
            const totalCases = records.reduce((s, r) => s + (r.no_of_cases || 0), 0);
            let fill, strokeWidth;
            if (isSelected) {
              fill = '#2563eb'; strokeWidth = 2.2;
            } else if (showRisk) {
              fill = totalCases >= 500 ? '#ef4444cc' : totalCases >= 100 ? '#f59e0bcc' : totalCases > 0 ? '#10b981cc' : '#94a3b8aa';
              strokeWidth = 1.4;
            } else {
              fill = STATE_PALETTE[idx % STATE_PALETTE.length] + 'cc';
              strokeWidth = 1.4;
            }
            const stroke = isSelected ? '#1e3a8a' : '#1a1a1a';
            return (
              <path
                key={'path-' + idx}
                d={pathD}
                fill={fill}
                stroke={stroke}
                strokeWidth={strokeWidth}
                strokeLinejoin="round"
                style={{ cursor: 'pointer', transition: 'fill 0.25s', filter: isSelected ? 'brightness(1.15) drop-shadow(0 0 6px #3b82f6aa)' : 'none' }}
                onClick={(e) => { if (!isDragging.current) { e.stopPropagation(); name && onSelectState(name); } }}
                onMouseMove={(e) => name && handleMouseMoveState(e, name)}
                onMouseLeave={() => setTooltip(null)}
              />
            );
          })}
          {DISPLAY_STATES.map((name) => {
            // Collect ALL district features for this state, pick the largest ring for centroid
            const stateFeatures = geoFeatures.filter(f => {
              const fname = f.properties?.st_nm || f.properties?.ST_NM || f.properties?.NAME_1 || f.properties?.name || '';
              return fname === name;
            });
            if (stateFeatures.length === 0) return null;

            // Find the single largest polygon ring across all districts → best centroid
            let bestRing = null, bestArea = 0;
            for (const f of stateFeatures) {
              const ring = largestRing(f.geometry);
              if (!ring || ring.length < 3) continue;
              const a = polygonArea(ring);
              if (a > bestArea) { bestArea = a; bestRing = ring; }
            }
            if (!bestRing) return null;

            let [lonC, latC] = polygonCentroid(bestRing);
            const nudge = LABEL_NUDGE[name];
            if (nudge) { lonC += nudge[0]; latC += nudge[1]; }
            const [px, py] = project(lonC, latC, W, H);

            // Font size based on total state area (sum of all districts)
            const totalArea = stateFeatures.reduce((sum, f) => {
              const r = largestRing(f.geometry);
              return sum + (r ? polygonArea(r) : 0);
            }, 0);
            const baseFontSize = Math.max(8, Math.min(15, 6 + Math.sqrt(totalArea) * 0.9));
            const fs = baseFontSize / scale;
            const lines = splitLabel(name);
            const lineH = fs * 1.5;
            const offsetY = -((lines.length - 1) * lineH) / 2;
            return (
              <g key={'lbl-' + name} pointerEvents="none">
                {lines.map((line, li) => (
                  <text
                    key={li}
                    x={px}
                    y={py + offsetY + li * lineH}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontFamily="'Arial Narrow', Arial, sans-serif"
                    fontWeight="700"
                    fontSize={fs}
                    fill="#ffffff"
                    stroke="#000000"
                    strokeWidth={fs * 0.6}
                    paintOrder="stroke"
                  >
                    {line}
                  </text>
                ))}
              </g>
            );
          })}
        </g>
      </svg>

      {tooltip && (
        <div
          className="absolute z-50 bg-slate-900/95 backdrop-blur-sm text-white text-xs font-mono rounded-lg px-3 py-2.5 pointer-events-none shadow-2xl border border-slate-600"
          style={{ left: tooltip.x, top: tooltip.y, maxWidth: 200, minWidth: 140 }}
        >
          <div className="font-bold text-blue-300 mb-1.5 text-sm">{tooltip.name}</div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Outbreaks</span>
            <span className="text-amber-300 font-semibold">{tooltip.count}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Total Cases</span>
            <span className="text-emerald-300 font-semibold">{tooltip.totalCases.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
export default function Monitor() {
  const [selectedState, setSelectedState] = useState('Ladakh');
  const [geoFeatures, setGeoFeatures] = useState([]);
  const [mapLoading, setMapLoading] = useState(true);
  const [showRisk, setShowRisk] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Sync dark mode with the html.dark class (set by Navbar toggler)
  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('theme');
    const isDark = saved ? saved === 'dark' : true; // default: dark
    setDarkMode(isDark);
    document.documentElement.classList.toggle('dark', isDark);
    // Watch for changes from Navbar toggle
    const observer = new MutationObserver(() => {
      setDarkMode(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  // API data: { [stateName]: Record[] }  — cached per state
  const [stateApiData, setStateApiData] = useState({});
  // Records for currently selected state
  const [selectedRecords, setSelectedRecords] = useState([]);
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [apiError, setApiError] = useState(null);

  // Load India GeoJSON
  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@ef25ebc/geojson/india.geojson')
      .then((r) => r.json())
      .then((geo) => {
        setGeoFeatures(geo.features || []);
        setMapLoading(false);
      })
      .catch(() => setMapLoading(false));
  }, []);

  // Fetch API records for selected state (with cache)
  useEffect(() => {
    if (!selectedState) return;
    if (stateApiData[selectedState]) {
      setSelectedRecords(stateApiData[selectedState]);
      return;
    }
    setLoadingRecords(true);
    setApiError(null);
    const slug = toApiSlug(selectedState);
    fetch(`/api/disease/${encodeURIComponent(slug)}`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((data) => {
        const records = Object.values(data).map((rec) => ({
          action: rec.action || '',
          disease: rec.disease || '—',
          district: rec.district || '—',
          no_of_cases: rec.no_of_cases != null && !isNaN(rec.no_of_cases) ? Number(rec.no_of_cases) : null,
          no_deaths: rec.no_deaths != null && !isNaN(rec.no_deaths) ? Number(rec.no_deaths) : null,
          state: rec.state || selectedState,
          status: rec.status || '—',
        }));
        setSelectedRecords(records);
        setStateApiData((prev) => ({ ...prev, [selectedState]: records }));
        setLoadingRecords(false);
      })
      .catch((err) => {
        setApiError(`Failed to load data: ${err.message}`);
        setSelectedRecords([]);
        setLoadingRecords(false);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedState]);

  // KPI stats from selected state records
  const totalCases = selectedRecords.reduce((s, r) => s + (r.no_of_cases ?? 0), 0);
  const totalDeaths = selectedRecords.reduce((s, r) => s + (r.no_deaths ?? 0), 0);
  const uniqueDiseases = [...new Set(selectedRecords.map((r) => r.disease).filter(Boolean))].length;
  const outbreakCount = selectedRecords.length;

  const kpis = [
    { label: 'Total Cases', value: totalCases.toLocaleString() },
    { label: 'Total Deaths', value: totalDeaths > 0 ? totalDeaths.toLocaleString() : '—' },
    { label: 'Diseases Tracked', value: uniqueDiseases },
    { label: 'Outbreak Events', value: outbreakCount },
  ];

  // Charcoal palette (ChatGPT-style)
  const bg      = darkMode ? '#212121' : '#ffffff';
  const cardBg  = darkMode ? '#2f2f2f' : '#f8fafc';
  const cardBg2 = darkMode ? '#3a3a3a' : '#f1f5f9';
  const border  = darkMode ? 'rgba(255,255,255,0.08)' : '#e2e8f0';
  const txtMain = darkMode ? '#ececec'  : '#0f172a';
  const txtSub  = darkMode ? '#8e8ea0'  : '#475569';
  const txtMute = darkMode ? '#6b7280'  : '#94a3b8';
  const inputBg = darkMode ? '#2f2f2f'  : '#f9fafb';

  return (
    <main className="min-h-screen pt-20 transition-colors duration-300" style={{ background: bg }}>
      <div className="max-w-[1280px] mx-auto px-6 py-8">

        {/* ─── Header ─── */}
        <div className="mb-8 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-4xl font-black tracking-tight mb-2" style={{ color: txtMain }}>
              🧫 India Disease Outbreak Monitor
            </h1>
            <p className="font-mono text-sm tracking-widest uppercase" style={{ color: txtMute }}>
              IDSP Surveillance · Live API Data Interface
            </p>
          </div>

          {/* ── Theme Toggle ── */}
          {mounted && (
            <button
              id="monitor-theme-toggle"
              onClick={toggleTheme}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              className="relative w-10 h-10 flex items-center justify-center rounded-xl flex-shrink-0 transition-all duration-200 overflow-hidden shadow-sm"
              style={{
                background: darkMode ? '#2f2f2f' : '#f9fafb',
                border: `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : '#e2e8f0'}`,
                color: darkMode ? '#fbbf24' : '#64748b',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = darkMode ? '#3a3a3a' : '#f1f5f9'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = darkMode ? '#2f2f2f' : '#f9fafb'; }}
            >
              {/* Sun — shown in dark mode */}
              <span style={{ position: 'absolute', transition: 'transform 0.3s ease, opacity 0.3s ease', transform: darkMode ? 'rotate(0deg) scale(1)' : 'rotate(90deg) scale(0.5)', opacity: darkMode ? 1 : 0 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              </span>
              {/* Moon — shown in light mode */}
              <span style={{ position: 'absolute', transition: 'transform 0.3s ease, opacity 0.3s ease', transform: darkMode ? 'rotate(-90deg) scale(0.5)' : 'rotate(0deg) scale(1)', opacity: darkMode ? 0 : 1 }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />
                </svg>
              </span>
            </button>
          )}
        </div>

        {/* ─── Controls ─── */}
        <div className="flex flex-col md:flex-row gap-4 items-start md:items-center mb-8 flex-wrap">
          <div className="flex items-center gap-3">
            <label className="font-mono text-xs font-semibold uppercase tracking-widest whitespace-nowrap" style={{ color: txtSub }}>State / UT</label>
            <select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              className="px-3 py-2 rounded-lg text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500/30 cursor-pointer"
              style={{ background: inputBg, border: `1px solid ${border}`, color: txtMain }}
            >
              {DISPLAY_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Risk / Colour toggle */}
          <button
            onClick={() => setShowRisk((v) => !v)}
            className="px-4 py-2 rounded-lg text-xs font-mono font-semibold transition-all"
            style={{
              background: showRisk ? '#2563eb' : inputBg,
              border: `1px solid ${showRisk ? '#3b82f6' : border}`,
              color: showRisk ? '#ffffff' : txtSub,
              boxShadow: showRisk ? '0 4px 12px rgba(37,99,235,0.25)' : 'none',
            }}
          >
            {showRisk ? '🔴 Risk Choropleth ON' : '🎨 Risk Choropleth OFF'}
          </button>

          {/* Legend */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-xs font-semibold uppercase tracking-widest" style={{ color: txtSub }}>RISK</span>
            {['HIGH', 'MEDIUM', 'LOW', 'NONE'].map((level) => (
              <span key={level} className={`text-xs font-mono font-semibold px-2.5 py-1 rounded border ${RISK_COLORS[level].badge}`}>{level}</span>
            ))}
          </div>
        </div>

        {/* ─── KPI Cards ─── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {kpis.map((kpi) => (
            <div key={kpi.label} className="rounded-xl p-4 transition-all duration-200"
              style={{ background: cardBg, border: `1px solid ${border}` }}>
              <div className="font-mono text-2xl font-black text-blue-500 mb-1">{kpi.value}</div>
              <div className="text-xs font-mono uppercase tracking-widest font-semibold" style={{ color: txtSub }}>{kpi.label}</div>
            </div>
          ))}
        </div>

        <div className="h-px mb-8" style={{ background: `linear-gradient(90deg, transparent, ${border} 50%, transparent)` }} />

        {/* ─── Main Content Grid ─── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">

          {/* ─── Map Section ─── */}
          <div className="lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <p className="font-mono text-xs font-semibold uppercase tracking-widest" style={{ color: txtSub }}>India Choropleth · Click a state to analyse</p>
              {mapLoading && (<span className="text-xs font-mono text-blue-500 animate-pulse">Loading map…</span>)}
            </div>
            <div className="rounded-xl overflow-hidden" style={{ height: 580, border: `1px solid ${border}`, background: cardBg }}>
              {mapLoading ? (
                <div className="flex items-center justify-center w-full h-full">
                  <div className="text-center">
                    <div className="text-5xl mb-3 animate-bounce">🗺️</div>
                    <p className="font-mono text-sm" style={{ color: txtSub }}>Loading India GeoJSON map…</p>
                  </div>
                </div>
              ) : geoFeatures.length === 0 ? (
                <div className="flex items-center justify-center w-full h-full">
                  <p className="font-mono text-sm" style={{ color: txtMute }}>Failed to load map. Check your connection.</p>
                </div>
              ) : (
                <IndiaMap
                  geoFeatures={geoFeatures}
                  stateApiData={stateApiData}
                  selectedState={selectedState}
                  onSelectState={setSelectedState}
                  showRisk={showRisk}
                />
              )}
            </div>

            {/* Map legend */}
            <div className="flex items-center gap-6 mt-3 flex-wrap">
              {showRisk ? (
                <>
                  {[{ label: '≥ 500 total cases', fill: '#ef4444' }, { label: '100–499 cases', fill: '#f59e0b' }, { label: '1–99 cases', fill: '#10b981' }, { label: 'No data loaded', fill: '#94a3b8' }].map(({ label, fill }) => (
                    <div key={label} className="flex items-center gap-1.5">
                      <span className="inline-block w-3 h-3 rounded-sm" style={{ background: fill }} />
                      <span className="text-xs font-mono" style={{ color: txtSub }}>{label}</span>
                    </div>
                  ))}
                </>
              ) : (
                <p className="text-xs font-mono" style={{ color: txtMute }}>🎨 Distinct colours per state — toggle Risk Choropleth to see outbreak severity</p>
              )}
            </div>
          </div>

          {/* ─── State Detail Panel ─── */}
          <div>
            <div className="mb-3">
              <p className="font-mono text-xs font-semibold uppercase tracking-widest" style={{ color: txtSub }}>State Detail Panel</p>
            </div>
            <select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500/30 cursor-pointer mb-3"
              style={{ background: inputBg, border: `1px solid ${border}`, color: txtMain }}
            >
              {DISPLAY_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>

            <div className="rounded-xl p-5" style={{ background: cardBg, border: `1px solid ${border}` }}>
              <h3 className="font-mono font-black text-lg mb-3 text-blue-400">{selectedState}</h3>

              {loadingRecords ? (
                <div className="flex items-center gap-2 py-4">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  <span className="font-mono text-xs" style={{ color: txtSub }}>Fetching API data…</span>
                </div>
              ) : apiError ? (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <p className="text-red-400 font-mono text-xs">{apiError}</p>
                </div>
              ) : selectedRecords.length === 0 ? (
                <p className="font-mono text-xs" style={{ color: txtMute }}>No outbreak records found for this state.</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div>
                      <div className="font-mono text-3xl font-black text-blue-400">{totalCases.toLocaleString()}</div>
                      <div className="text-xs font-mono uppercase tracking-widest font-semibold mt-1" style={{ color: txtSub }}>Total Cases</div>
                    </div>
                    <div>
                      <div className="font-mono text-3xl font-black" style={{ color: txtMain }}>{outbreakCount}</div>
                      <div className="text-xs font-mono uppercase tracking-widest font-semibold mt-1" style={{ color: txtSub }}>Outbreaks</div>
                    </div>
                  </div>
                  <div className="mb-3">
                    <p className="text-xs font-mono uppercase tracking-wider mb-2 font-semibold" style={{ color: txtSub }}>Diseases Reported</p>
                    <div className="flex flex-wrap gap-1.5">
                      {[...new Set(selectedRecords.map((r) => r.disease))].map((d) => (
                        <span key={d} className="text-xs font-mono px-2 py-0.5 rounded"
                          style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)', color: '#60a5fa' }}>
                          {d}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-lg border p-3" style={{ background: '#171717', border: `1px solid ${border}` }}>
                    <p className="text-xs font-mono leading-relaxed" style={{ color: txtSub }}>
                      📊 {outbreakCount} outbreak event{outbreakCount !== 1 ? 's' : ''} recorded. {uniqueDiseases} unique disease{uniqueDiseases !== 1 ? 's' : ''} tracked across {[...new Set(selectedRecords.map((r) => r.district))].length} district{selectedRecords.length !== 1 ? 's' : ''}.
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* ─── Data Table ─── */}
        <div>
          <div className="h-px mb-6" style={{ background: `linear-gradient(90deg, transparent, ${border} 50%, transparent)` }} />
          <div className="mb-4 flex items-center justify-between flex-wrap gap-2">
            <p className="font-mono text-xs font-semibold uppercase tracking-widest" style={{ color: txtSub }}>{selectedState} · Outbreak Records</p>
            {loadingRecords && (
              <span className="flex items-center gap-1.5 text-xs font-mono text-blue-500">
                <span className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin inline-block" />Loading…
              </span>
            )}
            {!loadingRecords && selectedRecords.length > 0 && (
              <span className="text-xs font-mono" style={{ color: txtMute }}>{selectedRecords.length} record{selectedRecords.length !== 1 ? 's' : ''} from API</span>
            )}
          </div>

          <div className="overflow-x-auto rounded-xl" style={{ border: `1px solid ${border}` }}>
            <table className="w-full text-sm">
              <thead style={{ background: cardBg2, borderBottom: `1px solid ${border}` }}>
                <tr>
                  {['State','District','Disease','No. of Cases','No. of Deaths','Status','Action'].map((h, i) => (
                    <th key={h} className={`${i >= 3 && i <= 4 ? 'text-right' : i === 5 ? 'text-center' : 'text-left'} px-4 py-3 font-mono font-semibold text-xs uppercase tracking-widest`}
                      style={{ color: txtSub }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loadingRecords ? (
                  <tr><td colSpan={7} className="text-center py-12 font-mono text-sm" style={{ color: txtSub }}>
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                      Fetching live data from API…
                    </div>
                  </td></tr>
                ) : apiError ? (
                  <tr><td colSpan={7} className="text-center py-12 text-red-400 font-mono text-sm">{apiError}</td></tr>
                ) : selectedRecords.length === 0 ? (
                  <tr><td colSpan={7} className="text-center py-12 font-mono text-sm" style={{ color: txtMute }}>
                    No outbreak records found for <strong style={{ color: txtMain }}>{selectedState}</strong>.
                  </td></tr>
                ) : (
                  selectedRecords.map((rec, i) => (
                    <tr key={i} className="transition-colors"
                      style={{
                        borderBottom: `1px solid ${border}`,
                        background: i % 2 === 0 ? bg : cardBg,
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(59,130,246,0.05)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = i % 2 === 0 ? bg : cardBg; }}
                    >
                      <td className="px-4 py-3 font-medium capitalize whitespace-nowrap" style={{ color: txtMain }}>{rec.state}</td>
                      <td className="px-4 py-3 font-mono text-xs whitespace-nowrap" style={{ color: txtSub }}>{rec.district}</td>
                      <td className="px-4 py-3">
                        <span className="inline-block text-xs font-mono font-semibold px-2 py-0.5 rounded whitespace-nowrap"
                          style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)', color: '#a5b4fc' }}>
                          {rec.disease}
                        </span>
                      </td>
                      <td className="text-right px-4 py-3 font-mono font-semibold" style={{ color: txtMain }}>
                        {rec.no_of_cases != null ? rec.no_of_cases.toLocaleString() : '—'}
                      </td>
                      <td className="text-right px-4 py-3 font-mono font-semibold" style={{ color: txtSub }}>
                        {rec.no_deaths != null ? rec.no_deaths.toLocaleString() : '—'}
                      </td>
                      <td className="text-center px-4 py-3"><StatusBadge status={rec.status} /></td>
                      <td className="px-4 py-3 text-xs leading-relaxed max-w-xs" style={{ color: txtMute }}>
                        <span title={rec.action}>{rec.action.length > 180 ? rec.action.slice(0, 180) + '…' : rec.action}</span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </main>
  );
}
