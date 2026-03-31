import { useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { PARTY_COLORS, PARTY_NAMES } from '../../constants/parties';
import type { HouseStateEntry } from '../../types';

const GEO_URL = './topojson/states-10m.json';

interface Props {
  stateMap: Record<string, HouseStateEntry>;
}

export function HouseMap({ stateMap }: Props) {
  const [tooltip, setTooltip] = useState<string | null>(null);

  return (
    <div>
      <div className="relative">
        {tooltip && (
          <div className="absolute top-2 left-2 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm z-10 pointer-events-none max-w-sm">
            {tooltip}
          </div>
        )}
        <ComposableMap projection="geoAlbersUsa" style={{ width: '100%', height: 'auto' }}>
          <defs>
            {Object.entries(stateMap).map(([fips, entry]) => {
              const sorted = Object.entries(entry.seats).sort((a, b) => b[1] - a[1]);
              const total = entry.totalSeats;
              let cum = 0;
              const stops: { offset: number; color: string }[] = [];
              for (const [party, seats] of sorted) {
                const color = PARTY_COLORS[party] ?? '#6b7280';
                stops.push({ offset: cum * 100, color });
                cum += seats / total;
                stops.push({ offset: cum * 100, color });
              }
              return (
                <linearGradient key={fips} id={`hgrad-${fips}`} x1="0%" x2="100%" y1="0%" y2="0%">
                  {stops.map((s, i) => (
                    <stop key={i} offset={`${s.offset.toFixed(1)}%`} stopColor={s.color} stopOpacity={0.85} />
                  ))}
                </linearGradient>
              );
            })}
          </defs>
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map(geo => {
                const fips = geo.id?.toString().padStart(2, '0') ?? '';
                const entry = stateMap[fips];
                const fill = entry ? `url(#hgrad-${fips})` : '#1e293b';

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke="#0f172a"
                    strokeWidth={0.8}
                    style={{
                      default: { outline: 'none' },
                      hover:   { outline: 'none', opacity: 0.8 },
                      pressed: { outline: 'none' },
                    }}
                    onMouseEnter={() => {
                      if (entry) {
                        const breakdown = Object.entries(entry.seats)
                          .sort((a, b) => b[1] - a[1])
                          .map(([p, n]) => `${p}:${n}`)
                          .join(' · ');
                        setTooltip(
                          `${entry.stateAbbr} — ${entry.totalSeats} seats (${breakdown})`
                        );
                      }
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>
      </div>

      <div className="flex flex-wrap gap-2 mt-3 justify-center">
        {Object.entries(PARTY_COLORS).map(([party, color]) => (
          <div
            key={party}
            className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-semibold"
            style={{ backgroundColor: color + '22', border: `1px solid ${color}55`, color }}
          >
            {PARTY_NAMES[party]}
          </div>
        ))}
      </div>

      <p className="text-xs text-slate-600 mt-2 text-center">
        Gradient shows seat share per party. Hover for full breakdown.
      </p>
    </div>
  );
}
