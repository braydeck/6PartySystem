import { useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { getBlendColor } from '../../constants/parties';
import type { PresidentialStateWinner } from '../../types';

const GEO_URL = './topojson/states-10m.json';

interface Props {
  stateWinners: Record<string, PresidentialStateWinner>;
}

export function PresidentialMap({ stateWinners }: Props) {
  const [tooltip, setTooltip] = useState<string | null>(null);

  // Tally winner counts for legend
  const tally: Record<string, number> = {};
  for (const [, s] of Object.entries(stateWinners)) {
    tally[s.winner] = (tally[s.winner] ?? 0) + 1;
  }
  const legendEntries = Object.entries(tally).sort((a, b) => b[1] - a[1]);

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
            {Object.entries(stateWinners).map(([fips, sw]) => {
              const sorted = Object.entries(sw.shares).sort((a, b) => b[1] - a[1]);
              let cum = 0;
              const stops: { offset: number; color: string }[] = [];
              for (const [code, share] of sorted) {
                const color = getBlendColor(code);
                stops.push({ offset: cum * 100, color });
                cum += share;
                stops.push({ offset: cum * 100, color });
              }
              return (
                <linearGradient key={fips} id={`egrad-${fips}`} x1="0%" x2="100%" y1="0%" y2="0%">
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
                const entry = stateWinners[fips];
                const fill = entry ? `url(#egrad-${fips})` : '#1e293b';

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke="#0f172a"
                    strokeWidth={0.8}
                    style={{
                      default: { outline: 'none' },
                      hover:   { outline: 'none', opacity: 0.85 },
                      pressed: { outline: 'none' },
                    }}
                    onMouseEnter={() => {
                      if (entry) {
                        const shareStr = Object.entries(entry.shares)
                          .sort((a, b) => b[1] - a[1])
                          .map(([c, v]) => `${c}: ${Math.round(v * 100)}%`)
                          .join(' · ');
                        setTooltip(`${entry.stateAbbr}: ${entry.winner} wins · ${shareStr} (n=${entry.nRespondents})`);
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
        {legendEntries.map(([code, count]) => (
          <div
            key={code}
            className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-semibold"
            style={{
              backgroundColor: getBlendColor(code) + '33',
              border: `1px solid ${getBlendColor(code)}88`,
              color: getBlendColor(code),
            }}
          >
            {code}
            <span className="bg-slate-700 text-slate-300 rounded px-1">{count} states</span>
          </div>
        ))}
      </div>

      <p className="text-xs text-slate-600 mt-2 text-center">
        IRV winner per state. Gradient shows R1 candidate share. Hover for details.
      </p>
    </div>
  );
}
