import { useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { getBlendColor } from '../../constants/parties';
import type { SenateSeat } from '../../types';

const GEO_URL = './topojson/states-10m.json';

interface Props {
  condorcet: SenateSeat[];
  irv: SenateSeat[];
}

export function SenateMap({ condorcet, irv }: Props) {
  const [mode, setMode] = useState<'condorcet' | 'irv'>('condorcet');
  const [tooltip, setTooltip] = useState<string | null>(null);

  const seats = mode === 'condorcet' ? condorcet : irv;
  const seatByFips = Object.fromEntries(seats.map(s => [s.stateFips, s]));

  return (
    <div>
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setMode('condorcet')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === 'condorcet'
              ? 'bg-teal-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          Condorcet
        </button>
        <button
          onClick={() => setMode('irv')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === 'irv'
              ? 'bg-teal-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          IRV
        </button>
      </div>

      <div className="relative">
        {tooltip && (
          <div className="absolute top-2 left-2 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm z-10 pointer-events-none max-w-xs">
            {tooltip}
          </div>
        )}
        <ComposableMap projection="geoAlbersUsa" style={{ width: '100%', height: 'auto' }}>
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map(geo => {
                const fips = geo.id?.toString().padStart(2, '0') ?? '';
                const seat = seatByFips[fips];
                const fill = seat
                  ? getBlendColor(seat.senatorCode) + 'cc'
                  : '#1e293b';

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke="#334155"
                    strokeWidth={1}
                    style={{
                      default: { outline: 'none' },
                      hover:   { outline: 'none', opacity: 0.8 },
                      pressed: { outline: 'none' },
                    }}
                    onMouseEnter={() => {
                      if (seat) {
                        setTooltip(`${seat.stateAbbr}: ${seat.senatorLabel} (${seat.senatorType})`);
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

      <p className="text-xs text-slate-500 mt-2 text-center">
        {mode === 'condorcet' ? 'Condorcet (head-to-head winner)' : 'IRV (instant runoff)'}
        {' '}— blended senators shown as interpolated colors · hover for details
      </p>
    </div>
  );
}
