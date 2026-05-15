import { useState } from 'react';
import { getBlendColor, F5_ORDER } from '../../constants/parties';
import type { HouseStateEntry } from '../../types';

interface Props {
  stateMap: Record<string, HouseStateEntry>;
}

// Geographic grid [row, col] — approximate US state positions in an 8×12 grid
const STATE_GRID: Record<string, [number, number]> = {
  AK: [0,0],
  WA: [1,2], MT: [1,3], ND: [1,4], MN: [1,5], WI: [1,6], MI: [1,7], NY: [1,9], VT: [1,10], NH: [1,11], ME: [1,12],
  OR: [2,2], ID: [2,3], SD: [2,4], IA: [2,5], IN: [2,6], OH: [2,7], PA: [2,8], NJ: [2,9], MA: [2,10], RI: [2,11],
  CA: [3,1], NV: [3,3], WY: [3,4], NE: [3,5], IL: [3,6], KY: [3,7], WV: [3,8], MD: [3,9], CT: [3,10], DE: [3,11],
             UT: [4,3], CO: [4,4], KS: [4,5], MO: [4,6], TN: [4,7], VA: [4,8], DC: [4,9],
  AZ: [5,2], NM: [5,3], OK: [5,4], AR: [5,5], MS: [5,6], AL: [5,7], NC: [5,8], SC: [5,9],
  HI: [6,1],            TX: [6,3], LA: [6,5],             GA: [6,7], FL: [6,8],
};

// Build FIPS → stateAbbr reverse lookup
const FIPS_TO_ABBR: Record<string, string> = {
  '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE','11':'DC',
  '12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA','20':'KS','21':'KY',
  '22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT',
  '31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH',
  '40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN','48':'TX','49':'UT',
  '50':'VT','51':'VA','53':'WA','54':'WV','55':'WI','56':'WY',
};

const CELL = 72;   // cell size in px
const GAP  = 6;    // gap between cells
const DOT  = 6;    // dot size in px
const DOT_GAP = 1; // gap between dots

export function HouseGridChart({ stateMap }: Props) {
  const [tooltip, setTooltip] = useState<string | null>(null);

  // Build abbr → entry map
  const byAbbr: Record<string, HouseStateEntry> = {};
  for (const [fips, entry] of Object.entries(stateMap)) {
    const abbr = FIPS_TO_ABBR[fips] ?? entry.stateAbbr;
    byAbbr[abbr] = entry;
  }

  // Determine grid bounds
  const maxRow = Math.max(...Object.values(STATE_GRID).map(([r]) => r));
  const maxCol = Math.max(...Object.values(STATE_GRID).map(([, c]) => c));

  const totalW = (maxCol + 1) * (CELL + GAP);
  const totalH = (maxRow + 1) * (CELL + GAP + 14); // +14 for state label below

  const dotsPerRow = Math.floor(CELL / (DOT + DOT_GAP));

  return (
    <div>
      {tooltip && (
        <div className="text-sm text-slate-700 bg-white border border-slate-200 rounded px-3 py-1.5 shadow-sm mb-2 inline-block">
          {tooltip}
        </div>
      )}
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${totalW} ${totalH}`}
          style={{ width: '100%', minWidth: 720 }}
          aria-label="House seat grid chart by state"
        >
          {Object.entries(STATE_GRID).map(([abbr, [row, col]]) => {
            const entry = byAbbr[abbr];
            if (!entry) return null;

            const cx = col * (CELL + GAP);
            const cy = row * (CELL + GAP + 14);

            // Build ordered seat dots: sorted by F5_ORDER, then fill dots
            const sortedParties = Object.entries(entry.seats).sort((a, b) => {
              const rA = F5_ORDER.indexOf(a[0] as typeof F5_ORDER[number]);
              const rB = F5_ORDER.indexOf(b[0] as typeof F5_ORDER[number]);
              return (rA === -1 ? 99 : rA) - (rB === -1 ? 99 : rB);
            });
            const dots: string[] = [];
            for (const [party, count] of sortedParties) {
              for (let i = 0; i < count; i++) dots.push(party);
            }

            const pluralityColor = getBlendColor(entry.pluralityParty);

            return (
              <g
                key={abbr}
                onMouseEnter={() => {
                  const breakdown = sortedParties.map(([p, n]) => `${p}:${n}`).join(' · ');
                  setTooltip(`${abbr} — ${entry.totalSeats} seats (${breakdown})`);
                }}
                onMouseLeave={() => setTooltip(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* State box background */}
                <rect
                  x={cx} y={cy} width={CELL} height={CELL}
                  fill={pluralityColor + '0a'}
                  stroke={pluralityColor + '55'}
                  strokeWidth={1}
                  rx={2}
                />

                {/* Seat dots */}
                {dots.map((party, i) => {
                  const dotCol = i % dotsPerRow;
                  const dotRow = Math.floor(i / dotsPerRow);
                  const dx = cx + dotCol * (DOT + DOT_GAP) + 2;
                  const dy = cy + dotRow * (DOT + DOT_GAP) + 2;
                  if (dy + DOT > cy + CELL) return null; // clip if overflow
                  return (
                    <rect
                      key={i}
                      x={dx} y={dy}
                      width={DOT} height={DOT}
                      fill={getBlendColor(party)}
                      opacity={0.85}
                      rx={1}
                    />
                  );
                })}

                {/* State abbreviation */}
                <text
                  x={cx + CELL / 2}
                  y={cy + CELL + 11}
                  textAnchor="middle"
                  fontSize={9}
                  fontWeight={600}
                  fill={pluralityColor}
                >
                  {abbr}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="text-xs text-slate-500 mt-2 text-center">
        Each square = 1 House seat, colored by party. Ordered left-to-right by Populist Conservatism (F5). Hover for details.
      </p>
    </div>
  );
}
