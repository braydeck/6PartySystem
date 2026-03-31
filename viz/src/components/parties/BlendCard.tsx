import type { BlendProfile } from '../../types';
import { getBlendColor, PARTY_COLORS, PARTY_NAMES } from '../../constants/parties';
import { FactorBar } from '../shared/FactorBar';

interface Props {
  profile: BlendProfile;
}

export function BlendCard({ profile }: Props) {
  const color = getBlendColor(profile.code);
  const parts = profile.code.split('/');

  return (
    <div
      className="rounded-xl border overflow-hidden flex flex-col"
      style={{ borderColor: color + '55' }}
    >
      {/* Header */}
      <div className="px-5 py-4" style={{ backgroundColor: color + '22' }}>
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-lg font-bold text-white font-mono">{profile.code}</div>
            <div className="flex gap-1 mt-1">
              {parts.map(p => (
                <span
                  key={p}
                  className="text-xs font-semibold px-2 py-0.5 rounded"
                  style={{ backgroundColor: (PARTY_COLORS[p] ?? '#6b7280') + '44', color: PARTY_COLORS[p] ?? '#94a3b8' }}
                >
                  {PARTY_NAMES[p] ?? p}
                </span>
              ))}
            </div>
          </div>
          <div className="text-right shrink-0">
            <div className="text-xs text-slate-500">Senate seats</div>
            <div className="flex gap-2 text-sm font-semibold mt-0.5">
              <span style={{ color }} title="Condorcet">{profile.seatsCond}C</span>
              <span className="text-slate-600">/</span>
              <span style={{ color: color + 'aa' }} title="IRV">{profile.seatsIRV}I</span>
            </div>
          </div>
        </div>
      </div>

      <div className="px-5 py-4 flex-1 flex flex-col gap-3">
        {/* Factor bars */}
        <div>
          {(['F1','F2','F3','F4','F5'] as const).map(f => (
            <FactorBar key={f} factor={f} value={(profile as any)[f]} />
          ))}
        </div>

        {/* Key positions */}
        {profile.keyPositions.length > 0 && (
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest mb-2">Key Positions</div>
            <ul className="space-y-1">
              {profile.keyPositions.map((pos, i) => (
                <li key={i} className="text-xs text-slate-300 flex items-start gap-1.5">
                  <span
                    className="mt-0.5 shrink-0"
                    style={{ color: pos.direction === 'supports' ? '#22c55e' : '#ef4444' }}
                  >
                    {pos.direction === 'supports' ? '▲' : '▼'}
                  </span>
                  <span>
                    {pos.question}
                    <span className="text-slate-500 ml-1">({Math.round(pos.pct)}% support)</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
