import { useState } from 'react';
import type { SenateSeat, VoteModelRow, SenateScenario } from '../types';
import { SenateMap } from '../components/senate/SenateMap';
import { VoteModelTable } from '../components/senate/VoteModelTable';
import { getBlendColor } from '../constants/parties';

interface Props {
  condorcetMixed: SenateSeat[];
  irvMixed:       SenateSeat[];
  condorcetPure:  SenateSeat[];
  irvPure:        SenateSeat[];
  voteModel:      VoteModelRow[];
}

const SCENARIO_LABELS: Record<SenateScenario, string> = {
  condMixed: 'Mixed · Condorcet',
  irvMixed:  'Mixed · IRV',
  condPure:  'Pure · Condorcet',
  irvPure:   'Pure · IRV',
};

function SeatSummary({ seats, label }: { seats: SenateSeat[]; label: string }) {
  const counts: Record<string, number> = {};
  for (const s of seats) {
    counts[s.senatorCode] = (counts[s.senatorCode] ?? 0) + 1;
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="bg-slate-800/50 rounded-xl p-4">
      <div className="text-xs text-slate-500 uppercase tracking-widest mb-3">{label} Composition</div>
      <div className="flex flex-wrap gap-2">
        {sorted.map(([code, count]) => {
          const color = getBlendColor(code);
          return (
            <div
              key={code}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-mono"
              style={{ backgroundColor: color + '22', border: `1px solid ${color}55`, color }}
            >
              <span className="font-bold">{code}</span>
              <span className="text-slate-400 font-sans">{count}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SenateTab({ condorcetMixed, irvMixed, condorcetPure, irvPure, voteModel }: Props) {
  const [scenario, setScenario] = useState<SenateScenario>('condMixed');

  const SEAT_MAP: Record<SenateScenario, SenateSeat[]> = {
    condMixed: condorcetMixed,
    irvMixed,
    condPure:  condorcetPure,
    irvPure,
  };
  const activeSeats = SEAT_MAP[scenario];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Senate</h2>
        <p className="text-slate-400 text-sm">
          State-level senate simulation. Mixed scenarios include blended coalition candidates;
          pure scenarios use only the 9 core party types. Condorcet selects the head-to-head
          winner; IRV uses instant runoff elimination.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {(Object.keys(SCENARIO_LABELS) as SenateScenario[]).map(s => (
          <button
            key={s}
            onClick={() => setScenario(s)}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
              scenario === s
                ? 'bg-teal-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {SCENARIO_LABELS[s]}
          </button>
        ))}
      </div>

      <SeatSummary seats={activeSeats} label={SCENARIO_LABELS[scenario]} />

      <div className="bg-slate-800/50 rounded-xl p-4">
        <SenateMap seats={activeSeats} />
      </div>

      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">
          Senate Vote Model — 37 Bills
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Highlighted rows show bills the senate passes but the president vetoes.
        </p>
        <VoteModelTable rows={voteModel} scenario={scenario} />
      </div>
    </div>
  );
}
