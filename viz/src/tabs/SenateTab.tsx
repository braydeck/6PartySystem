import type { SenateSeat, VoteModelRow } from '../types';
import { SenateMap } from '../components/senate/SenateMap';
import { VoteModelTable } from '../components/senate/VoteModelTable';
import { getBlendColor } from '../constants/parties';

interface Props {
  condorcet: SenateSeat[];
  irv: SenateSeat[];
  voteModel: VoteModelRow[];
}

function SeatSummary({ seats, label }: { seats: SenateSeat[]; label: string }) {
  // Count by full senator_code (preserve blended identities)
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

export function SenateTab({ condorcet, irv, voteModel }: Props) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Senate</h2>
        <p className="text-slate-400 text-sm">
          State-level senate simulation. Most senators are blended — they bridge two parties
          to win over cross-coalition voters. Condorcet selects the candidate who beats all
          others head-to-head; IRV uses instant runoff elimination.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <SeatSummary seats={condorcet} label="Condorcet" />
        <SeatSummary seats={irv} label="IRV" />
      </div>

      <div className="bg-slate-800/50 rounded-xl p-4">
        <SenateMap condorcet={condorcet} irv={irv} />
      </div>

      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">
          Senate Vote Model — 37 Bills
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Highlighted rows show bills where Condorcet and IRV chambers would vote differently.
        </p>
        <VoteModelTable rows={voteModel} />
      </div>
    </div>
  );
}
