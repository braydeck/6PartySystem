import { getBlendColor } from '../../constants/parties';
import type { CondorcetMatchup } from '../../types';

interface Props {
  matchups: CondorcetMatchup[];
  condorcetWinner: string;
}

export function CondorcetTable({ matchups, condorcetWinner }: Props) {
  // Sort by lock_order (already sorted in data) or by aWinsPct descending
  const sorted = [...matchups].sort((a, b) => {
    // Put matchups involving the condorcet winner first
    const aInvolvesWinner = a.candidateA === condorcetWinner || a.candidateB === condorcetWinner;
    const bInvolvesWinner = b.candidateA === condorcetWinner || b.candidateB === condorcetWinner;
    if (aInvolvesWinner && !bInvolvesWinner) return -1;
    if (!aInvolvesWinner && bInvolvesWinner) return 1;
    return b.margin - a.margin;
  });

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <div className="text-xs text-slate-500 uppercase tracking-widest">Condorcet Winner</div>
        <div
          className="text-sm font-bold px-3 py-1 rounded"
          style={{
            backgroundColor: getBlendColor(condorcetWinner) + '33',
            border: `1px solid ${getBlendColor(condorcetWinner)}88`,
            color: getBlendColor(condorcetWinner),
          }}
        >
          {condorcetWinner}
        </div>
        <div className="text-xs text-slate-500">beats all opponents head-to-head</div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-700">
              <th className="text-left py-2 pr-3">Candidate A</th>
              <th className="text-left py-2 pr-3">Candidate B</th>
              <th className="text-left py-2 pr-3">Winner</th>
              <th className="text-right py-2 pr-3">A wins %</th>
              <th className="text-right py-2">Margin</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => {
              const isWinnerRow = m.candidateA === condorcetWinner || m.candidateB === condorcetWinner;
              return (
                <tr
                  key={i}
                  className={`border-b border-slate-800 ${isWinnerRow ? 'bg-slate-800/40' : ''}`}
                >
                  <td className="py-2 pr-3 font-mono" style={{ color: getBlendColor(m.candidateA) }}>
                    {m.candidateA}
                  </td>
                  <td className="py-2 pr-3 font-mono" style={{ color: getBlendColor(m.candidateB) }}>
                    {m.candidateB}
                  </td>
                  <td className="py-2 pr-3">
                    <span
                      className="font-bold px-2 py-0.5 rounded text-xs"
                      style={{
                        backgroundColor: getBlendColor(m.winner) + '33',
                        color: getBlendColor(m.winner),
                      }}
                    >
                      {m.winner}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-300">
                    {m.aWinsPct.toFixed(1)}%
                  </td>
                  <td className="py-2 text-right font-mono text-slate-400">
                    {m.margin.toFixed(2)}pp
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
