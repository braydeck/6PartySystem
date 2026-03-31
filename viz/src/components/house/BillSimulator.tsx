import { useState, useMemo } from 'react';
import type { VoteModelRow } from '../../types';

interface Props {
  rows: VoteModelRow[];
}

function ProbBar({ prob }: { prob: number }) {
  const pct = Math.round(prob * 100);
  const color = prob > 0.7 ? '#22c55e' : prob > 0.3 ? '#f59e0b' : '#ef4444';
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{pct}%</span>
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const cls =
    verdict === 'PASS' ? 'bg-green-900/60 text-green-300 border-green-700' :
    verdict === 'FAIL' ? 'bg-red-900/60 text-red-300 border-red-700' :
    'bg-yellow-900/60 text-yellow-300 border-yellow-700';
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded border whitespace-nowrap ${cls}`}>
      {verdict}
    </span>
  );
}

export function BillSimulator({ rows }: Props) {
  const [domain, setDomain] = useState<string>('All');
  const domains = useMemo(() => {
    const d = Array.from(new Set(rows.map(r => r.domain))).sort();
    return ['All', ...d];
  }, [rows]);

  const filtered = domain === 'All' ? rows : rows.filter(r => r.domain === domain);

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {domains.map(d => (
          <button
            key={d}
            onClick={() => setDomain(d)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              domain === d
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {d}
          </button>
        ))}
      </div>

      <div className="space-y-1">
        {filtered.map(row => (
          <div
            key={row.variable}
            className="flex items-center gap-3 py-2 px-3 rounded bg-slate-800/60 hover:bg-slate-800"
          >
            <div className="flex-1 text-sm text-slate-300 min-w-0">
              <span>{row.question}</span>
              <span className="text-xs text-slate-500 ml-2">{row.domain}</span>
            </div>
            <ProbBar prob={row.probPass!} />
            <VerdictBadge verdict={row.verdict!} />
          </div>
        ))}
      </div>
    </div>
  );
}
