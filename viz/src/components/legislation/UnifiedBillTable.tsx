import { useState, useMemo } from 'react';
import type { VoteModelRow } from '../../types';

interface Props {
  houseRows: VoteModelRow[];
  senateRows: VoteModelRow[];
}

function ProbBar({ prob }: { prob: number }) {
  const pct = Math.round(prob * 100);
  const color = prob > 0.7 ? '#22c55e' : prob > 0.3 ? '#f59e0b' : '#ef4444';
  return (
    <div className="flex items-center gap-2 min-w-[90px]">
      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-7 text-right" style={{ color }}>{pct}%</span>
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const cls =
    verdict === 'PASS' ? 'bg-green-900/60 text-green-300 border-green-700' :
    verdict === 'FAIL' ? 'bg-red-900/60 text-red-300 border-red-700' :
    'bg-yellow-900/60 text-yellow-300 border-yellow-700';
  return (
    <span className={`text-xs font-bold px-1.5 py-0.5 rounded border whitespace-nowrap ${cls}`}>
      {verdict}
    </span>
  );
}

export function UnifiedBillTable({ houseRows, senateRows }: Props) {
  const [domain, setDomain] = useState<string>('All');

  const houseByVar = useMemo(() => Object.fromEntries(houseRows.map(r => [r.variable, r])), [houseRows]);
  const senateByVar = useMemo(() => Object.fromEntries(senateRows.map(r => [r.variable, r])), [senateRows]);

  // Union of all variables
  const allVars = useMemo(() => {
    const vars = new Set([...houseRows.map(r => r.variable), ...senateRows.map(r => r.variable)]);
    return Array.from(vars);
  }, [houseRows, senateRows]);

  const domains = useMemo(() => {
    const d = new Set<string>();
    for (const r of [...houseRows, ...senateRows]) d.add(r.domain);
    return ['All', ...Array.from(d).sort()];
  }, [houseRows, senateRows]);

  const filtered = domain === 'All'
    ? allVars
    : allVars.filter(v => (houseByVar[v] ?? senateByVar[v])?.domain === domain);

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

      {/* Column headers */}
      <div className="hidden md:grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-x-3 items-center px-3 py-1 text-xs text-slate-500 uppercase tracking-widest border-b border-slate-700 mb-1">
        <div>Bill</div>
        <div className="w-24 text-right">House prob</div>
        <div className="w-14 text-right">Verdict</div>
        <div className="w-24 text-right">Senate (Cond)</div>
        <div className="w-14 text-right">Verdict</div>
        <div className="w-24 text-right">Senate (IRV)</div>
      </div>

      <div className="space-y-0.5">
        {filtered.map(variable => {
          const hr = houseByVar[variable];
          const sr = senateByVar[variable];
          const ref = hr ?? sr;
          if (!ref) return null;

          const hVerdict = hr?.verdict ?? '—';
          const scVerdict = sr?.condVerdict ?? '—';
          const disagrees = hVerdict !== '—' && scVerdict !== '—' && hVerdict !== scVerdict;

          return (
            <div
              key={variable}
              className={`flex flex-col md:grid md:grid-cols-[1fr_auto_auto_auto_auto_auto] gap-x-3 items-start md:items-center py-2 px-3 rounded text-sm ${
                disagrees ? 'bg-amber-900/20 border border-amber-800/40' : 'bg-slate-800/60 hover:bg-slate-800'
              }`}
            >
              <div className="min-w-0">
                <span className="text-slate-300">{ref.question}</span>
                <span className="text-xs text-slate-500 ml-2">{ref.domain}</span>
              </div>

              {/* House */}
              <div className="w-24 mt-1 md:mt-0">
                {hr ? <ProbBar prob={hr.probPass!} /> : <span className="text-slate-600 text-xs">—</span>}
              </div>
              <div className="w-14">
                {hVerdict !== '—' ? <VerdictBadge verdict={hVerdict} /> : <span className="text-slate-600 text-xs">—</span>}
              </div>

              {/* Senate Condorcet */}
              <div className="w-24 mt-1 md:mt-0">
                {sr ? <ProbBar prob={sr.condProbPass!} /> : <span className="text-slate-600 text-xs">—</span>}
              </div>
              <div className="w-14">
                {scVerdict !== '—' ? <VerdictBadge verdict={scVerdict} /> : <span className="text-slate-600 text-xs">—</span>}
              </div>

              {/* Senate IRV */}
              <div className="w-24 mt-1 md:mt-0">
                {sr ? <ProbBar prob={sr.irvProbPass!} /> : <span className="text-slate-600 text-xs">—</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
