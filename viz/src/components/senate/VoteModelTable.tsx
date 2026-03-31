import { useMemo, useState } from 'react';
import type { VoteModelRow } from '../../types';

interface Props {
  rows: VoteModelRow[];
}

function VerdictBadge({ verdict }: { verdict: string; prob?: number }) {
  const color =
    verdict === 'PASS' ? 'bg-green-900 text-green-300 border-green-700' :
    verdict === 'FAIL' ? 'bg-red-900 text-red-300 border-red-700' :
    'bg-yellow-900 text-yellow-300 border-yellow-700';
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${color}`}>
      {verdict}
    </span>
  );
}

export function VoteModelTable({ rows }: Props) {
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
                ? 'bg-teal-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {d}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-2 pr-4 text-slate-400 font-medium">Bill</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium whitespace-nowrap">Condorcet</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium whitespace-nowrap">IRV</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(row => {
              const differs = row.condVerdict !== row.irvVerdict;
              return (
                <tr
                  key={row.variable}
                  className={`border-b border-slate-800 ${differs ? 'bg-yellow-900/10' : ''}`}
                >
                  <td className="py-2 pr-4 text-slate-300">
                    <div>{row.question}</div>
                    <div className="text-xs text-slate-500">{row.domain}</div>
                  </td>
                  <td className="py-2 px-3 text-center">
                    <VerdictBadge verdict={row.condVerdict!} prob={row.condProbPass!} />
                  </td>
                  <td className="py-2 px-3 text-center">
                    <VerdictBadge verdict={row.irvVerdict!} prob={row.irvProbPass!} />
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
