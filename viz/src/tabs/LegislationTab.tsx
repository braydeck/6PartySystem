import type { VoteModelRow } from '../types';
import { UnifiedBillTable } from '../components/legislation/UnifiedBillTable';

interface Props {
  houseVotes: VoteModelRow[];
  senateVotes: VoteModelRow[];
}

export function LegislationTab({ houseVotes, senateVotes }: Props) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Legislation</h2>
        <p className="text-slate-400 text-sm">
          Probability of passage across both chambers. Highlighted rows show where House and Senate
          (Condorcet composition) would vote differently.
        </p>
      </div>

      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Bill Passage Probability
        </h3>
        <UnifiedBillTable houseRows={houseVotes} senateRows={senateVotes} />
      </div>
    </div>
  );
}
