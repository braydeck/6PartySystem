import { useState } from 'react';
import type { PresidentialElection, PresidentialScenario } from '../types';
import { PresidentialMap } from '../components/presidential/PresidentialMap';
import { CondorcetTable } from '../components/presidential/CondorcetTable';
import { IRVRoundsChart } from '../components/presidential/IRVRoundsChart';
import { IRVSankey } from '../components/presidential/IRVSankey';
import { getBlendColor } from '../constants/parties';

interface Props {
  mixed: PresidentialElection;
  pure:  PresidentialElection;
}

const PRES_LABELS: Record<PresidentialScenario, string> = {
  mixed: 'Mixed (CON/SD)',
  pure:  'Pure (STY)',
};

export function PresidentialTab({ mixed, pure }: Props) {
  const [scenario, setScenario] = useState<PresidentialScenario>('mixed');
  const data = scenario === 'mixed' ? mixed : pure;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">2028 Presidential Election</h2>
        <p className="text-slate-400 text-sm">
          General election results using IRV and Condorcet methods. Mixed scenario includes blended
          coalition candidates; pure scenario uses only the 9 core party types.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {(Object.keys(PRES_LABELS) as PresidentialScenario[]).map(s => (
          <button
            key={s}
            onClick={() => setScenario(s)}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
              scenario === s
                ? 'bg-teal-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {PRES_LABELS[s]}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 px-5 py-4 flex items-center gap-4">
          <div className="text-3xl font-bold" style={{ color: getBlendColor(data.irvWinner) }}>
            {data.irvWinner}
          </div>
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">IRV Winner</div>
            <div className="text-sm text-slate-300">Instant Runoff Voting</div>
          </div>
        </div>
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 px-5 py-4 flex items-center gap-4">
          <div className="text-3xl font-bold" style={{ color: getBlendColor(data.condorcetWinner) }}>
            {data.condorcetWinner}
          </div>
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">Condorcet Winner</div>
            <div className="text-sm text-slate-300">Beats all opponents 1-on-1</div>
          </div>
        </div>
      </div>

      {/* State map — full width */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1">
          State Results (IRV)
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Gradient shows R1 candidate vote share per state. Hover for details.
        </p>
        <PresidentialMap stateWinners={data.irvStateWinners} />
      </div>

      {/* IRV vote flow Sankey */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1">
          IRV Vote Flow
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Each column is one elimination round. Eliminated candidates&apos; votes fan out to the remaining field.
        </p>
        <IRVSankey rounds={data.irvRounds} irvWinner={data.irvWinner} />
      </div>

      {/* IRV rounds + Condorcet table */}
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            IRV Rounds Detail
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Candidates eliminated each round until one clears 50%. Select a round to explore.
          </p>
          <IRVRoundsChart rounds={data.irvRounds} irvWinner={data.irvWinner} />
        </div>

        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Head-to-Head Matchups (Condorcet)
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Every possible pairing. The Condorcet winner beats all opponents.
          </p>
          <CondorcetTable matchups={data.condorcetMatchups} condorcetWinner={data.condorcetWinner} />
        </div>
      </div>
    </div>
  );
}
