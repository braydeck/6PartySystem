import { useState } from 'react';
import type { PrimaryData, PrimaryStateWinner, PrimarySankeyData } from '../types';
import { EliminationWaterfall } from '../components/primary/EliminationWaterfall';
import { IdeologicalScatter } from '../components/primary/IdeologicalScatter';
import { PrimaryStateMap } from '../components/primary/PrimaryStateMap';
import { StagedSankey } from '../components/primary/StagedSankey';

interface Props {
  data: PrimaryData;
  stateWinners: Record<string, PrimaryStateWinner>;
  sankey: PrimarySankeyData;
}

export function PrimaryTab({ data, stateWinners, sankey }: Props) {
  const [stageIdx, setStageIdx] = useState(0);
  const stage = data.stagesOrder[stageIdx];
  const quota = data.quotaByStage[stage] ?? 0;
  const stageLabel = data.stageLabels[stage] ?? stage;

  const activeCandidates = data.candidates.filter(c =>
    data.stagesOrder.some(s => (c.stages[s]?.votePct ?? 0) > 0)
  );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">2028 Presidential Primary</h2>
        <p className="text-slate-400 text-sm">
          A 4-round STV simulation across regional pods — watch as candidates consolidate
          from a crowded field to the final four survivors. Quota = {quota.toFixed(0)} votes.
        </p>
      </div>

      {/* Stage selector */}
      <div className="flex flex-wrap gap-2">
        {data.stagesOrder.map((s, i) => (
          <button
            key={s}
            onClick={() => setStageIdx(i)}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              stageIdx === i
                ? 'bg-amber-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {data.stageLabels[s] ?? s}
          </button>
        ))}
      </div>

      {/* State map — full width */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1">
          State Winners by Stage
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          States light up as their pod votes. Color = IRV winner in that state's race.
        </p>
        <PrimaryStateMap stateWinners={stateWinners} stage={stage} />
      </div>

      {/* Waterfall + scatter side by side */}
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            National Vote Share — {stageLabel}
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Yellow line = quota threshold. Red border = eliminated this round.
          </p>
          <EliminationWaterfall
            candidates={activeCandidates}
            stage={stage}
            quota={quota}
          />
        </div>

        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Ideological Positions
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Bubble size = vote share. Opacity = Religious Traditionalism (F4).
            Outlined = eliminated.
          </p>
          <IdeologicalScatter candidates={activeCandidates} stage={stage} />
        </div>
      </div>

      {/* Staged Sankey — elimination flow */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1">
          Vote Transfer Flows
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          All 20 candidates start at left. Each column is an elimination round.
          Link width = vote share flowing between stages.
        </p>
        <StagedSankey data={sankey} />
      </div>

      {/* Stage summary cards */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
          How the Primary Unfolds
        </h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          {data.stagesOrder.map((s, i) => {
            const survivors = data.candidates.filter(c =>
              ['surviving', 'elected'].includes(c.stages[s]?.status ?? '')
            );
            const eliminated = data.candidates.filter(c =>
              c.stages[s]?.status === 'eliminated_this_round'
            );
            return (
              <div
                key={s}
                className={`rounded-lg p-3 border cursor-pointer transition-colors ${
                  stageIdx === i
                    ? 'border-amber-600/60 bg-amber-900/20'
                    : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                }`}
                onClick={() => setStageIdx(i)}
              >
                <div className="text-xs text-amber-500 font-semibold mb-1">Stage {i + 1}</div>
                <div className="font-medium text-white text-xs mb-2">{data.stageLabels[s]}</div>
                <div className="text-xs text-slate-400">
                  {survivors.length} surviving, {eliminated.length} eliminated
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
