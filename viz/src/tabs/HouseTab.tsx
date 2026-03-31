import type { HouseSeat, CoalitionProfile, TransferMatrix, VoteModelRow, HouseStateEntry } from '../types';
import { SeatDistributionBar } from '../components/house/SeatDistributionBar';
import { IdeologicalConstellation } from '../components/house/IdeologicalConstellation';
import { BillSimulator } from '../components/house/BillSimulator';
import { HouseMap } from '../components/house/HouseMap';

interface Props {
  seats: HouseSeat[];
  coalitions: CoalitionProfile[];
  transfers: TransferMatrix;
  voteModel: VoteModelRow[];
  stateMap: Record<string, HouseStateEntry>;
}

export function HouseTab({ seats, coalitions, transfers, voteModel, stateMap }: Props) {
  const totalSeats = seats.reduce((s, r) => s + r.national, 0);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">House of Representatives</h2>
        <p className="text-slate-400 text-sm">
          {totalSeats} seats allocated via STV proportional representation across 3 district types.
        </p>
      </div>

      {/* House map — full width */}
      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
          State Plurality Map
        </h3>
        <HouseMap stateMap={stateMap} />
      </div>

      <div className="bg-slate-800/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Seat Distribution by District Type
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Sorted by national seat total. Darker = urban, medium = suburban, lighter = rural.
        </p>
        <SeatDistributionBar seats={seats} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Ideological Constellation
          </h3>
          <IdeologicalConstellation coalitions={coalitions} transfers={transfers} />
        </div>

        <div className="bg-slate-800/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Bill Simulator
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Probability of passage based on the full House seat composition.
          </p>
          <BillSimulator rows={voteModel} />
        </div>
      </div>
    </div>
  );
}
