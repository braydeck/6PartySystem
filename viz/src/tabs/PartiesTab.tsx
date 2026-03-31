import type { ClusterProfile } from '../types';
import { PartyCard } from '../components/parties/PartyCard';

interface Props {
  clusters: ClusterProfile[];
}

const PARTY_ORDER = ['CON','SD','STY','REF','CTR','LIB','NAT','DSA','PRG'];

export function PartiesTab({ clusters }: Props) {
  const sorted = [...clusters]
    .filter(c => c.party)
    .sort((a, b) => PARTY_ORDER.indexOf(a.party) - PARTY_ORDER.indexOf(b.party));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">The 9 Parties</h2>
        <p className="text-slate-400 text-sm">
          A 10-cluster model of the American electorate, with the Blue Dog remnant (C7) merged
          into adjacent clusters. Each party reflects a distinct ideological constellation
          derived from CES 2024 survey data.
        </p>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {sorted.map(c => (
          <PartyCard key={c.id} cluster={c} />
        ))}
      </div>
    </div>
  );
}
