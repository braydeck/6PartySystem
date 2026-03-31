import type { BlendProfile } from '../types';
import { BlendCard } from '../components/parties/BlendCard';

interface Props {
  profiles: BlendProfile[];
}

export function BlendedPartiesTab({ profiles }: Props) {
  const blended = profiles.filter(p => !p.isPure);
  const pure = profiles.filter(p => p.isPure);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Blended Parties</h2>
        <p className="text-slate-400 text-sm max-w-2xl">
          Most Senate races are won by candidates who straddle two parties, attracting
          cross-coalition voters. These "blended" senators share the ideological DNA of
          both component parties but sit in a distinct position on every policy axis.
          Counts show seats won under Condorcet (C) and IRV (I) methods.
        </p>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {blended.map(p => (
          <BlendCard key={p.code} profile={p} />
        ))}
      </div>

      {pure.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-slate-300 mb-3">Pure-Party Senate Winners</h3>
          <p className="text-slate-500 text-sm mb-4">
            These states returned a senator fully aligned with one party — rarer, but it happens.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {pure.map(p => (
              <BlendCard key={p.code} profile={p} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
