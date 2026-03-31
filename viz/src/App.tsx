import { useState } from 'react';
import { PrimaryTab } from './tabs/PrimaryTab';
import { SenateTab } from './tabs/SenateTab';
import { HouseTab } from './tabs/HouseTab';
import { QuizTab } from './tabs/QuizTab';
import { PartiesTab } from './tabs/PartiesTab';
import { BlendedPartiesTab } from './tabs/BlendedPartiesTab';
import { PresidentialTab } from './tabs/PresidentialTab';
import { LegislationTab } from './tabs/LegislationTab';

import primaryData from './data/primary.json';
import primaryStateWinnersData from './data/primaryStateWinners.json';
import presidentialElectionData from './data/presidentialElection.json';
import primarySankeyData from './data/primarySankey.json';
import senateCondorcetData from './data/senateCondorcet.json';
import senateIRVData from './data/senateIRV.json';
import senateVoteModelData from './data/senateVoteModel.json';
import houseSeatsData from './data/houseSeats.json';
import houseVoteModelData from './data/houseVoteModel.json';
import houseStateMapData from './data/houseStateMap.json';
import coalitionProfilesData from './data/coalitionProfiles.json';
import transferMatrixData from './data/transferMatrix.json';
import clusterProfilesData from './data/clusterProfiles.json';
import blendProfilesData from './data/blendProfiles.json';
import quizQuestionsData from './data/quizQuestions.json';

import type {
  PrimaryData, PrimaryStateWinner, SenateSeat, VoteModelRow, HouseSeat,
  HouseStateEntry, CoalitionProfile, TransferMatrix, ClusterProfile,
  QuizQuestion, BlendProfile, PresidentialElection,
  PrimarySankeyData,
} from './types';

const TABS = [
  { id: 'primary',        label: 'Presidential Primary' },
  { id: 'presidential',   label: 'Presidential Election' },
  { id: 'senate',         label: 'Senate' },
  { id: 'house',          label: 'House' },
  { id: 'legislation',    label: 'Legislation' },
  { id: 'blends',         label: 'Blended Parties' },
  { id: 'parties',        label: 'Parties' },
  { id: 'quiz',           label: 'Who Are You?' },
] as const;

type TabId = typeof TABS[number]['id'];

export default function App() {
  const [tab, setTab] = useState<TabId>('primary');

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center gap-3 mb-3">
            <div className="text-xl font-bold text-white">STV 2028</div>
            <div className="text-sm text-slate-500">Proportional Democracy Simulation</div>
          </div>
          <nav className="flex gap-1 overflow-x-auto pb-px">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded text-sm font-medium whitespace-nowrap transition-colors ${
                  tab === t.id
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {tab === 'primary' && (
          <PrimaryTab
            data={primaryData as unknown as PrimaryData}
            stateWinners={primaryStateWinnersData as unknown as Record<string, PrimaryStateWinner>}
            sankey={primarySankeyData as unknown as PrimarySankeyData}
          />
        )}
        {tab === 'presidential' && (
          <PresidentialTab data={presidentialElectionData as unknown as PresidentialElection} />
        )}
        {tab === 'senate' && (
          <SenateTab
            condorcet={senateCondorcetData as SenateSeat[]}
            irv={senateIRVData as SenateSeat[]}
            voteModel={senateVoteModelData as VoteModelRow[]}
          />
        )}
        {tab === 'house' && (
          <HouseTab
            seats={houseSeatsData as HouseSeat[]}
            coalitions={coalitionProfilesData as CoalitionProfile[]}
            transfers={transferMatrixData as unknown as TransferMatrix}
            voteModel={houseVoteModelData as VoteModelRow[]}
            stateMap={houseStateMapData as unknown as Record<string, HouseStateEntry>}
          />
        )}
        {tab === 'legislation' && (
          <LegislationTab
            houseVotes={houseVoteModelData as VoteModelRow[]}
            senateVotes={senateVoteModelData as VoteModelRow[]}
          />
        )}
        {tab === 'blends' && (
          <BlendedPartiesTab profiles={blendProfilesData as unknown as BlendProfile[]} />
        )}
        {tab === 'parties' && (
          <PartiesTab clusters={clusterProfilesData as ClusterProfile[]} />
        )}
        {tab === 'quiz' && (
          <QuizTab
            questions={quizQuestionsData as QuizQuestion[]}
            clusters={clusterProfilesData as ClusterProfile[]}
            houseVotes={houseVoteModelData as VoteModelRow[]}
          />
        )}
      </main>

      <footer className="border-t border-slate-800 mt-12 py-6 text-center text-xs text-slate-600">
        Built on CES 2024 survey data · 10-party STV simulation · 873 House seats · 50 Senate seats
      </footer>
    </div>
  );
}
