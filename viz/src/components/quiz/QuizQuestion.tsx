interface Props {
  question: string;
  domain: string;
  selected: number | null;
  onSelect: (v: number) => void;
}

const OPTIONS = [
  { value: 1, label: 'Strongly Agree' },
  { value: 0.75, label: 'Agree' },
  { value: 0.5, label: 'Neutral' },
  { value: 0.25, label: 'Disagree' },
  { value: 0, label: 'Strongly Disagree' },
];

export function QuizQuestion({ question, domain, selected, onSelect }: Props) {
  return (
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-widest mb-2">{domain}</div>
      <div className="text-xl font-semibold text-slate-900 mb-6 leading-snug">{question}</div>
      <div className="flex flex-col gap-2">
        {OPTIONS.map(opt => (
          <button
            key={opt.value}
            onClick={() => onSelect(opt.value)}
            className={`px-4 py-3 rounded text-left transition-all text-sm font-medium border ${
              selected === opt.value
                ? 'bg-teal-600 border-teal-400 text-white'
                : 'bg-white border-slate-200 text-slate-700 hover:border-teal-400 hover:text-slate-900'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
