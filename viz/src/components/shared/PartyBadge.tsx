import { getPartyColor } from '../../constants/parties';

interface Props {
  code: string;
  size?: 'sm' | 'md';
}

export function PartyBadge({ code, size = 'md' }: Props) {
  const color = getPartyColor(code);
  const px = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm';
  return (
    <span
      className={`inline-block rounded font-semibold ${px}`}
      style={{ backgroundColor: color + '33', color, border: `1px solid ${color}88` }}
    >
      {code}
    </span>
  );
}
