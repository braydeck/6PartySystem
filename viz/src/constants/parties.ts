export const PARTY_COLORS: Record<string, string> = {
  CON: '#DC2626',
  SD:  '#2563EB',
  STY: '#7C3AED',
  REF: '#EA580C',
  CTR: '#0D9488',
  LIB: '#16A34A',
  NAT: '#BE123C',
  DSA: '#DB2777',
  PRG: '#4F46E5',
};

export const PARTY_NAMES: Record<string, string> = {
  CON: 'Conservative',
  SD:  'Social Democrat',
  STY: 'Solidarity',
  REF: 'Reform',
  CTR: 'Center',
  LIB: 'Liberal',
  NAT: 'Nationalist',
  DSA: 'Democratic Socialists',
  PRG: 'Progressive',
};

export const CLUSTER_TO_PARTY: Record<string, string> = {
  '0': 'CON',
  '1': 'SD',
  '2': 'STY',
  '3': 'NAT',
  '4': 'LIB',
  '5': 'REF',
  '6': 'CTR',
  '8': 'DSA',
  '9': 'PRG',
};

export const PARTY_TO_CLUSTER: Record<string, string> = Object.fromEntries(
  Object.entries(CLUSTER_TO_PARTY).map(([k, v]) => [v, k])
);

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [128, 128, 128];
}

function blendHex(hex1: string, hex2: string): string {
  const [r1, g1, b1] = hexToRgb(hex1);
  const [r2, g2, b2] = hexToRgb(hex2);
  const r = Math.round((r1 + r2) / 2);
  const g = Math.round((g1 + g2) / 2);
  const b = Math.round((b1 + b2) / 2);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/** Returns color for a party code, blending if composite (e.g. "CON/STY") */
export function getBlendColor(code: string): string {
  if (!code) return '#6b7280';
  const parts = code.split('/');
  if (parts.length === 1) return PARTY_COLORS[parts[0]] ?? '#6b7280';
  const c1 = PARTY_COLORS[parts[0]] ?? '#6b7280';
  const c2 = PARTY_COLORS[parts[1]] ?? '#6b7280';
  return blendHex(c1, c2);
}

/** Given a senator_code like "CON/STY" or "CON", return the primary party code */
export function getPrimaryParty(code: string): string {
  if (!code) return '';
  return code.split('/')[0];
}

/** Use blend color for composite codes, pure party color for singles */
export function getPartyColor(code: string): string {
  return getBlendColor(code);
}

export const FACTOR_LABELS: Record<string, string> = {
  F1: 'Security & Order',
  F2: 'Electoral Skepticism',
  F3: 'Government Distrust',
  F4: 'Religious Traditionalism',
  F5: 'Populist Conservatism',
};
