import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import type { CoalitionProfile, TransferMatrix } from '../../types';
import { PARTY_COLORS, PARTY_NAMES } from '../../constants/parties';

interface Props {
  coalitions: CoalitionProfile[];
  transfers: TransferMatrix;
}

const PARTY_ORDER = ['CON','SD','STY','REF','CTR','LIB','NAT','DSA','PRG'];
const THRESHOLD = 1.5; // transfer affinity threshold for drawing links

export function IdeologicalConstellation({ coalitions, transfers }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const W = 520, H = 420;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const nodes = coalitions
      .filter(c => PARTY_ORDER.includes(c.type))
      .map(c => ({
        id: c.type,
        label: PARTY_NAMES[c.type] ?? c.type,
        seats: c.seatsHouse,
        F1: c.F1,
        F5: c.F5,
        color: PARTY_COLORS[c.type] ?? '#6b7280',
      }));

    // Build links from transfer matrix
    const links: { source: string; target: string; value: number }[] = [];
    const partyRowKeys = Object.keys(transfers.matrix);
    for (const src of partyRowKeys) {
      const srcCode = mapMatrixKeyToParty(src);
      if (!srcCode) continue;
      for (const [tgt, val] of Object.entries(transfers.matrix[src] ?? {})) {
        if (val < THRESHOLD) continue;
        const tgtCode = mapMatrixKeyToParty(tgt);
        if (!tgtCode || tgtCode === srcCode) continue;
        // Avoid duplicate links
        if (links.some(l =>
          (l.source === srcCode && l.target === tgtCode) ||
          (l.source === tgtCode && l.target === srcCode)
        )) continue;
        links.push({ source: srcCode, target: tgtCode, value: val });
      }
    }

    const maxSeats = d3.max(nodes, d => d.seats) ?? 1;
    const rScale = d3.scaleSqrt().domain([0, maxSeats]).range([8, 40]);
    const xScale = d3.scaleLinear().domain([-1.5, 1.0]).range([60, W - 60]);
    const yScale = d3.scaleLinear().domain([-1.2, 1.7]).range([H - 60, 60]);

    const sim = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).strength(0.05))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('x', d3.forceX((d: any) => xScale(d.F1)).strength(0.4))
      .force('y', d3.forceY((d: any) => yScale(d.F5)).strength(0.4))
      .force('collide', d3.forceCollide((d: any) => rScale(d.seats) + 6))
      .stop();

    for (let i = 0; i < 200; i++) sim.tick();

    // Draw links
    const linkSel = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('x1', (d: any) => (d.source as any).x)
      .attr('y1', (d: any) => (d.source as any).y)
      .attr('x2', (d: any) => (d.target as any).x)
      .attr('y2', (d: any) => (d.target as any).y)
      .attr('stroke', '#475569')
      .attr('stroke-width', (d) => Math.max(0.5, d.value / 2))
      .attr('opacity', 0.5);

    // Draw nodes
    svg.append('g')
      .selectAll('circle')
      .data(nodes as any)
      .enter()
      .append('circle')
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', (d: any) => rScale(d.seats))
      .attr('fill', (d: any) => d.color + '99')
      .attr('stroke', (d: any) => d.color)
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseenter', function (_event, d: any) {
        d3.select(this).attr('fill', d.color + 'cc');
        // highlight connected links
        linkSel
          .attr('opacity', (l: any) =>
            (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1
          )
          .attr('stroke', (l: any) =>
            (l.source.id === d.id || l.target.id === d.id) ? '#94a3b8' : '#475569'
          );
      })
      .on('mouseleave', function (_event, d: any) {
        d3.select(this).attr('fill', d.color + '99');
        linkSel.attr('opacity', 0.5).attr('stroke', '#475569');
      });

    // Labels
    svg.append('g')
      .selectAll('text')
      .data(nodes as any)
      .enter()
      .append('text')
      .attr('x', (d: any) => d.x)
      .attr('y', (d: any) => d.y + 4)
      .attr('text-anchor', 'middle')
      .style('fill', '#f1f5f9')
      .style('font-size', '11px')
      .style('font-weight', '600')
      .style('pointer-events', 'none')
      .text((d: any) => d.id);

    // Axis labels
    svg.append('text')
      .attr('x', W / 2).attr('y', H - 4)
      .attr('text-anchor', 'middle')
      .style('fill', '#475569').style('font-size', '11px')
      .text('← Liberal on Security & Order   |   Conservative →');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -H / 2).attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', '#475569').style('font-size', '11px')
      .text('← Cosmopolitan   |   Populist →');

  }, [coalitions, transfers]);

  return (
    <div>
      <svg ref={svgRef} width={W} height={H} style={{ maxWidth: '100%' }} />
      <p className="text-xs text-slate-500 mt-1 text-center">
        Node size = House seats. Lines = voter transfer affinity &gt; {THRESHOLD}. Hover to highlight connections.
      </p>
    </div>
  );
}

function mapMatrixKeyToParty(key: string): string | null {
  const map: Record<string, string> = {
    'C0 Conservative': 'CON',
    'C1 Social Democrat': 'SD',
    'C2 Solidarity': 'STY',
    'C3 Nationalist': 'NAT',
    'C4 Liberal': 'LIB',
    'C5 Reform': 'REF',
    'C6 Center': 'CTR',
    'C8 DSA': 'DSA',
    'C9 Progressive': 'PRG',
  };
  return map[key] ?? null;
}
