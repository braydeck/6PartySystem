import type { QuizQuestion } from '../types';

const ACTIVE_CLUSTERS = ['0','1','2','3','4','5','6','8','9'];
const FACTORS = ['F1','F2','F3','F4','F5'];

interface ScoreResult {
  clusterId: string;
  score: number;
}

export function scoreQuiz(
  questions: QuizQuestion[],
  answers: Record<number, number>
): ScoreResult[] {
  const clusterScores: Record<string, number[]> = {};
  for (const cid of ACTIVE_CLUSTERS) {
    clusterScores[cid] = [];
  }

  // Group questions by factor
  const byFactor: Record<string, QuizQuestion[]> = {};
  for (const q of questions) {
    if (!byFactor[q.factor]) byFactor[q.factor] = [];
    byFactor[q.factor].push(q);
  }

  // Per-factor alignment per cluster
  for (const cid of ACTIVE_CLUSTERS) {
    let totalAlignment = 0;
    for (const factor of FACTORS) {
      const qs = byFactor[factor] ?? [];
      if (qs.length === 0) continue;
      let factorAlignment = 0;
      for (const q of qs) {
        const qIdx = questions.indexOf(q);
        const userScore = answers[qIdx] ?? 0.5;
        const clusterMean = q.clusterSupport[cid] ?? 0.5;
        factorAlignment += 1 - Math.abs(userScore - clusterMean);
      }
      totalAlignment += factorAlignment / qs.length;
    }
    clusterScores[cid] = [totalAlignment];
  }

  return ACTIVE_CLUSTERS.map(cid => ({
    clusterId: cid,
    score: clusterScores[cid][0] ?? 0,
  })).sort((a, b) => b.score - a.score);
}
