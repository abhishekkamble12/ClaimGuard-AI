import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { scoreDispute, getDatasetCases, getDatasetCase, getDisputeAudit, batchScoreDisputes } from '../api/client';

/**
 * Fetch the full list of dispute cases from the dataset.
 */
export function useDatasetCases() {
  return useQuery({
    queryKey: ['dataset', 'cases'],
    queryFn: getDatasetCases,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Fetch a single dispute case by ID.
 */
export function useDatasetCase(disputeId) {
  return useQuery({
    queryKey: ['dataset', 'case', disputeId],
    queryFn: () => getDatasetCase(disputeId),
    enabled: !!disputeId,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Score a dispute case. Returns the full scoring result.
 * Uses mutation since scoring has side-effects (audit log).
 */
export function useScoreDispute() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ disputeCase, options }) => scoreDispute(disputeCase, options),
    onSuccess: (data) => {
      queryClient.setQueryData(['dispute', 'score', data.dispute_id], data);
    },
  });
}

/**
 * Get cached scoring result for a dispute.
 */
export function useCachedScore(disputeId) {
  return useQuery({
    queryKey: ['dispute', 'score', disputeId],
    queryFn: () => null,
    enabled: false,
    staleTime: Infinity,
  });
}

/**
 * Fetch audit log for a dispute.
 */
export function useDisputeAudit(disputeId) {
  return useQuery({
    queryKey: ['dispute', 'audit', disputeId],
    queryFn: () => getDisputeAudit(disputeId),
    enabled: !!disputeId,
    retry: false,
  });
}

/**
 * Batch score disputes.
 */
export function useBatchScore() {
  return useMutation({
    mutationFn: ({ disputes, apiKey }) => batchScoreDisputes(disputes, apiKey),
  });
}
