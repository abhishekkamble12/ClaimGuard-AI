import { useQuery } from '@tanstack/react-query';
import { getHealth, getMetrics, getVersion, getDriftReport } from '../api/client';

/**
 * Service health including model status.
 */
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 60 * 1000,
    staleTime: 30 * 1000,
  });
}

/**
 * Operational metrics: latency, tokens, cost.
 */
export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: getMetrics,
    staleTime: 60 * 1000,
  });
}

/**
 * Model version and registry.
 */
export function useVersion() {
  return useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * PSI drift report.
 */
export function useDriftReport() {
  return useQuery({
    queryKey: ['drift'],
    queryFn: getDriftReport,
    staleTime: 5 * 60 * 1000,
  });
}
