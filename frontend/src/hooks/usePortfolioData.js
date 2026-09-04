import { useQuery } from '@tanstack/react-query';
import { getPortfolioReport, getMerchantProfiles } from '../api/client';

/**
 * Portfolio-level analytics report.
 */
export function usePortfolioReport() {
  return useQuery({
    queryKey: ['portfolio', 'report'],
    queryFn: getPortfolioReport,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * All merchant risk profiles with VAMP tiers.
 */
export function useMerchantProfiles() {
  return useQuery({
    queryKey: ['portfolio', 'merchants'],
    queryFn: getMerchantProfiles,
    staleTime: 5 * 60 * 1000,
  });
}
