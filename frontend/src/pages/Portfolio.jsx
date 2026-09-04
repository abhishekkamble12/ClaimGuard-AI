import { usePortfolioReport, useMerchantProfiles } from '../hooks/usePortfolioData';
import PortfolioKPIs from '../components/portfolio/PortfolioKPIs';
import BusinessImpactCard from '../components/portfolio/BusinessImpactCard';
import MerchantRiskTable from '../components/portfolio/MerchantRiskTable';
import SystemicGapsChart from '../components/portfolio/SystemicGapsChart';
import VelocityTimeline from '../components/portfolio/VelocityTimeline';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import './Portfolio.css';

export default function Portfolio() {
  const { data: report, isLoading: isReportLoading, error: reportError, refetch: refetchReport } = usePortfolioReport();
  const { data: merchants, isLoading: isMerchantsLoading, error: merchantsError, refetch: refetchMerchants } = useMerchantProfiles();

  if (isReportLoading || isMerchantsLoading) {
    return (
      <div className="portfolio-page">
        <LoadingState lines={8} height="28px" />
      </div>
    );
  }

  if (reportError || merchantsError) {
    return (
      <div className="portfolio-page">
        <ErrorState
          message={reportError?.message || merchantsError?.message}
          onRetry={() => {
            refetchReport();
            refetchMerchants();
          }}
        />
      </div>
    );
  }

  return (
    <div className="portfolio-page animate-fade-in">
      <div className="portfolio-header">
        <h2>Merchant Portfolio Risk & Dispute Intelligence</h2>
        <p className="text-secondary text-sm">
          Macro financial recovery tracking, VAMP/VCMP chargeback ratio ceilings, and systemic Indian merchant evidence gaps.
        </p>
      </div>

      {/* Business Impact & VAMP Health Summary Card */}
      <BusinessImpactCard report={report} merchants={merchants || []} />

      {/* Top 4 Financial Recovery KPIs */}
      <PortfolioKPIs report={report} />

      {/* Merchant Cohort Risk Profiles Table */}
      <MerchantRiskTable merchants={merchants || []} />

      {/* Systemic Gaps & Velocity Charts */}
      <div className="portfolio-charts-grid">
        <SystemicGapsChart gaps={report?.top_systemic_evidence_gaps || []} />
        <VelocityTimeline timeline={report?.daily_velocity_timeline || []} />
      </div>
    </div>
  );
}
