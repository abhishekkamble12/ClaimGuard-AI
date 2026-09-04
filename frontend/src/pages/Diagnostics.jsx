import { useHealth, useMetrics, useVersion, useDriftReport } from '../hooks/useModelHealth';
import ModelCard from '../components/diagnostics/ModelCard';
import ModelComparison from '../components/diagnostics/ModelComparison';
import DriftMonitor from '../components/diagnostics/DriftMonitor';
import LLMTelemetry from '../components/diagnostics/LLMTelemetry';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import './Diagnostics.css';

export default function Diagnostics() {
  const { data: healthData, isLoading: isHealthLoading, error: healthError, refetch: refetchHealth } = useHealth();
  const { data: metricsData, isLoading: isMetricsLoading } = useMetrics();
  const { data: versionData } = useVersion();
  const { data: driftData } = useDriftReport();

  if (isHealthLoading || isMetricsLoading) {
    return (
      <div className="diagnostics-page">
        <LoadingState lines={8} height="28px" />
      </div>
    );
  }

  if (healthError) {
    return (
      <div className="diagnostics-page">
        <ErrorState message={healthError.message} onRetry={refetchHealth} />
      </div>
    );
  }

  return (
    <div className="diagnostics-page animate-fade-in">
      <div className="diagnostics-header">
        <h2>ML Architecture & Model Registry Diagnostics</h2>
        <p className="text-secondary text-sm">
          Supervised ensemble validation, Brier score calibration, PSI drift tracking, and live inference telemetry.
        </p>
      </div>

      {/* Active Model Overview KPIs */}
      <ModelCard healthData={healthData} versionData={versionData} />

      {/* Model Benchmark Comparison Table */}
      <ModelComparison />

      {/* PSI Concept Drift & Stability Monitor */}
      <DriftMonitor driftData={driftData} />

      {/* LLM Inference Telemetry Spans */}
      <LLMTelemetry metricsData={metricsData} />
    </div>
  );
}
