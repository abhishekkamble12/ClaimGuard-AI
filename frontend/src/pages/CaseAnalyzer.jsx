import { useState, useEffect, useMemo } from 'react';
import { useDatasetCases, useScoreDispute } from '../hooks/useDisputeScore';
import { useStreamResponse } from '../hooks/useStreamResponse';
import CaseSelector from '../components/case/CaseSelector';
import WebhookBanner from '../components/case/WebhookBanner';
import MetricStrip from '../components/case/MetricStrip';
import EvidenceTable from '../components/case/EvidenceTable';
import EvidenceContribChart from '../components/case/EvidenceContribChart';
import CounterfactualSim from '../components/case/CounterfactualSim';
import EconomicDecisionCard from '../components/case/EconomicDecisionCard';
import ResponseGate from '../components/ai/ResponseGate';
import StreamingResponse from '../components/ai/StreamingResponse';
import GapGuidanceCards from '../components/ai/GapGuidanceCards';
import XAISummaryCard from '../components/xai/XAISummaryCard';
import SHAPWaterfall from '../components/xai/SHAPWaterfall';
import DecisionTrace from '../components/xai/DecisionTrace';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import './CaseAnalyzer.css';

export default function CaseAnalyzer() {
  const { data: datasetData, isLoading: isDatasetLoading, error: datasetError, refetch: refetchDataset } = useDatasetCases();
  const scoreMutation = useScoreDispute();
  const streamHook = useStreamResponse();

  const [selectedDisputeId, setSelectedDisputeId] = useState(null);
  const [networkFilter, setNetworkFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [addedEvidence, setAddedEvidence] = useState({}); // { [disputeId]: string[] }
  const [selectedGaps, setSelectedGaps] = useState([]);
  const [scoringResult, setScoringResult] = useState(null);
  const [staticDraftText, setStaticDraftText] = useState(null);

  const cases = useMemo(() => datasetData?.cases || [], [datasetData]);

  // Filter cases by network and search query
  const filteredCases = useMemo(() => {
    return cases.filter((c) => {
      const netMatch =
        networkFilter === 'ALL' ||
        (networkFilter === 'UPI' && c.network === 'UPI') ||
        (networkFilter === 'CARD' && (c.network === 'CARD' || c.network === 'AMEX'));

      const q = searchQuery.toLowerCase().trim();
      const searchMatch =
        !q ||
        (c.dispute_id && c.dispute_id.toLowerCase().includes(q)) ||
        (c.merchant_name && c.merchant_name.toLowerCase().includes(q)) ||
        (c.reason_title && c.reason_title.toLowerCase().includes(q)) ||
        (c.reason_code && c.reason_code.toLowerCase().includes(q));

      return netMatch && searchMatch;
    });
  }, [cases, networkFilter, searchQuery]);

  // Set default selected dispute
  useEffect(() => {
    if (!selectedDisputeId && filteredCases.length > 0) {
      setSelectedDisputeId(filteredCases[0].dispute_id);
    }
  }, [filteredCases, selectedDisputeId]);

  // Active base dispute case
  const activeBaseCase = useMemo(() => {
    return cases.find((c) => c.dispute_id === selectedDisputeId) || cases[0] || null;
  }, [cases, selectedDisputeId]);

  // Augmented dispute case with simulated added evidence
  const activeDisputeCase = useMemo(() => {
    if (!activeBaseCase) return null;
    const addedForThis = addedEvidence[selectedDisputeId] || [];
    if (addedForThis.length === 0) return activeBaseCase;

    const cloned = JSON.parse(JSON.stringify(activeBaseCase));
    cloned.evidence_documents = cloned.evidence_documents || {};
    addedForThis.forEach((eid) => {
      const label = eid.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
      cloned.evidence_documents[eid] = `Official ${label} uploaded by merchant. Document verifies match with transaction records and valid timestamp.`;
    });
    return cloned;
  }, [activeBaseCase, addedEvidence, selectedDisputeId]);

  // Score dispute when active dispute changes
  useEffect(() => {
    if (activeDisputeCase) {
      scoreMutation.mutate(
        { disputeCase: activeDisputeCase },
        {
          onSuccess: (data) => {
            setScoringResult(data);
            setStaticDraftText(null);
          },
        }
      );
    }
  }, [activeDisputeCase]);

  // Handle selecting gap checkboxes in Counterfactual Simulator
  const handleToggleGap = (eid) => {
    setSelectedGaps((prev) =>
      prev.includes(eid) ? prev.filter((id) => id !== eid) : [...prev, eid]
    );
  };

  // Apply simulated evidence gaps and re-score
  const handleApplyGaps = () => {
    if (!selectedDisputeId) return;
    const current = addedEvidence[selectedDisputeId] || [];
    const merged = [...new Set([...current, ...selectedGaps])];
    setAddedEvidence((prev) => ({ ...prev, [selectedDisputeId]: merged }));
    setSelectedGaps([]);
  };

  // Reset case to original state
  const handleResetCase = () => {
    if (!selectedDisputeId) return;
    setAddedEvidence((prev) => ({ ...prev, [selectedDisputeId]: [] }));
    setSelectedGaps([]);
  };

  // Handle streaming AI response
  const handleStreamDraft = () => {
    if (selectedDisputeId) {
      streamHook.startStream(selectedDisputeId);
    }
  };

  // Handle static draft generation (fallback text)
  const handleGenerateStatic = () => {
    const text = `FORMAL CHARGEBACK DISPUTE REBUTTAL
Date: 2026-09-01
To: Payment Processing Dispute Review Department
Merchant: ${activeDisputeCase?.merchant_name || 'Merchant'}
Dispute ID: ${activeDisputeCase?.dispute_id || 'N/A'}
Order ID: ${activeDisputeCase?.transaction?.order_id || 'N/A'}
Disputed Amount: INR ${activeDisputeCase?.transaction?.amount || 0}
Reason Code: ${activeDisputeCase?.reason_code || '4553'} (${activeDisputeCase?.reason_title || 'Dispute'})

SUMMARY OF CONTESTATION:
The merchant hereby formally contests this chargeback claim. Complete evidence has been submitted and verified to demonstrate that the transaction was fully legitimate, authorized, and delivered as agreed.

VERIFIED EVIDENCE ENCLOSURES:
1. Proof of Order Placement & Confirmation
2. Official Delivery Tracking Log & Digital Timestamp
3. Physical Customer Delivery Signature (POD)
4. Comprehensive Transaction Metadata & Authorization Logs

CONCLUSION:
Based on the indisputable factual evidence enclosed, the merchant respectfully requests that this dispute be decided in the merchant's favor and all disputed funds be credited back.`;
    setStaticDraftText(text);
  };

  if (isDatasetLoading) {
    return (
      <div className="case-analyzer-page">
        <LoadingState lines={8} height="24px" />
      </div>
    );
  }

  if (datasetError) {
    return (
      <div className="case-analyzer-page">
        <ErrorState message={datasetError.message} onRetry={refetchDataset} />
      </div>
    );
  }

  const addedForActive = addedEvidence[selectedDisputeId] || [];
  const fixableGaps = scoringResult?.gap_explanation || [];

  return (
    <div className="case-analyzer-page animate-fade-in">
      {/* Top Header Controls */}
      <CaseSelector
        cases={filteredCases}
        selectedId={selectedDisputeId}
        onSelectCase={(id) => {
          setSelectedDisputeId(id);
          setSelectedGaps([]);
        }}
        networkFilter={networkFilter}
        onNetworkFilterChange={setNetworkFilter}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      {/* Webhook HMAC Security Banner */}
      <WebhookBanner disputeCase={activeDisputeCase} />

      {/* Top 6 KPI Metric Strip */}
      <MetricStrip scoringResult={scoringResult} />

      {/* Two-Column Middle Section: Evidence & Counterfactual Simulator */}
      <div className="case-grid-two-col">
        <div className="case-col-left">
          <EvidenceTable evidenceElements={scoringResult?.evidence_elements} />
          <EvidenceContribChart evidenceElements={scoringResult?.evidence_elements} />
        </div>

        <div className="case-col-right">
          <CounterfactualSim
            missingEvidence={scoringResult?.missing_evidence || []}
            weakEvidence={scoringResult?.weak_evidence || []}
            selectedGaps={selectedGaps}
            onToggleGap={handleToggleGap}
            onApplyGaps={handleApplyGaps}
            onResetCase={handleResetCase}
            hasSimulatedEvidence={addedForActive.length > 0}
            isScoring={scoreMutation.isPending}
          />

          <EconomicDecisionCard
            economicRecommendation={scoringResult?.economic_recommendation}
            expectedFinancialValue={scoringResult?.expected_financial_value}
          />
        </div>
      </div>

      {/* AI Reasoning Gate & Response Drafting */}
      <div className="case-ai-section">
        <ResponseGate
          routingDecision={scoringResult?.routing_decision}
          readinessPct={scoringResult?.completeness_pct}
          confidencePct={scoringResult?.confidence_pct}
          onStreamClick={handleStreamDraft}
          onGenerateStaticClick={handleGenerateStatic}
          isStreaming={streamHook.isStreaming}
        />

        {/* If Gate is Passed and Stream / Static requested */}
        {(streamHook.text || streamHook.isStreaming || staticDraftText) && (
          <StreamingResponse
            text={streamHook.text || staticDraftText}
            isStreaming={streamHook.isStreaming}
            error={streamHook.error}
            onStop={streamHook.stopStream}
            onRetry={handleStreamDraft}
          />
        )}

        {/* If Gate is Blocked: Actionable Gap Recommendations */}
        {scoringResult?.routing_decision !== 'auto_draft_response' && (
          <GapGuidanceCards
            gaps={fixableGaps}
            onSelectEvidence={(eid) => {
              setSelectedGaps([eid]);
            }}
          />
        )}
      </div>

      {/* XAI Explainability Section */}
      <div className="case-xai-section">
        <XAISummaryCard scoringResult={scoringResult} disputeCase={activeDisputeCase} />
        <div className="case-grid-two-col">
          <SHAPWaterfall
            featureImportances={scoringResult?.ml_feature_importances}
            localShap={scoringResult?.local_shap_explanation}
          />
          <DecisionTrace steps={scoringResult?.decision_trace} />
        </div>
      </div>
    </div>
  );
}
