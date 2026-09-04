import { useState } from 'react';
import { useDatasetCases } from '../hooks/useDisputeScore';
import { Search, Eye, FileJson } from 'lucide-react';
import StatusBadge from '../components/common/StatusBadge';
import RiskBadge from '../components/common/RiskBadge';
import LoadingState from '../components/common/LoadingState';
import './History.css';

export default function History() {
  const { data: datasetData, isLoading } = useDatasetCases();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCase, setSelectedCase] = useState(null);

  const cases = datasetData?.cases || [];

  const filtered = cases.filter((c) => {
    const q = searchTerm.toLowerCase();
    return (
      !q ||
      c.dispute_id?.toLowerCase().includes(q) ||
      c.merchant_name?.toLowerCase().includes(q) ||
      c.reason_title?.toLowerCase().includes(q)
    );
  });

  if (isLoading) {
    return (
      <div className="history-page">
        <LoadingState lines={8} height="28px" />
      </div>
    );
  }

  return (
    <div className="history-page animate-fade-in">
      <div className="history-header">
        <h2>Dispute Audit Logs & Defense Trace History</h2>
        <p className="text-secondary text-sm">
          Search immutable decision audit logs, inspect raw dispute payloads, and verify past AI risk decisions.
        </p>
      </div>

      <div className="history-search card">
        <Search size={16} className="text-muted" />
        <input
          type="text"
          placeholder="Search dispute audit logs by ID, merchant name, or reason..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="history-grid">
        <div className="history-list card">
          <div className="history-list-header">
            <h4>Dispute History ({filtered.length})</h4>
          </div>

          <div className="history-list-scroll">
            {filtered.map((c) => {
              const dId = c.dispute_id;
              const isSelected = selectedCase?.dispute_id === dId;
              const isAuto = c.expected_route === 'auto_draft_response';

              return (
                <div
                  key={dId}
                  className={`history-item ${isSelected ? 'history-item--active' : ''}`}
                  onClick={() => setSelectedCase(c)}
                >
                  <div className="history-item-top">
                    <strong className="text-sm">{dId}</strong>
                    <RiskBadge level={isAuto ? 'LOW' : 'MEDIUM'} />
                  </div>
                  <span className="text-xs text-muted">{c.merchant_name} · ₹{c.transaction?.amount?.toLocaleString()} INR</span>
                  <span className="text-xs text-secondary">{c.reason_title}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="history-detail card">
          {selectedCase ? (
            <div className="history-detail-content">
              <div className="history-detail-top">
                <h4>Audit Trace: {selectedCase.dispute_id}</h4>
                <span className="text-xs text-muted">Immutable JSON Decision Defense</span>
              </div>

              <div className="history-detail-metrics">
                <div className="detail-pill">
                  <span className="text-muted text-xs">Merchant</span>
                  <strong>{selectedCase.merchant_name}</strong>
                </div>
                <div className="detail-pill">
                  <span className="text-muted text-xs">Amount</span>
                  <strong>₹{selectedCase.transaction?.amount?.toLocaleString()}</strong>
                </div>
                <div className="detail-pill">
                  <span className="text-muted text-xs">Network</span>
                  <strong>{selectedCase.network}</strong>
                </div>
                <div className="detail-pill">
                  <span className="text-muted text-xs">Category</span>
                  <strong>{selectedCase.reason_category}</strong>
                </div>
              </div>

              <pre className="history-json-viewer">
                {JSON.stringify(selectedCase, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="history-empty-detail">
              <FileJson size={40} className="text-muted" />
              <p>Select a dispute from the list to view its immutable defense audit trace.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
