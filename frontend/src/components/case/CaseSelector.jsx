import { Filter, Search } from 'lucide-react';
import './CaseSelector.css';

/**
 * CaseSelector: Payment network filter tabs, search input, and dispute selector dropdown.
 */
export default function CaseSelector({
  cases = [],
  selectedId,
  onSelectCase,
  networkFilter,
  onNetworkFilterChange,
  searchQuery,
  onSearchChange,
}) {
  const networks = [
    { id: 'ALL', label: 'All Networks' },
    { id: 'UPI', label: 'UPI (NPCI)' },
    { id: 'CARD', label: 'Cards (Visa / MC / Amex)' },
  ];

  return (
    <div className="case-selector card">
      <div className="case-selector-top">
        {/* Network Filter Pills */}
        <div className="network-filters">
          <Filter size={15} className="text-muted" />
          {networks.map(net => (
            <button
              key={net.id}
              className={`network-filter-btn ${networkFilter === net.id ? 'network-filter-btn--active' : ''}`}
              onClick={() => onNetworkFilterChange(net.id)}
            >
              {net.label}
            </button>
          ))}
        </div>

        {/* Search Bar */}
        <div className="case-search-box">
          <Search size={14} className="text-muted" />
          <input
            type="text"
            placeholder="Search dispute ID, merchant, reason..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>
      </div>

      {/* Case Dropdown */}
      <div className="case-dropdown-row">
        <label htmlFor="case-select" className="case-dropdown-label">
          Active Dispute Case:
        </label>
        <select
          id="case-select"
          value={selectedId || ''}
          onChange={(e) => onSelectCase(e.target.value)}
          className="case-select-element"
        >
          {cases.map((c) => {
            const dId = c.dispute_id || c.legacy_dispute_id;
            const net = c.network || 'CARD';
            const amt = c.transaction?.amount || 0;
            const merchant = c.merchant_name || 'Merchant';
            const reason = (c.reason_title || c.reason_category || '').slice(0, 32);
            return (
              <option key={dId} value={dId}>
                {dId} · [{net}] ₹{amt.toLocaleString()} · {merchant} · {reason}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}
