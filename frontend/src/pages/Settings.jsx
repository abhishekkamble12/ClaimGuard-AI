import { useState } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { Save, Key, ShieldCheck, Cpu } from 'lucide-react';
import './Settings.css';

export default function Settings() {
  const [apiKey, setApiKey] = useLocalStorage('proofpilot_api_key', '');
  const [provider, setProvider] = useLocalStorage('proofpilot_provider', 'groq');
  const [webhookSecret, setWebhookSecret] = useLocalStorage('proofpilot_webhook_secret', 'rzp_sec_buildathon_2026_claimguard');
  const [minScore, setMinScore] = useLocalStorage('proofpilot_min_score', 80);
  const [minConf, setMinConf] = useLocalStorage('proofpilot_min_conf', 75);
  const [savedToast, setSavedToast] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    setSavedToast(true);
    setTimeout(() => setSavedToast(false), 2500);
  };

  return (
    <div className="settings-page animate-fade-in">
      <div className="settings-header">
        <h2>System Configuration & Gateway Security Settings</h2>
        <p className="text-secondary text-sm">
          Manage live LLM reasoning provider, API keys, Razorpay webhook secrets, and automated risk gate thresholds.
        </p>
      </div>

      <form onSubmit={handleSave} className="settings-form">
        {/* Section 1: LLM Provider Configuration */}
        <div className="settings-section card">
          <div className="section-title">
            <Cpu size={18} className="text-primary" />
            <h4>LLM Reasoning Engine</h4>
          </div>
          <p className="section-sub">
            ProofPilot uses Groq Cloud (Llama-3.3-70B) or Google Gemini 2.5 Flash for live reasoning, with zero-dependency deterministic NLP fallback.
          </p>

          <div className="form-group">
            <label>Active LLM Provider:</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="groq">Groq Cloud (llama-3.3-70b-versatile, ~700 tok/sec)</option>
              <option value="gemini">Google Gemini 2.5 Flash</option>
              <option value="local">Deterministic Offline NLP Engine (No API Key Required)</option>
            </select>
          </div>

          <div className="form-group">
            <label>API Key (Optional for Offline Mode):</label>
            <div className="input-with-icon">
              <Key size={14} className="text-muted" />
              <input
                type="password"
                placeholder={provider === 'groq' ? 'gsk_...' : provider === 'gemini' ? 'AIzaSy...' : 'Not required for local NLP'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
            </div>
            <span className="form-hint">
              Stored locally in your browser. Never transmitted to third-party logging.
            </span>
          </div>
        </div>

        {/* Section 2: Gateway Security */}
        <div className="settings-section card">
          <div className="section-title">
            <ShieldCheck size={18} className="text-success" />
            <h4>Razorpay Webhook Cryptographic Security</h4>
          </div>
          <p className="section-sub">
            All webhook dispute ingestion events verify SHA256 HMAC digest authentication with constant-time equality matching.
          </p>

          <div className="form-group">
            <label>Webhook Secret:</label>
            <input
              type="text"
              value={webhookSecret}
              onChange={(e) => setWebhookSecret(e.target.value)}
            />
          </div>
        </div>

        {/* Section 3: Decision Gate Thresholds */}
        <div className="settings-section card">
          <div className="section-title">
            <ShieldCheck size={18} className="text-warning" />
            <h4>Automated Response Routing Thresholds</h4>
          </div>

          <div className="grid-2 gap-4">
            <div className="form-group">
              <label>Minimum Readiness Score (%):</label>
              <input
                type="number"
                min="50"
                max="95"
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
              />
              <span className="form-hint">Default: 80%. Score required to unlock automated response drafting.</span>
            </div>

            <div className="form-group">
              <label>Minimum Confidence (%):</label>
              <input
                type="number"
                min="50"
                max="95"
                value={minConf}
                onChange={(e) => setMinConf(Number(e.target.value))}
              />
              <span className="form-hint">Default: 75%. Confidence gate penalized by weak evidence items.</span>
            </div>
          </div>
        </div>

        <div className="settings-actions">
          <button type="submit" className="btn btn-primary btn-lg">
            <Save size={16} />
            <span>Save Settings</span>
          </button>
          {savedToast && <span className="save-toast text-success font-medium">✅ Settings saved successfully!</span>}
        </div>
      </form>
    </div>
  );
}
