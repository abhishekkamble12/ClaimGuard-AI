import { Link } from 'react-router-dom';
import { Shield, Zap, Lock, BarChart2, CheckCircle2, ArrowRight, Activity, TrendingUp } from 'lucide-react';
import MetricCard from '../components/common/MetricCard';
import './Landing.css';

export default function Landing() {
  return (
    <div className="landing-page animate-fade-in">
      {/* Hero Banner */}
      <section className="landing-hero">
        <div className="landing-hero-badge">
          <Shield size={14} />
          <span>Razorpay AI Buildathon 2026 · Track 02: AI Risk Manager</span>
        </div>
        <h1 className="landing-hero-title">
          Enterprise Pre-Submission <br />
          <span className="gradient-text">AI Dispute Risk Firewall</span>
        </h1>
        <p className="landing-hero-subtitle">
          Evaluate merchant chargeback evidence readiness before submission, predict calibrated win probabilities via a Stacked ML Ensemble, calculate monetary Expected Value in ₹, and enforce HMAC-SHA256 gateway security.
        </p>

        <div className="landing-hero-actions">
          <Link to="/cases" className="btn btn-primary btn-lg">
            <Zap size={18} />
            <span>Launch Case Risk Analyzer</span>
            <ArrowRight size={16} />
          </Link>
          <Link to="/portfolio" className="btn btn-ghost btn-lg">
            <BarChart2 size={18} />
            <span>View Portfolio Intelligence</span>
          </Link>
        </div>
      </section>

      {/* Live Model Stats Strip */}
      <section className="landing-stats-strip grid-4 gap-4">
        <MetricCard label="Calibrated ROC-AUC" value="93.4%" variant="success" />
        <MetricCard label="Calibration Brier Score" value="0.0890" variant="success" />
        <MetricCard label="Dispute Reason Codes" value="NPCI + Cards" variant="info" />
        <MetricCard label="Gate Security" value="HMAC-SHA256" variant="default" />
      </section>

      {/* Side-by-Side: Blind AI vs Risk Firewall */}
      <section className="landing-comparison card">
        <h3>Why ProofPilot vs. Generic AI Writers</h3>
        <p className="landing-comparison-sub">
          Generic LLM tools blindly draft letters for doomed disputes, triggering network losses and non-refundable ₹500 dispute penalty fees.
        </p>

        <div className="comparison-grid">
          <div className="comparison-col comparison-col--bad">
            <h4 className="text-danger">❌ Standard AI Letter Generators</h4>
            <ul>
              <li>Blindly drafts response letters for 100% of disputes</li>
              <li>Causes guaranteed loss of ₹500 penalty fees on doomed cases</li>
              <li>Ignores missing evidence documents & quality gaps</li>
              <li>Black-box hallucinated legal jargon without probability estimates</li>
              <li>No merchant portfolio VAMP/VCMP monitoring</li>
            </ul>
          </div>

          <div className="comparison-col comparison-col--good">
            <h4 className="text-success">✅ ProofPilot Pre-Submission Firewall</h4>
            <ul>
              <li>Pre-submission readiness gate: auto-drafts ONLY high-win cases</li>
              <li>Recommends ACCEPT_LOSS when EV is negative to save ₹500 fee</li>
              <li>Counterfactual Simulator projects score gains for missing items</li>
              <li>Calibrated Stacked ML Ensemble P(Win) with SHAP Explainability</li>
              <li>Real-time VAMP/VCMP chargeback ratio health & anomaly tracking</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3 Starter Actions */}
      <section className="landing-starter-cards grid-3 gap-4">
        <Link to="/cases" className="starter-card card">
          <div className="starter-card-icon" style={{ background: 'var(--accent-primary-dim)', color: 'var(--accent-primary)' }}>
            <Zap size={24} />
          </div>
          <h4>Interactive Case Risk Analyzer</h4>
          <p>Score live disputes, simulate adding missing documents, view SHAP attributions, and stream AI response letters.</p>
          <span className="starter-card-link">Explore Cases &rarr;</span>
        </Link>

        <Link to="/portfolio" className="starter-card card">
          <div className="starter-card-icon" style={{ background: 'var(--accent-success-dim)', color: 'var(--accent-success)' }}>
            <TrendingUp size={24} />
          </div>
          <h4>Portfolio Intelligence & VAMP Tiers</h4>
          <p>Monitor 20 Indian merchant cohort profiles, chargeback ratios against card scheme limits, and systemic evidence gaps.</p>
          <span className="starter-card-link">View Portfolio &rarr;</span>
        </Link>

        <Link to="/diagnostics" className="starter-card card">
          <div className="starter-card-icon" style={{ background: 'var(--accent-info-dim)', color: 'var(--accent-info)' }}>
            <Activity size={24} />
          </div>
          <h4>ML Architecture & Diagnostics</h4>
          <p>Review model comparison benchmarks, Brier score calibration, PSI drift tracking, and RAGAS reasoning metrics.</p>
          <span className="starter-card-link">View Diagnostics &rarr;</span>
        </Link>
      </section>
    </div>
  );
}
