import { CheckCircle2, XCircle, Info } from 'lucide-react';
import './DecisionTrace.css';

/**
 * DecisionTrace: Multi-step stepper trace explaining gate-by-gate routing resolution.
 */
export default function DecisionTrace({ steps = [] }) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="decision-trace-card card">
      <div className="decision-trace-header">
        <h4>Decision Pathway Trace (How ProofPilot Decided)</h4>
        <span className="text-xs text-muted">Gate verification steps</span>
      </div>

      <div className="decision-trace-stepper">
        {steps.map((step, idx) => {
          const isPass = step.status === 'pass';
          const isInfo = step.status === 'info';

          return (
            <div key={idx} className={`trace-step trace-step--${step.status} animate-fade-in-up stagger-${idx + 1}`}>
              <div className="trace-step-marker">
                {isPass ? (
                  <CheckCircle2 size={16} className="text-success" />
                ) : isInfo ? (
                  <Info size={16} className="text-info" />
                ) : (
                  <XCircle size={16} className="text-danger" />
                )}
                {idx < steps.length - 1 && <div className="trace-step-line" />}
              </div>

              <div className="trace-step-body">
                <div className="trace-step-title-row">
                  <span className="trace-gate-name">
                    Step {step.step}: {step.gate}
                  </span>
                  <span className="trace-values text-mono text-xs">
                    Value: <strong>{step.value}</strong> | Threshold: <code>{step.threshold}</code>
                  </span>
                </div>
                <p className="trace-message">{step.message}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
