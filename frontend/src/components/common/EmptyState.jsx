import { Search, Zap, BarChart3 } from 'lucide-react';
import './EmptyState.css';

const PROMPT_CARDS = [
  {
    icon: Search,
    title: 'Score a Dispute',
    description: 'Try scoring disp_1000000000009 — a medium-risk goods_not_received case',
    action: 'cases',
  },
  {
    icon: Zap,
    title: 'Counterfactual Sim',
    description: 'See what happens when you add delivery signature evidence',
    action: 'cases',
  },
  {
    icon: BarChart3,
    title: 'Portfolio View',
    description: 'Review 20 merchant cohort profiles with VAMP tier health',
    action: 'portfolio',
  },
];

/**
 * Empty state with sample prompt cards.
 * Props: title, subtitle, prompts (optional custom), onPromptClick
 */
export default function EmptyState({ title, subtitle, onPromptClick }) {
  return (
    <div className="empty-state animate-fade-in-up">
      <div className="empty-state-icon">
        <Search size={40} strokeWidth={1.5} />
      </div>
      <h3 className="empty-state-title">{title || 'No data to display'}</h3>
      <p className="empty-state-subtitle">
        {subtitle || 'Select a dispute case or try one of these starter actions:'}
      </p>
      <div className="empty-state-prompts">
        {PROMPT_CARDS.map((card) => {
          const Icon = card.icon;
          return (
            <button
              key={card.title}
              className="empty-state-prompt card"
              onClick={() => onPromptClick?.(card.action)}
            >
              <Icon size={20} />
              <div>
                <strong>{card.title}</strong>
                <span>{card.description}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
