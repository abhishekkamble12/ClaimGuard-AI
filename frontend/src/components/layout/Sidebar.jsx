import { NavLink } from 'react-router-dom';
import {
  Shield,
  Search,
  BarChart3,
  FlaskConical,
  History,
  Settings,
  Activity,
} from 'lucide-react';
import './Sidebar.css';

const NAV_ITEMS = [
  { to: '/',            icon: Shield,       label: 'Home' },
  { to: '/cases',       icon: Search,       label: 'Case Analyzer' },
  { to: '/portfolio',   icon: BarChart3,    label: 'Portfolio' },
  { to: '/diagnostics', icon: FlaskConical, label: 'ML Diagnostics' },
  { to: '/history',     icon: History,      label: 'History' },
  { to: '/settings',    icon: Settings,     label: 'Settings' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <Shield size={22} strokeWidth={2.5} />
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-name">ProofPilot</span>
          <span className="sidebar-brand-tag">AI Risk Manager</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'sidebar-link--active' : ''}`
            }
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer Status */}
      <div className="sidebar-footer">
        <div className="sidebar-status">
          <Activity size={14} className="sidebar-status-dot" />
          <span>System Healthy</span>
        </div>
        <span className="sidebar-version">v2.0.0</span>
      </div>
    </aside>
  );
}
