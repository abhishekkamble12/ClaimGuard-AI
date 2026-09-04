import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppShell from './components/layout/AppShell';
import Landing from './pages/Landing';
import CaseAnalyzer from './pages/CaseAnalyzer';
import Portfolio from './pages/Portfolio';
import Diagnostics from './pages/Diagnostics';
import History from './pages/History';
import Settings from './pages/Settings';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<Landing />} />
            <Route path="cases" element={<CaseAnalyzer />} />
            <Route path="cases/:disputeId" element={<CaseAnalyzer />} />
            <Route path="portfolio" element={<Portfolio />} />
            <Route path="diagnostics" element={<Diagnostics />} />
            <Route path="history" element={<History />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
