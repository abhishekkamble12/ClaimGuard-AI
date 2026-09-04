import axios from 'axios';

const API_BASE = '/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/* Inject API key from localStorage when available */
client.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('proofpilot_api_key');
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

/* Normalize error responses */
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';
    return Promise.reject(new Error(message));
  }
);

/* ── API Methods ─────────────────────────────────────────────── */

export async function scoreDispute(disputeCase, options = {}) {
  const { data } = await client.post('/disputes/score', {
    dispute: disputeCase,
    api_key: options.apiKey || null,
    use_ground_truth: options.useGroundTruth || false,
  });
  return data;
}

export async function batchScoreDisputes(disputes, apiKey = null) {
  const { data } = await client.post('/disputes/batch-score', {
    disputes,
    api_key: apiKey,
  });
  return data;
}

export async function getDisputeAudit(disputeId) {
  const { data } = await client.get(`/disputes/${disputeId}/audit`);
  return data;
}

export async function getHealth() {
  const { data } = await client.get('/health');
  return data;
}

export async function getMetrics() {
  const { data } = await client.get('/metrics');
  return data;
}

export async function getVersion() {
  const { data } = await client.get('/version');
  return data;
}

export async function getDriftReport() {
  const { data } = await client.get('/drift/evaluate');
  return data;
}

export async function getDatasetCases() {
  const { data } = await client.get('/dataset/cases');
  return data;
}

export async function getDatasetCase(disputeId) {
  const { data } = await client.get(`/dataset/cases/${disputeId}`);
  return data;
}

export async function getPortfolioReport() {
  const { data } = await client.get('/portfolio/report');
  return data;
}

export async function getMerchantProfiles() {
  const { data } = await client.get('/portfolio/merchants');
  return data;
}

/**
 * Create an SSE EventSource for streaming dispute response.
 * Returns the EventSource instance for manual control (close, etc).
 */
export function createResponseStream(disputeId, apiKey = null) {
  let url = `${API_BASE}/disputes/${disputeId}/stream-response`;
  if (apiKey) {
    url += `?api_key=${encodeURIComponent(apiKey)}`;
  }
  return new EventSource(url);
}

export default client;
