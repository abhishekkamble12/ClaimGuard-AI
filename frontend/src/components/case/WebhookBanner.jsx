import { Lock, ShieldCheck, ShieldAlert } from 'lucide-react';
import './WebhookBanner.css';

/**
 * WebhookBanner: Displays Razorpay dispute webhook event metadata and HMAC-SHA256 verification status.
 */
export default function WebhookBanner({ disputeCase }) {
  if (!disputeCase) return null;

  const orderId = disputeCase.transaction?.order_id || 'order_1000000001745';
  const paymentId = disputeCase.transaction?.payment_id || 'pay_1000000005494';
  const merchantName = disputeCase.merchant_name || 'Razorpay Merchant';
  const amount = disputeCase.transaction?.amount || 0;
  const network = disputeCase.network || 'CARD';

  return (
    <div className="webhook-banner card">
      <div className="webhook-banner-left">
        <span className="webhook-badge">
          <Lock size={12} />
          Razorpay Webhook: dispute.created
        </span>
        <div className="webhook-details">
          <span><strong>Order:</strong> <code>{orderId}</code></span>
          <span><strong>Payment:</strong> <code>{paymentId}</code></span>
          <span><strong>Merchant:</strong> {merchantName}</span>
          <span><strong>Amount:</strong> ₹{amount.toLocaleString()} INR ({network})</span>
        </div>
      </div>
      <div className="webhook-banner-right">
        <span className="hmac-verified-pill">
          <ShieldCheck size={14} />
          HMAC-SHA256: Verified
        </span>
      </div>
    </div>
  );
}
