# Internal Agent Guidelines — Novus Bank Support

## CONFIDENTIAL — FOR NOVUS BANK SUPPORT AGENTS ONLY
*This document is intended for trained Novus Bank customer support agents and relationship managers. Do not share with customers. Last reviewed: Q1 2026.*

---

## Tone and Communication Standards

### Core Principles
- **Empathetic, not apologetic:** Acknowledge the customer's concern; do not over-apologize. "I understand this is frustrating" is better than "I'm so sorry, I'm so sorry."
- **Confident, not bureaucratic:** Avoid jargon. "Your account was flagged by our fraud detection system" is better than "A regulatory compliance threshold was triggered on your account."
- **Resolved, not deflected:** Own the issue. Do not say "the system did this" — say "here's what I can do for you."

### Language Checklist
- Never say "I cannot help with that" — say "Let me check what options are available."
- Never say "That's our policy" without explaining the policy reason.
- Always end calls with: "Is there anything else I can help you with today?"

---

## Fee Waiver Authority

### Waiver Limits by Agent Tier
| Agent Level | Max Waiver per Interaction | Max per Month per Customer |
|-------------|---------------------------|---------------------------|
| Frontline (L1) | ₹500 | ₹1,000 |
| Senior Agent (L2) | ₹2,000 | ₹5,000 |
| Relationship Manager | ₹5,000 | ₹10,000 |
| Branch Manager | Unlimited (log required) | — |

### What Can Be Waived
- SMS alert charges.
- ATM transaction charges (over-limit incidents).
- NEFT/IMPS charges (goodwill, first-time error).
- Annual maintenance fee (one-time for new complaints, retention cases).
- EMI bounce fee (one waiver per account per year).
- Cheque return fee (first instance per year).

### What CANNOT Be Waived by Agent
- TDS on FD interest.
- GST on bank charges.
- Credit card late payment fees without supervisor approval.
- Loan processing fees (once disbursed).
- SWIFT correspondent bank charges (not within Novus Bank's control).

### Waiver Process
1. Verify waiver eligibility (check waiver history in CRM).
2. Log reason in CRM before processing.
3. Waiver reflects in account within **2 working days**.
4. Inform customer of waiver amount and timeline.

---

## Escalation Triggers — When to Escalate Immediately

### Always Escalate to Senior Agent/Supervisor
- Customer explicitly requests to speak with a manager.
- Customer mentions "RBI complaint," "banking ombudsman," or "consumer court."
- Disputed amount above ₹1 lakh.
- Customer reports identity theft or account takeover (not just unauthorized transaction).
- Customer is visibly distressed or using threatening language.
- Issue involves a deceased account holder.
- Complaint about a Novus Bank employee.

### Always Escalate to Fraud Team
- Any transaction the customer did not authorize AND amount > ₹10,000.
- SIM swap suspected.
- Customer received OTPs they did not request.
- Multiple failed login attempts from unknown device (security alert raised).

### Do NOT Escalate Immediately (Handle at L1)
- Routine fee queries and waivers within L1 limit.
- FD interest rate questions.
- Duplicate statement requests.
- Card PIN reset (guide through app).
- Balance and mini-statement queries.

---

## Elite Customer Retention Guidelines

### Identifying At-Risk Elite Customers
An Elite customer is "at-risk" if ANY of:
- AQB has dropped below ₹4 lakh for 2 consecutive quarters (not yet downgraded, but close).
- Customer has raised 2+ complaints in 60 days.
- Customer explicitly asks about closing account or switching banks.
- Recent large outward transfer (>₹5 lakh) to another bank — potential fund migration.

### Retention Offers (Elite Only — Do Not Offer to Standard/Plus)
Retention offers require Relationship Manager or senior agent authorization:

| Situation | Authorized Offer |
|-----------|-----------------|
| AQB drop, no recent complaint | 1-quarter AQB waiver (no downgrade) |
| Multiple complaints | Fee waiver up to ₹5,000 + priority callback in 2 hrs |
| Asking about account closure | RM escalation + 3-month AQB waiver offer |
| Competitor rate offered on FD | FD rate match up to 0.25% above published Elite rate |

**Important:** Do not proactively mention competitor comparison. If customer brings it up: "I understand you've seen other offers. Let me check what we can do for you as a valued Elite member."

---

## Product Misinformation Protocol

### When Customer Has Wrong Information
- Do not say "That's wrong" — say "Let me clarify how this works."
- Provide the correct information with the specific policy source (e.g., "As per our FD policy...").
- If customer insists: offer to send written confirmation via email.
- Do not argue; if customer is determined: log the interaction and escalate if the misinformation could cause financial harm.

### Common Misinformation to Correct
- "My FD will auto-break if I open a new one" — **False.** Multiple FDs can co-exist.
- "UPI transfers above ₹50,000 need branch approval" — **False.** Up to ₹1 lakh per transaction is allowed via app.
- "You can give me my OTP to reset my account" — **False and red flag.** Never ask for or confirm an OTP. If customer says someone asked them for their OTP, treat as fraud attempt.
- "NEFT transfers are instant" — **Incorrect.** NEFT is batch-based; explain timelines clearly.

---

## Data Privacy — What Agents Must NOT Share
- Full account number over chat/phone (share only last 4 digits).
- Transaction details with a third party without customer's explicit verbal consent on the call.
- Balance information without completing identity verification (at minimum: DOB + registered mobile OTP).
- KYC documents stored in the system — agents can confirm "KYC is complete" but cannot share copies.

**Identity Verification (Mandatory before any account action):**
1. Full name as on account.
2. Date of birth.
3. Last 4 digits of registered mobile OR OTP.
4. For high-risk actions (transfer above ₹50,000, loan actions): all three above.

---

## Call/Chat Documentation Standards
- Every interaction must be logged in CRM within **30 minutes** of closure.
- Required fields: Customer ID, Issue category, Resolution provided, Waiver amount (if any), Escalation flag.
- Interactions involving fraud, legal threats, or complaints must be marked "Sensitive" — reviewed by QA team within 24 hours.
