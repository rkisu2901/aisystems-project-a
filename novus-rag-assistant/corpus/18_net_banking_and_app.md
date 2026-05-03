# Net Banking and Mobile App — Novus Bank

## Novus App

### Supported Platforms
- **iOS:** Version 15.0 and above.
- **Android:** Version 10.0 and above.
- **Novus Web (net banking):** app.novusbank.in — Chrome, Firefox, Edge (latest two versions).
- **Note:** Jailbroken/rooted devices are blocked for security; app will not install.

### App Registration (First-Time)
1. Download from App Store / Google Play.
2. Enter registered mobile number → OTP verification.
3. Enter Debit Card number + PIN (to verify identity before setting up app login).
4. Set 6-digit MPIN (app login PIN) or enable biometric.
5. Account is live immediately after setup.

---

## Login and Authentication

### App Login
- **MPIN (6-digit):** Primary login method.
- **Biometric (fingerprint / Face ID):** Available after MPIN setup; enable in app settings.
- **3 wrong MPIN attempts:** App locks for **2 hours**. After lockout, login requires OTP to reset.
- **5 consecutive wrong MPIN attempts (cumulative):** MPIN reset required (via OTP + debit card verification).

### Net Banking Login
- Username (customer ID) + password.
- OTP required for every login (sent to registered mobile).
- **3 wrong password attempts:** Account locked for **2 hours**; unlock via OTP or branch visit.
- Password policy: Minimum 8 characters, at least one uppercase, one number, one special character.
- Password expiry: **180 days**; customer prompted to change after expiry.

### Session Management
- **Auto-timeout:** 15 minutes of inactivity logs out session automatically.
- Re-authentication required after timeout — MPIN or OTP.
- Concurrent sessions: One active session per device (new login from another device auto-terminates the previous session).
- Customer is notified via SMS when a new session is initiated from a new device.

---

## Two-Factor Authentication (2FA)

### When 2FA is Required
- Any fund transfer above **₹10,000** requires OTP (even within the app).
- Changes to registered mobile number or email.
- Adding a new beneficiary.
- Enabling/disabling international card usage.
- Password or MPIN change.
- Loan prepayment or foreclosure.
- Nomination change.

### 2FA Delivery
- OTP sent to registered mobile via SMS.
- OTP validity: **3 minutes**.
- If OTP not received within 1 minute: Resend option available (max 3 resends per session).
- Voice OTP available for customers who cannot receive SMS (call-based delivery).

---

## Key App Features

### Account and Cards
- View account balance and mini-statement (last 10 transactions without OTP).
- Detailed statement (3 years): Download as PDF/CSV.
- Block/unblock debit card (instant).
- Set transaction limits for debit card, UPI, and net banking.
- Enable/disable international card usage.
- Request cheque book.
- View and pay credit card bills.

### Payments
- UPI transfers (instant, up to ₹1L/day).
- NEFT/RTGS/IMPS to saved and new beneficiaries.
- Bharat BillPay (electricity, water, gas, DTH, broadband, insurance).
- Schedule future-dated payments (up to 90 days in advance).
- Set up recurring UPI AutoPay mandates.

### Investments and Deposits
- Open/close FDs and RDs.
- Buy/sell mutual funds (SIP and lump-sum).
- Buy digital gold.
- View NPS portfolio and add contributions.

### Loans
- View EMI schedule and outstanding principal.
- Pay EMI or part-prepayment.
- Download loan statements and interest certificates.
- Request loan closure/foreclosure.

### Services
- Update personal details (address, email — mobile number change requires branch).
- KYC document upload and status check.
- Submit Form 15G/15H.
- Raise service requests and complaints.
- Schedule video KYC appointments.

---

## Security Features

### Device Binding
- Novus app is bound to a specific device (SIM + device combination).
- If SIM is moved to a new phone: Re-registration required with OTP + debit card verification.
- If device is changed: Same re-registration process.

### Suspicious Login Alerts
- Push notification + SMS for every new device login.
- If you receive a login alert you did not initiate: Immediately use "Block Access" from the notification or call 1800-NOVUS.

### Remote Account Lock
- Available in the app even after account compromise: Novus app → Emergency → Lock Account.
- Account lock blocks all outward transactions within 60 seconds.
- Unlock: Only via video KYC or branch visit (cannot unlock remotely once emergency locked).

### Anti-Phishing Measures
- Novus app displays a personal greeting (set during registration) on the login screen — verify this before entering MPIN.
- If greeting is missing or wrong: Do not log in; report immediately to 1800-NOVUS.

---

## Net Banking — Additional Notes
- **Transaction history:** Available for up to **10 years** on net banking; app shows 3 years by default.
- **Bulk payment upload:** Available for business accounts (NEFT batch upload via CSV).
- **API banking (corporate):** Separate registration; contact relationship manager.
- **Browser certificate pinning:** Net banking validates Novus Bank's SSL certificate; if browser warns about certificate errors, do not proceed and report to support.
