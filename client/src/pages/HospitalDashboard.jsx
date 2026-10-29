import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'
import './HospitalDashboard.css'

const API_BASE = 'http://localhost:8000'
const POLL_INTERVAL_MS = 4000
const TABS = ['pending', 'approved', 'rejected', 'blocked']

// Backend domain errors (api/errors.py) are already safe, user-facing strings.
// Anything else (validation arrays, missing body, network failure) falls back
// to a generic message -- never a stack trace, hostname, or raw exception text.
const safeDetail = (body, fallback) =>
  body && typeof body.detail === 'string' ? body.detail : fallback

const NETWORK_ERROR = 'Unable to reach the server. Please try again in a moment.'

const protocolSummary = (r) => {
  const strategy = r.aggregation_strategy === 'coordinate_median' ? 'Coordinate-median' : 'FedAvg'
  const prox = r.fedprox_mu > 0 ? ` · FedProx μ=${r.fedprox_mu}` : ''
  return `${r.fl_rounds} rounds · ${strategy}${prox}`
}

const RequestCard = ({ r, showDecisionActions, onApprove, onReject, onBlock }) => (
  <article className="hd-card">
    <header className="hd-card-header">
      <h3 className="hd-card-title">Request #{r.request_id}</h3>
      <span className={`hd-status hd-status-${r.status}`}>{r.status}</span>
    </header>

    <p className="hd-dataset">{r.dataset_name} (#{r.dataset_id})</p>

    <dl className="hd-grid">
      <div className="hd-field">
        <dt>Requester</dt>
        <dd>{r.requester_email}</dd>
      </div>
      <div className="hd-field">
        <dt>Purpose</dt>
        <dd>{r.purpose || '—'}</dd>
      </div>
      <div className="hd-field">
        <dt>Submitted</dt>
        <dd>{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</dd>
      </div>
      <div className="hd-field">
        <dt>Proposed FL protocol</dt>
        <dd>{protocolSummary(r)}</dd>
      </div>
      <div className="hd-field">
        <dt>Privacy</dt>
        <dd>
          {r.dp_enabled
            ? `Differential privacy on (noise multiplier ${r.dp_noise_multiplier})`
            : '⚠ No differential privacy'}
        </dd>
      </div>
      <div className="hd-field">
        <dt>Aggregation</dt>
        <dd>
          {r.secagg_enabled
            ? 'Secure aggregation on'
            : "⚠ No secure aggregation — the coordinator sees each hospital's update individually"}
        </dd>
      </div>
    </dl>

    <footer className="hd-actions">
      {showDecisionActions && (
        <>
          <button className="hd-btn hd-btn-approve" onClick={() => onApprove(r.request_id)}>Approve</button>
          <button className="hd-btn" onClick={() => onReject(r.request_id)}>Reject</button>
        </>
      )}
      <button className="hd-btn hd-btn-danger" onClick={() => onBlock(r.requester_id, r.requester_email)}>
        Block researcher
      </button>
    </footer>
  </article>
)

const HospitalDashboard = () => {
  const navigate = useNavigate()
  const { username, institution, institutionId, userId } = useAppContext()

  // Identity comes from the login session, and the backend independently
  // re-derives the approver's institution from the DB row (see
  // services/access_service.py) -- this is a UX convenience, not the
  // security boundary. A mismatched id here gets a 403, not a bypass.
  const approverId = userId
  const [activeTab, setActiveTab] = useState('pending')
  const [requests, setRequests] = useState([])
  const [blockedResearchers, setBlockedResearchers] = useState([])
  const [notifications, setNotifications] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    if (!institutionId || !approverId) {
      navigate('/login')
    }
  }, [institutionId, approverId, navigate])

  const fetchRequests = useCallback(async () => {
    if (!institutionId) return
    try {
      const status = activeTab === 'blocked' ? 'pending' : activeTab
      const res = await fetch(
        `${API_BASE}/dataset/requests/by-institution?institution_id=${institutionId}&status=${status}`
      )
      if (res.ok) setRequests(await res.json())
    } catch {
      setError(NETWORK_ERROR)
    } finally {
      setLoading(false)
    }
  }, [institutionId, activeTab])

  const fetchBlocked = useCallback(async () => {
    if (!institutionId || !approverId) return
    try {
      const res = await fetch(
        `${API_BASE}/hospitals/${institutionId}/blocked-researchers?admin_user_id=${approverId}`
      )
      if (res.ok) setBlockedResearchers(await res.json())
    } catch {
      /* non-critical background refresh */
    }
  }, [institutionId, approverId])

  const fetchNotifications = useCallback(async () => {
    if (!institutionId || !approverId) return
    try {
      const res = await fetch(
        `${API_BASE}/notifications/institution?institution_id=${institutionId}&admin_user_id=${approverId}&unread_only=true`
      )
      if (res.ok) setNotifications(await res.json())
    } catch {
      /* non-critical background refresh */
    }
  }, [institutionId, approverId])

  useEffect(() => {
    const refresh = () => { fetchRequests(); fetchBlocked(); fetchNotifications() }
    refresh()
    if (!institutionId) return undefined
    const interval = setInterval(refresh, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [institutionId, fetchRequests, fetchBlocked, fetchNotifications])

  const decide = async (requestId, decision) => {
    const verb = decision === 'approved' ? 'approve' : 'reject'
    const prompt = decision === 'approved'
      ? `Approve request #${requestId}? Your institution's data will take part in federated training for this project. Raw patient data never leaves your hospital.`
      : `Reject request #${requestId}? This cannot be undone.`
    if (!window.confirm(prompt)) return

    setError('')
    setSuccess('')
    try {
      const res = await fetch(`${API_BASE}/dataset/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          request_id: requestId,
          approver_id: Number(approverId),
          decision,
          comment: decision === 'approved' ? 'Approved from UI' : 'Rejected from UI',
        }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(safeDetail(body, `Unable to ${verb} this request. Please try again.`))
        return
      }
      const base = body.status === 'approved'
        ? `Request #${body.request_id} approved.${body.experiment_id ? ` Federated experiment #${body.experiment_id} started.` : ''}`
        : `Request #${body.request_id} ${body.status}.`
      setSuccess(body.warning ? `${base} Note: ${body.warning}` : base)
      await fetchRequests()
    } catch {
      setError(NETWORK_ERROR)
    }
  }

  const blockResearcher = async (researcherId, researcherEmail) => {
    const reason = window.prompt(
      `Block ${researcherEmail || `researcher #${researcherId}`} from requesting access to your institution's datasets?\n\n` +
      `This only affects YOUR institution; other hospitals are unaffected. Their pending requests to you will be withdrawn.\n\n` +
      `Optional reason (kept internal, never shown to the researcher):`
    )
    if (reason === null) return

    setError('')
    setSuccess('')
    try {
      const res = await fetch(`${API_BASE}/hospitals/${institutionId}/researchers/${researcherId}/block`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blocked_by_user_id: Number(approverId), reason: reason || null }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(safeDetail(body, 'Unable to block this researcher. Please try again.'))
        return
      }
      setSuccess(`Researcher #${researcherId} blocked. Their pending requests to your institution were withdrawn.`)
      await Promise.all([fetchRequests(), fetchBlocked()])
    } catch {
      setError(NETWORK_ERROR)
    }
  }

  const unblockResearcher = async (researcherId) => {
    if (!window.confirm(`Unblock researcher #${researcherId}? They will be able to request access again.`)) return
    setError('')
    setSuccess('')
    try {
      const res = await fetch(`${API_BASE}/hospitals/${institutionId}/researchers/${researcherId}/block`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unblocked_by_user_id: Number(approverId) }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(safeDetail(body, 'Unable to unblock this researcher. Please try again.'))
        return
      }
      setSuccess(`Researcher #${researcherId} unblocked.`)
      await fetchBlocked()
    } catch {
      setError(NETWORK_ERROR)
    }
  }

  const markAllNotificationsRead = async () => {
    try {
      await Promise.all(
        notifications.map((n) =>
          fetch(`${API_BASE}/notifications/${n.id}/read?user_id=${approverId}`, { method: 'POST' })
        )
      )
      await fetchNotifications()
    } catch {
      setError(NETWORK_ERROR)
    }
  }

  return (
    <div className="hospital-page">
      <header className="hd-page-header">
        <div>
          <h1 className="hd-title">{institution || 'Hospital'} — Access Requests</h1>
          <p className="hd-muted">
            Signed in as {username || 'Hospital admin'} · Institution #{institutionId} · User #{approverId}
          </p>
        </div>
        <button className="hd-btn" onClick={() => navigate('/login')}>Change user</button>
      </header>

      {notifications.length > 0 && (
        <section className="hd-panel hd-notice" aria-live="polite">
          <div className="hd-notice-row">
            <strong>{notifications.length} new notification{notifications.length === 1 ? '' : 's'}</strong>
            <button className="hd-btn" onClick={markAllNotificationsRead}>Mark all as read</button>
          </div>
          <ul className="hd-notice-list">
            {notifications.slice(0, 3).map((n) => (
              <li key={n.id}>{n.message}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="hd-panel">
        <nav className="hd-tabs" aria-label="Request status">
          {TABS.map((tab) => (
            <button
              key={tab}
              className={`hd-tab ${activeTab === tab ? 'hd-tab-active' : ''}`}
              onClick={() => { setActiveTab(tab); setLoading(true) }}
            >
              {tab[0].toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>

        {error && <div className="hd-message hd-error" role="alert">{error}</div>}
        {success && <div className="hd-message hd-success" role="status">{success}</div>}

        {activeTab === 'blocked' ? (
          blockedResearchers.length === 0 ? (
            <p className="hd-muted">No researchers are currently blocked by your institution.</p>
          ) : (
            blockedResearchers.map((b) => (
              <article className="hd-card" key={b.block_id}>
                <header className="hd-card-header">
                  <h3 className="hd-card-title">{b.researcher_email || `Researcher #${b.researcher_id}`}</h3>
                  <span className="hd-status hd-status-blocked">blocked</span>
                </header>
                <dl className="hd-grid">
                  <div className="hd-field">
                    <dt>Blocked since</dt>
                    <dd>{b.blocked_at ? new Date(b.blocked_at).toLocaleString() : '—'}</dd>
                  </div>
                  <div className="hd-field">
                    <dt>Internal reason</dt>
                    <dd>{b.reason || '—'}</dd>
                  </div>
                </dl>
                <footer className="hd-actions">
                  <button className="hd-btn" onClick={() => unblockResearcher(b.researcher_id)}>Unblock</button>
                </footer>
              </article>
            ))
          )
        ) : loading ? (
          <p className="hd-muted">Loading requests…</p>
        ) : requests.length === 0 ? (
          <p className="hd-muted">No {activeTab} requests for this institution.</p>
        ) : (
          requests.map((r) => (
            <RequestCard
              key={r.request_id}
              r={r}
              showDecisionActions={activeTab === 'pending'}
              onApprove={(id) => decide(id, 'approved')}
              onReject={(id) => decide(id, 'rejected')}
              onBlock={blockResearcher}
            />
          ))
        )}
      </section>
    </div>
  )
}

export default HospitalDashboard
