import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'
import LiveTrainingPanel from './LiveTrainingPanel'
import './ResearcherDashboard.css'

const API_BASE = 'http://localhost:8000'
const POLL_INTERVAL_MS = 4000

const ResearcherDashboard = () => {
  const navigate = useNavigate()
  const { username, institution, userId } = useAppContext()
  const requesterId = userId

  const [datasets, setDatasets] = useState([])
  const [error, setError] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [inProgress, setInProgress] = useState(false)

  // A researcher can only ever have ONE active project (draft/pending/
  // active/running) at a time -- enforced by the backend (a Postgres
  // partial unique index, not just a frontend check: see
  // db/models.py's ResearchProject). This drives the entire UI: if one
  // exists, the "start a new project" form is replaced by that project's
  // status, with an explanation of why a second can't be started yet.
  const [activeProject, setActiveProject] = useState(null)
  const [myProjects, setMyProjects] = useState([])

  // --- New project form state ---------------------------------------
  const [topic, setTopic] = useState('')
  const [purposeDetail, setPurposeDetail] = useState('')
  const [hospitalScope, setHospitalScope] = useState('all_eligible') // 'specific' | 'multiple' | 'all_eligible'
  const [eligibleHospitals, setEligibleHospitals] = useState([])
  const [selectedHospitalIds, setSelectedHospitalIds] = useState([])

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [flConfig, setFlConfig] = useState({
    fl_rounds: 5,
    fedprox_mu: 0.05,
    dp_enabled: false,
    dp_clip_norm: 1.0,
    dp_noise_multiplier: 1.0,
    dp_delta: 0.00001,
    aggregation_strategy: 'fedavg',
    secagg_enabled: false,
  })
  const configConflict = flConfig.secagg_enabled && flConfig.aggregation_strategy === 'coordinate_median'

  useEffect(() => {
    if (!requesterId) {
      navigate('/login')
    }
  }, [requesterId, navigate])

  useEffect(() => {
    fetch(`${API_BASE}/datasets`)
      .then(res => res.json())
      .then(setDatasets)
      .catch(() => setError('Failed to load datasets'))
  }, [])

  const fetchProjects = useCallback(async () => {
    if (!requesterId) return
    try {
      const [activeRes, listRes] = await Promise.all([
        fetch(`${API_BASE}/research-projects/active?researcher_id=${requesterId}`),
        fetch(`${API_BASE}/research-projects?researcher_id=${requesterId}`),
      ])
      if (activeRes.ok) {
        const activeData = await activeRes.json()
        setActiveProject(activeData)
      }
      if (listRes.ok) {
        setMyProjects(await listRes.json())
      }
    } catch {
      setError('Failed to load your research projects')
    }
  }, [requesterId])

  useEffect(() => {
    fetchProjects()
    if (!requesterId) return undefined
    const interval = setInterval(fetchProjects, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [requesterId, fetchProjects])

  // Suggestion chips derived from the real datasets' disease codes (not a
  // hardcoded, unrelated list) -- clicking one sets the research topic,
  // which drives BOTH the suggestion text and the eligible-hospitals
  // lookup below.
  const diseaseChips = useMemo(() => {
    const seen = new Set()
    const chips = []
    for (const ds of datasets) {
      const label = (ds.disease_code || ds.name).replace(/_/g, ' ')
      if (!seen.has(label)) {
        seen.add(label)
        chips.push(label)
      }
    }
    return chips
  }, [datasets])

  const filteredChips = useMemo(() => {
    if (!topic) return diseaseChips
    return diseaseChips.filter(d => d.toLowerCase().includes(topic.toLowerCase()))
  }, [topic, diseaseChips])

  // Fetches which hospitals are actually eligible for the CURRENT topic --
  // "eligible" already excludes any hospital that has blocked this
  // researcher (the researcher is never told that's why; a blocked
  // hospital just never shows up as selectable at all).
  useEffect(() => {
    if (!topic || !requesterId) {
      setEligibleHospitals([])
      return
    }
    const controller = new AbortController()
    fetch(`${API_BASE}/research-projects/eligible-hospitals?topic=${encodeURIComponent(topic)}&researcher_id=${requesterId}`, {
      signal: controller.signal,
    })
      .then(res => (res.ok ? res.json() : []))
      .then(setEligibleHospitals)
      .catch(() => {})
    return () => controller.abort()
  }, [topic, requesterId])

  useEffect(() => {
    // Keep the checkbox selection valid as eligibility changes with the topic.
    setSelectedHospitalIds(prev => prev.filter(id => eligibleHospitals.some(h => h.institution_id === id)))
  }, [eligibleHospitals])

  const toggleHospital = (institutionId) => {
    setSelectedHospitalIds(prev => {
      if (hospitalScope === 'specific') {
        return prev.includes(institutionId) ? [] : [institutionId]
      }
      return prev.includes(institutionId)
        ? prev.filter(id => id !== institutionId)
        : [...prev, institutionId]
    })
  }

  const canSubmit = topic && requesterId && !activeProject && !configConflict && (
    hospitalScope === 'all_eligible'
      ? eligibleHospitals.length > 0
      : selectedHospitalIds.length > 0
  )

  const submitProject = async () => {
    if (!canSubmit) return
    setInProgress(true)
    setError('')
    setStatusMessage('')

    try {
      const res = await fetch(`${API_BASE}/research-projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          researcher_id: Number(requesterId),
          topic,
          purpose: purposeDetail || topic,
          hospital_scope: hospitalScope,
          hospital_ids: hospitalScope === 'all_eligible' ? null : selectedHospitalIds,
          fl_rounds: Number(flConfig.fl_rounds),
          fedprox_mu: Number(flConfig.fedprox_mu),
          dp_enabled: flConfig.dp_enabled,
          dp_clip_norm: Number(flConfig.dp_clip_norm),
          dp_noise_multiplier: Number(flConfig.dp_noise_multiplier),
          dp_delta: Number(flConfig.dp_delta),
          aggregation_strategy: flConfig.aggregation_strategy,
          secagg_enabled: flConfig.secagg_enabled,
        }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(body.detail || 'Failed to create research project')
        return
      }
      setStatusMessage(
        `Project #${body.project_id} created with ${body.hospital_requests.length} hospital request(s). ` +
        'Each hospital reviews independently.'
      )
      setTopic('')
      setPurposeDetail('')
      setSelectedHospitalIds([])
      await fetchProjects()
    } catch {
      setError('Network error while creating the research project')
    } finally {
      setInProgress(false)
    }
  }

  const cancelActiveProject = async () => {
    if (!activeProject) return
    if (!window.confirm(`Cancel project "${activeProject.topic}"? Pending hospital requests will be withdrawn.`)) {
      return
    }
    try {
      const res = await fetch(
        `${API_BASE}/research-projects/${activeProject.project_id}/cancel?researcher_id=${requesterId}`,
        { method: 'POST' }
      )
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail || 'Failed to cancel project')
        return
      }
      setStatusMessage('Project cancelled. You can now start a new one.')
      await fetchProjects()
    } catch {
      setError('Network error while cancelling the project')
    }
  }

  const requestAllRemainingEligible = async () => {
    if (!activeProject) return
    try {
      const res = await fetch(
        `${API_BASE}/research-projects/${activeProject.project_id}/request-all-eligible-hospitals?researcher_id=${requesterId}`,
        { method: 'POST' }
      )
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(body.detail || 'Failed to add hospitals')
        return
      }
      setStatusMessage(`Added ${body.added.length} more hospital request(s) to this project.`)
      await fetchProjects()
    } catch {
      setError('Network error while adding hospitals')
    }
  }

  return (
    <div className="researcher-page">
      <div className="glimmer-overlay" />
      <header className="researcher-header">
        <div className="title-wrap">
          <h1>Welcome, Researcher{username ? ` ${username}` : ''} from {institution || '—'}</h1>
          <p className="subtitle">Global Federated Research Network</p>
        </div>
        <button className="link-btn" onClick={() => navigate('/login')}>← Change user</button>
      </header>

      <main className="researcher-main">
        <section className="search-card">
          <label className="search-label">Your Researcher ID</label>
          <p className="muted">#{requesterId} (from your login session)</p>
        </section>

        {error && <div className="error-box">{error}</div>}
        {statusMessage && <div className="info-box">{statusMessage}</div>}

        {activeProject ? (
          <section className="progress-card active-project-card">
            <h3>Active Research Project</h3>
            <p>
              <strong>{activeProject.topic}</strong>{' '}
              <span className={`status-badge status-${activeProject.status}`}>{activeProject.status}</span>
            </p>
            <p className="muted">
              You have one active project at a time. Complete, cancel, or wait for a decision on
              every hospital below before starting a new one.
            </p>
            <table className="requests-table">
              <thead>
                <tr><th>Hospital</th><th>Status</th></tr>
              </thead>
              <tbody>
                {activeProject.hospital_requests.map(hr => (
                  <tr key={hr.request_id}>
                    <td>{hr.institution_name || `#${hr.institution_id}`}</td>
                    <td><span className={`status-badge status-${hr.status}`}>{hr.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="project-actions">
              <button className="link-btn" onClick={requestAllRemainingEligible}>
                + Request remaining eligible hospitals
              </button>
              <button className="link-btn danger" onClick={cancelActiveProject}>
                Cancel this project
              </button>
            </div>
          </section>
        ) : (
          <section className="search-card">
            <label htmlFor="topic" className="search-label">Research topic / disease</label>
            <div className="search-row">
              <input
                id="topic"
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. COVID19, Lung Cancer, Diabetes"
                className="search-input"
              />
            </div>
            {filteredChips.length > 0 && (
              <div className="suggestions">
                {filteredChips.map(d => (
                  <button key={d} className="chip" onClick={() => setTopic(d)}>{d}</button>
                ))}
              </div>
            )}

            <label className="search-label" style={{ marginTop: 14 }}>Research purpose (optional detail)</label>
            <textarea
              className="search-input"
              rows={2}
              value={purposeDetail}
              onChange={e => setPurposeDetail(e.target.value)}
              placeholder="What is this research for? (shown to approving hospitals)"
            />

            <label className="search-label" style={{ marginTop: 14 }}>Hospital scope</label>
            <div className="hospital-scope-row">
              <label className="radio-row">
                <input type="radio" name="scope" checked={hospitalScope === 'specific'}
                  onChange={() => { setHospitalScope('specific'); setSelectedHospitalIds([]) }} />
                Specific hospital (choose one)
              </label>
              <label className="radio-row">
                <input type="radio" name="scope" checked={hospitalScope === 'multiple'}
                  onChange={() => { setHospitalScope('multiple'); setSelectedHospitalIds([]) }} />
                Multiple selected hospitals
              </label>
              <label className="radio-row">
                <input type="radio" name="scope" checked={hospitalScope === 'all_eligible'}
                  onChange={() => { setHospitalScope('all_eligible'); setSelectedHospitalIds([]) }} />
                All eligible hospitals
              </label>
            </div>

            {topic && hospitalScope !== 'all_eligible' && (
              <div className="hospital-checklist">
                {eligibleHospitals.length === 0 && (
                  <p className="muted">No eligible hospitals found for this topic yet.</p>
                )}
                {eligibleHospitals.map(h => (
                  <label key={h.institution_id} className="checkbox-row">
                    <input
                      type={hospitalScope === 'specific' ? 'radio' : 'checkbox'}
                      checked={selectedHospitalIds.includes(h.institution_id)}
                      onChange={() => toggleHospital(h.institution_id)}
                    />
                    {h.institution_name} — {h.dataset_name}
                  </label>
                ))}
              </div>
            )}
            {topic && hospitalScope === 'all_eligible' && (
              <p className="muted">
                {eligibleHospitals.length > 0
                  ? `Will request from all ${eligibleHospitals.length} currently eligible hospital(s): ${eligibleHospitals.map(h => h.institution_name).join(', ')}. Each decides independently.`
                  : 'No eligible hospitals found for this topic yet.'}
              </p>
            )}

            <button
              type="button"
              className="link-btn fl-config-toggle"
              onClick={() => setShowAdvanced(v => !v)}
            >
              {showAdvanced ? '▾' : '▸'} FL protocol settings (rounds, privacy, robustness)
            </button>

            {showAdvanced && (
              <div className="fl-config-panel">
                <p className="muted">
                  This is your proposed training protocol -- every hospital you request sees these
                  settings before approving, and they're what actually configures the
                  federated experiment once approved.
                </p>

                <div className="fl-config-grid">
                  <label>Rounds
                    <input type="number" min="1" max="50" value={flConfig.fl_rounds}
                      onChange={e => setFlConfig({ ...flConfig, fl_rounds: e.target.value })} />
                  </label>
                  <label>FedProx μ <span className="hint">(0 = plain FedAvg; helps with non-IID data)</span>
                    <input type="number" min="0" max="1" step="0.01" value={flConfig.fedprox_mu}
                      onChange={e => setFlConfig({ ...flConfig, fedprox_mu: e.target.value })} />
                  </label>
                </div>

                <div className="fl-config-section">
                  <label className="checkbox-row">
                    <input type="checkbox" checked={flConfig.dp_enabled}
                      onChange={e => setFlConfig({ ...flConfig, dp_enabled: e.target.checked })} />
                    Differential privacy (DP-SGD) -- adds calibrated noise so no single
                    patient record can be reverse-engineered from the released model updates
                  </label>
                  {flConfig.dp_enabled && (
                    <div className="fl-config-grid indented">
                      <label>Clip norm
                        <input type="number" min="0.01" step="0.1" value={flConfig.dp_clip_norm}
                          onChange={e => setFlConfig({ ...flConfig, dp_clip_norm: e.target.value })} />
                      </label>
                      <label>Noise multiplier <span className="hint">(higher = more private, less accurate)</span>
                        <input type="number" min="0" step="0.1" value={flConfig.dp_noise_multiplier}
                          onChange={e => setFlConfig({ ...flConfig, dp_noise_multiplier: e.target.value })} />
                      </label>
                      <label>δ (delta)
                        <input type="number" min="0" step="0.00001" value={flConfig.dp_delta}
                          onChange={e => setFlConfig({ ...flConfig, dp_delta: e.target.value })} />
                      </label>
                    </div>
                  )}
                </div>

                <div className="fl-config-section">
                  <label>Aggregation strategy
                    <select value={flConfig.aggregation_strategy}
                      onChange={e => setFlConfig({ ...flConfig, aggregation_strategy: e.target.value })}>
                      <option value="fedavg">FedAvg (weighted average)</option>
                      <option value="coordinate_median">Coordinate-median (robust to a malicious hospital)</option>
                    </select>
                  </label>
                </div>

                <div className="fl-config-section">
                  <label className="checkbox-row">
                    <input type="checkbox" checked={flConfig.secagg_enabled}
                      onChange={e => setFlConfig({ ...flConfig, secagg_enabled: e.target.checked })} />
                    Secure Aggregation -- the coordinating server only ever sees the combined
                    total across hospitals, never any single hospital's update on its own
                  </label>
                </div>

                {configConflict && (
                  <div className="error-box">
                    Secure Aggregation and coordinate-median aggregation can't both be on:
                    median needs to see individual updates, SecAgg exists to hide them.
                  </div>
                )}
              </div>
            )}

            <button
              className="primary-btn"
              style={{ marginTop: 14 }}
              onClick={submitProject}
              disabled={inProgress || !canSubmit}
            >
              {inProgress ? 'Submitting...' : 'Submit Research Project'}
            </button>
          </section>
        )}

        <section className="progress-card">
          <h3>Your Research Projects</h3>
          {myProjects.length === 0 && <p className="muted">No projects yet.</p>}
          {myProjects.length > 0 && myProjects.map(p => (
            <div key={p.project_id} className="project-history-row">
              <span className={`status-badge status-${p.status}`}>{p.status}</span>
              <strong>{p.topic}</strong>
              <span className="muted">
                {p.hospital_requests.map(hr => `${hr.institution_name || '#' + hr.institution_id}: ${hr.status}`).join(' · ')}
              </span>
            </div>
          ))}
        </section>

        <LiveTrainingPanel requesterId={requesterId} />
      </main>
    </div>
  )
}

export default ResearcherDashboard
