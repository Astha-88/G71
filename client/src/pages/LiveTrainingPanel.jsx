import { useEffect, useState } from 'react'
import { useExperimentSocket } from '../hooks/useExperimentSocket'
import './LiveTrainingPanel.css'

const API_BASE = 'http://localhost:8000'
const POLL_INTERVAL_MS = 4000

const emptyPatient = {
  age: 60,
  gender: 'F',
  fever: 1,
  cough: 1,
  oxygen_level: 91,
  blood_pressure: 130,
}

const LiveTrainingPanel = ({ requesterId }) => {
  const [experiments, setExperiments] = useState([])
  const [watchingId, setWatchingId] = useState(null)
  const [patient, setPatient] = useState(emptyPatient)
  const [prediction, setPrediction] = useState(null)
  const [predictError, setPredictError] = useState('')

  const { rounds, status } = useExperimentSocket(watchingId)

  const fetchExperiments = async () => {
    if (!requesterId) return
    try {
      const res = await fetch(`${API_BASE}/experiments?requester_id=${requesterId}`)
      if (!res.ok) return
      const data = await res.json()
      setExperiments(data)
    } catch {
      // silent -- this panel is supplementary, not the primary workflow
    }
  }

  useEffect(() => {
    fetchExperiments()
    const interval = setInterval(fetchExperiments, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [requesterId])

  const watched = experiments.find((e) => e.id === watchingId)
  const latestRound = rounds[rounds.length - 1]

  const runPrediction = async () => {
    setPredictError('')
    setPrediction(null)
    try {
      const res = await fetch(`${API_BASE}/experiments/${watchingId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patient),
      })
      const body = await res.json()
      if (!res.ok) {
        setPredictError(body.detail || 'Prediction failed')
        return
      }
      setPrediction(body)
    } catch {
      setPredictError('Network error while predicting')
    }
  }

  if (experiments.length === 0) {
    return (
      <section className="progress-card live-training-card">
        <h3>Federated Training Runs</h3>
        <p className="muted">No experiments yet -- these appear automatically once a hospital approves one of your requests.</p>
      </section>
    )
  }

  return (
    <section className="progress-card live-training-card">
      <h3>Federated Training Runs</h3>
      <ul className="experiment-list">
        {experiments.map((exp) => (
          <li key={exp.id} className={exp.id === watchingId ? 'selected' : ''}>
            <button className="link-btn" onClick={() => setWatchingId(exp.id)}>
              #{exp.id} {exp.name} — <span className={`status-badge status-${exp.status}`}>{exp.status}</span>
            </button>
          </li>
        ))}
      </ul>

      {watchingId && (
        <div className="live-view">
          <div className="live-view-header">
            <h4>Watching experiment #{watchingId}{watched ? ` (${watched.status})` : ''}</h4>
            <span className={`ws-status ws-${status}`}>{status}</span>
          </div>

          {rounds.length === 0 && (
            <p className="muted">
              Waiting for the first round to complete... (three real collaborator
              processes are training locally right now, over real HTTP)
            </p>
          )}

          {rounds.length > 0 && (
            <>
              {latestRound?.dp_enabled && (
                <div className="dp-badge">
                  🔒 Differential privacy on — ε≈
                  {(() => {
                    const epsilons = (latestRound.per_client || [])
                      .map((c) => c.dp?.epsilon_spent)
                      .filter((e) => e != null)
                    return epsilons.length ? Math.max(...epsilons).toFixed(2) : '—'
                  })()}
                  {' '}(worst-case across hospitals, δ=1e-5)
                </div>
              )}
              <table className="rounds-table">
                <thead>
                  <tr><th>Round</th><th>Loss</th><th>Accuracy</th><th>AUC</th></tr>
                </thead>
                <tbody>
                  {rounds.map((r) => (
                    <tr key={r.round}>
                      <td>{r.round}</td>
                      <td>{r.loss?.toFixed(4)}</td>
                      <td>{(r.accuracy * 100)?.toFixed(1)}%</td>
                      <td>{r.auc?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="loss-bars" aria-hidden="true">
                {rounds.map((r) => (
                  <div
                    key={r.round}
                    className="loss-bar"
                    style={{ height: `${Math.min(100, r.loss * 100)}%` }}
                    title={`Round ${r.round}: loss ${r.loss.toFixed(4)}`}
                  />
                ))}
              </div>
            </>
          )}

          {latestRound && (
            <div className="predict-form">
              <h4>Try the current global model</h4>
              <p className="muted">Real-time inference against the model as of the latest completed round.</p>
              <div className="predict-grid">
                <label>Age
                  <input type="number" value={patient.age}
                    onChange={e => setPatient({ ...patient, age: e.target.value })} />
                </label>
                <label>Gender
                  <select value={patient.gender} onChange={e => setPatient({ ...patient, gender: e.target.value })}>
                    <option value="F">F</option>
                    <option value="M">M</option>
                  </select>
                </label>
                <label>Fever
                  <select value={patient.fever} onChange={e => setPatient({ ...patient, fever: Number(e.target.value) })}>
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </label>
                <label>Cough
                  <select value={patient.cough} onChange={e => setPatient({ ...patient, cough: Number(e.target.value) })}>
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </label>
                <label>Oxygen level
                  <input type="number" value={patient.oxygen_level}
                    onChange={e => setPatient({ ...patient, oxygen_level: e.target.value })} />
                </label>
                <label>Blood pressure
                  <input type="number" value={patient.blood_pressure}
                    onChange={e => setPatient({ ...patient, blood_pressure: e.target.value })} />
                </label>
              </div>
              <button className="primary-btn" onClick={runPrediction} disabled={watched?.status !== 'completed'}>
                {watched?.status === 'completed' ? 'Predict' : 'Model not finalized yet'}
              </button>
              {predictError && <div className="error-box">{predictError}</div>}
              {prediction && (
                <div className="success-box">
                  Predicted hospitalization risk: <strong>{(prediction.hospitalization_risk * 100).toFixed(1)}%</strong>
                  {' '}— recommendation: <strong>{prediction.recommendation}</strong>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default LiveTrainingPanel
