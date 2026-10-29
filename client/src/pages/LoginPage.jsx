import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'
import './LoginPage.css'

const API_BASE = 'http://localhost:8000'

// The dashboards' role field ("Researcher" / "Hospital") is the UI label;
// the backend's User.role column expects "researcher" / "hospital_admin".
// Keeping the mapping in one place avoids the two drifting apart again.
const ROLE_TO_BACKEND = {
  Researcher: 'researcher',
  Hospital: 'hospital_admin',
}

const LoginPage = () => {
  const navigate = useNavigate()
  const { setUsername, setRole, setInstitution, setUserId, setInstitutionId } = useAppContext()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [institutions, setInstitutions] = useState([])
  const [formData, setFormData] = useState({
    email: '',
    name: '',
    role: 'Researcher',
    institutionId: '',
  })

  // Institution.type is seeded as "hospital" or "research" (see
  // scripts/seed_data.py). Previously the dropdown showed every
  // institution regardless of role -- a researcher could "belong to" a
  // hospital and vice versa, which doesn't match how institutions and
  // roles actually relate in this system (hospital_admins approve
  // requests FOR their own hospital's datasets; researchers request
  // access FROM a research institution).
  const relevantType = formData.role === 'Hospital' ? 'hospital' : 'research'
  const filteredInstitutions = institutions.filter((inst) => inst.type === relevantType)

  useEffect(() => {
    // Populate the picker from what actually exists in the DB (previously
    // this was a hardcoded list of institution names that didn't match any
    // institution seed_data.py actually creates).
    fetch(`${API_BASE}/institutions`)
      .then((res) => res.json())
      .then((data) => {
        setInstitutions(data)
      })
      .catch(() => setError('Could not reach the backend to load institutions.'))
  }, [])

  useEffect(() => {
    // Whenever the role changes (including the initial load), make sure
    // the selected institution is actually a valid one FOR that role --
    // switching from Researcher to Hospital shouldn't leave a research
    // institute's id silently selected underneath an unfiltered dropdown.
    const stillValid = filteredInstitutions.some(
      (inst) => String(inst.id) === formData.institutionId
    )
    if (!stillValid && filteredInstitutions.length > 0) {
      setFormData((prev) => ({ ...prev, institutionId: String(filteredInstitutions[0].id) }))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [formData.role, institutions])

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')
    try {
      const backendRole = ROLE_TO_BACKEND[formData.role]
      const institutionId = formData.institutionId ? Number(formData.institutionId) : null

      // Prototype auth: no password hashing exists in the schema yet (see
      // IMPLEMENTATION_ROADMAP.md), so this looks the user up by email and
      // registers them on first login. What matters for the rest of the
      // app is that requester_id / institution_id now come from a real
      // backend row instead of being typed into a text box on the
      // dashboard with nothing behind it.
      let res = await fetch(`${API_BASE}/users/by-email?email=${encodeURIComponent(formData.email)}`)

      let user
      if (res.status === 404) {
        res = await fetch(`${API_BASE}/users/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: formData.email,
            name: formData.name || formData.email.split('@')[0],
            role: backendRole,
            institution_id: institutionId,
          }),
        })
        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          throw new Error(body.detail || 'Registration failed')
        }
        user = await res.json()
      } else if (res.ok) {
        user = await res.json()
      } else {
        throw new Error('Login failed')
      }

      setUsername(user.name)
      setRole(formData.role)
      setUserId(user.id)
      setInstitutionId(user.institution_id)
      const inst = institutions.find((i) => i.id === user.institution_id)
      setInstitution(inst ? inst.name : '')

      navigate(formData.role === 'Researcher' ? '/researcher' : '/hospital')
    } catch (err) {
      console.error('Login error:', err)
      setError(err.message || 'Something went wrong. Is the backend running on :8000?')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="glimmer-overlay" />
      <div className="login-container">
        <div className="login-header">
          <img src="/images/vienk.png" alt="Vienkō" className="login-logo" />
          <h1 className="login-title">Global Federated Research Network</h1>
          <p className="login-subtitle">Prototype login — looks up or creates your user via the backend</p>
        </div>

        <div className="login-form-container">
          <form className="login-form" onSubmit={handleLogin}>
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                required
                placeholder="you@example.com"
              />
            </div>

            <div className="form-group">
              <label htmlFor="name">Name (used only if this is your first login)</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Jane Doe"
              />
            </div>

            <div className="form-group">
              <label htmlFor="role">Role</label>
              <select id="role" name="role" value={formData.role} onChange={handleInputChange}>
                <option>Researcher</option>
                <option>Hospital</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="institutionId">Institution</label>
              <select
                id="institutionId"
                name="institutionId"
                value={formData.institutionId}
                onChange={handleInputChange}
              >
                {filteredInstitutions.map((inst) => (
                  <option key={inst.id} value={inst.id}>
                    {inst.name}
                  </option>
                ))}
              </select>
            </div>

            {error && <p className="login-error">{error}</p>}

            <button
              type="submit"
              className="login-submit-btn"
              disabled={isLoading}
            >
              {isLoading ? 'Signing in...' : 'Login'}
            </button>
          </form>

          <button
            className="back-btn"
            onClick={() => navigate('/')}
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
