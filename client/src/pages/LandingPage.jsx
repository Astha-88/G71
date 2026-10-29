import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useAuth from '../hooks/useAuth'
import './LandingPage.css'

const LandingPage = () => {
  const navigate = useNavigate()
  const [isLoading, setIsLoading] = useState(false)
  const { user, loading: authLoading, logout } = useAuth()

  const handleLogin = () => {
    setIsLoading(true)
    setTimeout(() => {
      navigate('/login')
      setIsLoading(false)
    }, 300)
  }

  return (
    <div className="landing-page">
      {/* Header - brand left, professional login button right */}
      <header className="header">
        <div className="brand">
          <img src="/images/vienk.png" alt="Vienkō" className="brand-logo" />
        </div>
        <div className="header-actions">
          {user ? (
            <div className="user-info">
              <img src={user.picture} alt={user.name} className="user-avatar" />
              <span className="user-name">{user.name}</span>
              <button className="logout-btn" onClick={logout}>
                Logout
              </button>
            </div>
          ) : (
            <button 
              className="btn btn-primary" 
              onClick={handleLogin}
              disabled={isLoading || authLoading}
            >
              {isLoading ? 'Loading…' : 'Login'}
            </button>
          )}
        </div>
      </header>

      {/* Sidebar removed per request */}

      {/* Background visuals */}
      <div className="hero-background" />
      <div className="glimmer-effect" />

      {/* Two-column hero (no scrolling) */}
      <main className="hero">
        <section className="hero-left">
          <div className="hero-logo-container">
            <img src="/images/main.png" alt="Heart" className="hero-heart" />
            <h1 className="hero-title">
              <span className="typewriter-text">vienkō</span>
              <span className="cursor">|</span>
            </h1>
          </div>
          <p className="hero-subtitle">Precision. Presence. Personal health, beautifully visualized.</p>
        </section>
        <section className="hero-right">
          <img src="/images/main.png" alt="Vienkō main illustration" className="hero-main-image" />
        </section>
      </main>
    </div>
  )
}

export default LandingPage
