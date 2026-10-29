import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import ResearcherDashboard from './pages/ResearcherDashboard'
import HospitalDashboard from './pages/HospitalDashboard'
import { AppProvider } from './context/AppContext'
import './App.css'

function App() {
  return (
    <Router>
      <AppProvider>
        <div className="App">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/researcher" element={<ResearcherDashboard />} />
            <Route path="/hospital" element={<HospitalDashboard />} />
          </Routes>
        </div>
      </AppProvider>
    </Router>
  )
}

export default App