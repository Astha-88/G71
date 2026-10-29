import { createContext, useContext, useMemo, useState, useEffect } from 'react'

const AppContext = createContext(null)

export const AppProvider = ({ children }) => {
  const [username, setUsername] = useState('')
  const [role, setRole] = useState('Researcher')
  const [institution, setInstitution] = useState('')
  const [selectedDisease, setSelectedDisease] = useState('')

  // Previously the dashboards asked the person to type their numeric
  // requester_id / institution_id into a text box by hand, with nothing
  // tying it back to who actually logged in. These two fields hold the ids
  // the FastAPI backend returned at login/register, so every subsequent
  // API call is scoped to the real authenticated identity instead of
  // whatever number happened to be typed in.
  const [userId, setUserId] = useState(null)
  const [institutionId, setInstitutionId] = useState(null)

  useEffect(() => {
    try {
      const saved = sessionStorage.getItem('vienko_app_state')
      if (saved) {
        const parsed = JSON.parse(saved)
        setUsername(parsed.username || '')
        setRole(parsed.role || 'Researcher')
        setInstitution(parsed.institution || '')
        setSelectedDisease(parsed.selectedDisease || '')
        setUserId(parsed.userId ?? null)
        setInstitutionId(parsed.institutionId ?? null)
      }
    } catch (_) {}
  }, [])

  useEffect(() => {
    const state = { username, role, institution, selectedDisease, userId, institutionId }
    try {
      sessionStorage.setItem('vienko_app_state', JSON.stringify(state))
    } catch (_) {}
  }, [username, role, institution, selectedDisease, userId, institutionId])

  const logoutLocal = () => {
    setUsername('')
    setRole('Researcher')
    setInstitution('')
    setUserId(null)
    setInstitutionId(null)
    try {
      sessionStorage.removeItem('vienko_app_state')
    } catch (_) {}
  }

  const value = useMemo(() => ({
    username,
    role,
    institution,
    selectedDisease,
    userId,
    institutionId,
    setUsername,
    setRole,
    setInstitution,
    setSelectedDisease,
    setUserId,
    setInstitutionId,
    logoutLocal,
  }), [username, role, institution, selectedDisease, userId, institutionId])

  return (
    <AppContext.Provider value={value}>{children}</AppContext.Provider>
  )
}

export const useAppContext = () => {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useAppContext must be used within AppProvider')
  return ctx
}

export default AppContext
