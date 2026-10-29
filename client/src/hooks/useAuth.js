import { useState, useEffect } from 'react'

const useAuth = () => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/user', {
        credentials: 'include'
      })
      const data = await response.json()
      
      if (data.success) {
        setUser(data.user)
      } else {
        setUser(null)
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      setUser(null)
    } finally {
      setLoading(false)
    }
  }

  const logout = async () => {
    try {
      await fetch('http://localhost:5000/auth/logout', {
        credentials: 'include'
      })
      setUser(null)
      window.location.href = '/'
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  return { user, loading, logout, checkAuthStatus }
}

export default useAuth
