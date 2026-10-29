import { useEffect, useRef, useState } from 'react'

const WS_BASE = 'ws://localhost:8000'

// Connects to /experiments/{id}/ws and accumulates each round_complete
// message as it arrives. This replaces polling GET /rounds on a timer for
// anyone actively watching a run -- the socket only exists while a
// component is mounted with a live experimentId, and cleans itself up on
// unmount or when the experiment finishes.
export function useExperimentSocket(experimentId) {
  const [rounds, setRounds] = useState([])
  const [status, setStatus] = useState('idle')
  const socketRef = useRef(null)

  useEffect(() => {
    setRounds([])
    if (!experimentId) {
      setStatus('idle')
      return undefined
    }

    setStatus('connecting')
    const ws = new WebSocket(`${WS_BASE}/experiments/${experimentId}/ws`)
    socketRef.current = ws

    ws.onopen = () => setStatus('watching')

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'round_complete') {
          setRounds((prev) => [...prev, { round: msg.round, ...msg.metrics }])
        } else if (msg.type === 'status') {
          setStatus(msg.status)
        }
      } catch {
        // ignore malformed frame
      }
    }

    ws.onerror = () => setStatus('error')
    ws.onclose = () => setStatus((s) => (s === 'error' ? s : 'closed'))

    return () => {
      ws.close()
      socketRef.current = null
    }
  }, [experimentId])

  return { rounds, status }
}
