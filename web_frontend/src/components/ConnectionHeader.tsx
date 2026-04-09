import { useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import { api } from '../api/client'

export function ConnectionHeader() {
  const q = useQuery({
    queryKey: ['connection-status'],
    queryFn: async () => (await api.get('/api/connection-status')).data,
    refetchInterval: 30_000,
  })
  if (!q.data) return null
  const d = q.data
  return (
    <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap' }}>
      <Box sx={{ flex: '1 1 200px', minWidth: 180 }}>
        {d.qdrant_ok ? (
          <Alert severity="success">{d.qdrant_message}</Alert>
        ) : (
          <Alert severity="error">{d.qdrant_message}</Alert>
        )}
      </Box>
      <Box sx={{ flex: '1 1 200px', minWidth: 180 }}>
        {d.ollama_ok ? (
          <Alert severity="success">{d.ollama_message}</Alert>
        ) : (
          <Alert severity="error">{d.ollama_message}</Alert>
        )}
      </Box>
      <Box sx={{ flex: '2 1 280px', minWidth: 220 }}>
        {d.monitor ? (
          <Alert severity="success">
            Sunucu: CPU %{Number(d.monitor.cpu_usage ?? 0).toFixed(1)} | GPU %
            {Number(d.monitor.gpu_usage ?? 0).toFixed(1)} | VRAM {d.monitor.vram_used}/
            {d.monitor.vram_total} MB
          </Alert>
        ) : (
          <Alert severity="warning">
            Sunucu: {d.monitor_error || 'Monitor yok'}
          </Alert>
        )}
      </Box>
    </Stack>
  )
}
